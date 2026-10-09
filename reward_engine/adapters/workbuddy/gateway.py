"""One campaign-wide request gate for actors, roles, and isolated official scoring."""

import argparse
import asyncio
import json
import time
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from reward_engine.adapters.workbuddy.io import (
    STREAM_TIMEOUT_S,
    auth,
    model_matches,
    read,
    role_timeout_s,
    write,
)


class Capacity:
    """Change future admission without cancelling requests already in flight."""

    def __init__(self, limit):
        self.limit, self.active = limit, 0
        self.condition = asyncio.Condition()

    async def acquire(self):
        async with self.condition:
            await self.condition.wait_for(lambda: self.active < self.limit)
            self.active += 1

    async def release(self):
        async with self.condition:
            self.active -= 1
            self.condition.notify_all()

    async def resize(self, limit):
        async with self.condition:
            self.limit = limit
            self.condition.notify_all()


def application(
    config,
    *,
    backend_key=None,
    client_factory=httpx.AsyncClient,
    retry_wait=asyncio.sleep,
    stream_timeout_s=STREAM_TIMEOUT_S,
):
    app = FastAPI()
    limits_path = Path(config["output"]) / "transport_limits.json"
    limit = (
        read(limits_path)["global_limit"]
        if limits_path.exists()
        else config["model_concurrency"]
    )
    ceiling = config.get("max_model_concurrency", 8)
    if type(ceiling) is not int or not 1 <= ceiling <= 128:
        raise ValueError("Invalid configured transport ceiling")
    if type(limit) is not int or not 1 <= limit <= ceiling:
        raise ValueError("Transport concurrency exceeds configured ceiling")
    gate = Capacity(limit)
    secret = (
        backend_key
        or dotenv_values(config["credentials_file"])[
            config.get("backend_key_env", "YIBU_API_KEY")
        ]
    )
    allowed = set(config["roles"].values())
    per_model_limit = config.get("per_model_concurrency", 2)
    if type(per_model_limit) is not int or not 1 <= per_model_limit <= ceiling:
        raise ValueError("Per-model concurrency exceeds configured ceiling")
    model_gates = {model: asyncio.Semaphore(per_model_limit) for model in allowed}
    model_inflight = {model: 0 for model in allowed}
    counters = {"inflight": 0, "peak": 0, "requests": 0}
    role_stream_timeout_s = role_timeout_s(config, stream_timeout_s)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "limit": gate.limit,
            "per_model_limit": per_model_limit,
            "model_inflight": model_inflight,
            "stream_timeout_s": stream_timeout_s,
            "role_stream_timeout_s": role_stream_timeout_s,
            **counters,
        }

    @app.post("/control/concurrency")
    async def concurrency(request: Request):
        if request.headers.get("authorization") != "Bearer " + auth(config):
            raise HTTPException(401, "Invalid campaign credential")
        body = await request.json()
        limit = body.get("global_limit") if isinstance(body, dict) else None
        if type(limit) is not int or not 1 <= limit <= ceiling:
            raise HTTPException(400, "global_limit exceeds configured ceiling")
        event = {
            "global_limit": limit,
            "previous_limit": gate.limit,
            "per_model_limit": per_model_limit,
            "updated_at": time.time(),
        }
        write(
            Path(config["output"]) / "transport_limit_events" / (uuid4().hex + ".json"),
            event,
        )
        write(limits_path, event)
        await gate.resize(limit)
        return await health()

    @app.post("/{channel}/v1/chat/completions")
    async def completion(channel: str, request: Request):
        if request.headers.get("authorization") != "Bearer " + auth(config):
            raise HTTPException(401, "Invalid campaign credential")
        if not channel.replace("-", "").replace("_", "").isalnum():
            raise HTTPException(400, "Invalid channel")
        body = await request.json()
        if body.get("model") not in allowed:
            raise HTTPException(400, "Unconfigured model")
        if channel.startswith("wb-"):
            body.setdefault("temperature", 0.95)
        root = Path(config["output"]) / (
            "official.private" if channel.startswith("grade") else "transport"
        )
        root = root / channel / uuid4().hex
        write(root / "request.json", body)
        record = {
            "channel": channel,
            "model": body["model"],
            "queued_at": time.time(),
            "status": "queued",
        }
        write(root / "receipt.json", record)
        model_gate = model_gates[body["model"]]
        await model_gate.acquire()
        try:
            await gate.acquire()
        except BaseException:
            model_gate.release()
            raise
        counters["inflight"] += 1
        model_inflight[body["model"]] += 1
        counters["peak"] = max(counters["peak"], counters["inflight"])
        counters["requests"] += 1
        started = time.monotonic()
        request_timeout_s = (
            role_stream_timeout_s if channel == "roles" else stream_timeout_s
        )
        record.update(
            status="running",
            queue_s=time.time() - record["queued_at"],
            stream_timeout_s=request_timeout_s,
        )
        write(root / "receipt.json", record)
        client = client_factory(timeout=httpx.Timeout(840, read=120), trust_env=False)
        response = None

        async def open_upstream():
            """Open a response without exposing partial content to the caller."""
            nonlocal response
            waits = (5, 15)
            for retry in range(3):
                record["attempts"] = record.get("attempts", 0) + 1
                try:
                    outgoing = client.build_request(
                        "POST",
                        config["backend_url"].rstrip("/") + "/chat/completions",
                        json=body,
                        headers={"Authorization": "Bearer " + secret},
                    )
                    response = await client.send(outgoing, stream=True)
                except (
                    httpx.ConnectError,
                    httpx.ConnectTimeout,
                    httpx.ReadTimeout,
                ) as error:
                    record.setdefault("retries", []).append(
                        {
                            "reason": type(error).__name__,
                            "attempt": record["attempts"],
                            "at": time.time(),
                        }
                    )
                    write(root / "receipt.json", record)
                    if retry == 2:
                        raise
                    await retry_wait(waits[retry])
                    continue
                if response.status_code >= 400:
                    record.setdefault("http_errors", []).append(
                        {
                            "status": response.status_code,
                            "attempt": record["attempts"],
                            "at": time.time(),
                        }
                    )
                if response.status_code in {429, 500, 502, 503, 504} and retry < 2:
                    record.setdefault("retries", []).append(
                        {
                            "reason": "upstream_http_status",
                            "status": response.status_code,
                            "attempt": record["attempts"],
                            "at": time.time(),
                        }
                    )
                    write(root / "receipt.json", record)
                    await response.aclose()
                    response = None
                    await retry_wait(waits[retry])
                    continue
                response.raise_for_status()
                return
            raise RuntimeError("Unreachable upstream retry state")

        async def close():
            if response is not None:
                await response.aclose()
            await client.aclose()
            counters["inflight"] -= 1
            model_inflight[body["model"]] -= 1
            await gate.release()
            model_gate.release()
            record["elapsed_s"] = time.monotonic() - started
            write(root / "receipt.json", record)

        try:
            # Retry only before any response body is exposed to the caller.
            await open_upstream()
            if not body.get("stream"):
                raw = await response.aread()
                (root / "response.json").write_bytes(raw)
                data = json.loads(raw)
                if data.get("model") and not model_matches(
                    config, body["model"], data["model"]
                ):
                    raise ValueError("Provider model identity differs")
                record.update(
                    status="completed",
                    usage=data.get("usage"),
                    actual_model=data.get("model"),
                )
                await close()
                return Response(raw, media_type="application/json")
        except BaseException as error:
            record.update(status="failed", error_type=type(error).__name__)
            if isinstance(error, httpx.HTTPStatusError):
                record["upstream_http_status"] = error.response.status_code
            await close()
            if isinstance(error, asyncio.CancelledError):
                raise
            raise HTTPException(
                502, "Upstream request failed; see campaign transport receipt"
            ) from None

        async def stream():
            nonlocal response
            try:
                async with asyncio.timeout(request_timeout_s):
                    for stream_attempt in range(6):
                        buffered, completed, finish, model, usage = (
                            [],
                            False,
                            None,
                            None,
                            None,
                        )
                        retry_reason = None
                        try:
                            with (root / f"response-{stream_attempt + 1}.sse").open(
                                "w"
                            ) as log:
                                async for line in response.aiter_lines():
                                    log.write(line + "\n")
                                    log.flush()
                                    buffered.append(line + "\n")
                                    if not line.startswith("data:"):
                                        continue
                                    value = line[5:].strip()
                                    if value == "[DONE]":
                                        completed = True
                                        continue
                                    if not value:
                                        continue
                                    try:
                                        event = json.loads(value)
                                    except json.JSONDecodeError:
                                        retry_reason = "malformed_stream"
                                        break
                                    if "error" in event:
                                        error = event["error"]
                                        if isinstance(error, dict):
                                            error_type = error.get("type")
                                            message = str(
                                                error.get("message", "")
                                            ).lower()
                                            if error_type == "rate_limit_error":
                                                retry_reason = "rate_limit_error"
                                                break
                                            if (
                                                error_type == "new_api_error"
                                                and "temporarily unavailable" in message
                                            ):
                                                retry_reason = (
                                                    "provider_temporarily_unavailable"
                                                )
                                                break
                                            if (
                                                error_type == "new_api_error"
                                                and "stream ended before the response was complete"
                                                in message
                                            ):
                                                retry_reason = (
                                                    "provider_incomplete_stream"
                                                )
                                                break
                                        raise RuntimeError("Provider stream error")
                                    if event.get("model"):
                                        model = event["model"]
                                        if not model_matches(
                                            config, body["model"], model
                                        ):
                                            raise ValueError(
                                                "Provider model identity differs"
                                            )
                                    usage = event.get("usage") or usage
                                    for choice in event.get("choices", []):
                                        finish = choice.get("finish_reason") or finish
                        except (httpx.ReadTimeout, httpx.RemoteProtocolError):
                            retry_reason = "interrupted_stream"
                        attempt_record = {
                            "attempt": record["attempts"],
                            "actual_model": model,
                            "usage": usage,
                            "finish_reason": finish,
                            "done": completed,
                        }
                        record.setdefault("stream_attempts", []).append(attempt_record)
                        if not retry_reason and (not completed or not finish):
                            retry_reason = "incomplete_stream"
                        if not retry_reason:
                            record.update(
                                status="completed",
                                actual_model=model,
                                finish_reason=finish,
                                usage=usage,
                            )
                            # No partial model content or tool arguments reach the runtime.
                            (root / "response.sse").write_text("".join(buffered))
                            for line in buffered:
                                yield line.encode()
                            break
                        attempt_record["retry_reason"] = retry_reason
                        if stream_attempt == 5:
                            raise RuntimeError("Provider transport retries exhausted")
                        record.setdefault("retries", []).append(
                            {"reason": retry_reason, "at": time.time()}
                        )
                        write(root / "receipt.json", record)
                        await response.aclose()
                        response = None
                        # Comments keep the client connection alive without exposing model output.
                        yield b": retrying incomplete upstream response\n\n"
                        await retry_wait(min(15 * 2**stream_attempt, 60))
                        await open_upstream()
            except BaseException as error:
                record.update(
                    status="failed",
                    error_type=type(error).__name__,
                    error_reason=str(error),
                )
                if isinstance(error, asyncio.CancelledError):
                    raise
                yield b'data: {"error":{"message":"Campaign upstream stream failed"}}\n\n'
            finally:
                await close()

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    uvicorn.run(
        application(config),
        host="0.0.0.0",
        port=config["gateway_port"],
        access_log=False,
    )


if __name__ == "__main__":
    main()
