"""One campaign-wide request gate for actors, roles, and isolated official scoring."""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from reward_engine.adapters.model_config import validate_native_grader
from reward_engine.adapters.workbuddy.io import (
    STREAM_TIMEOUT_S,
    auth,
    model_matches,
    read,
    role_timeout_s,
    write,
)


def completion_timeout(content):
    """Recognize the observed provider timeout envelope, not quoted model prose."""
    if not isinstance(content, str):
        return None
    match = re.match(
        r"\A\[req_[A-Za-z0-9_-]+\] \[[^\]\r\n]+\]\r?\n"
        r"\*\*Request exceeded ([0-9]+)s limit\*\*(?:\r?\n|$)", content
    )
    if match is None:
        return None
    return {"code": "provider_response_timeout", "type": "upstream_error",
            "limit_s": int(match.group(1)),
            "message": "Provider returned a timeout notice as completion content"}


class Capacity:
    """Share slots fairly across roles, actors and graders without cancelling work."""

    lanes = ("roles", "actor", "grading")

    def __init__(self, limit, model_limits=None):
        self.limit, self.active = limit, 0
        self.model_limits = dict(model_limits or {})
        self.model_active = dict.fromkeys(self.model_limits, 0)
        self.condition = asyncio.Condition()
        self.waiters = []
        self.last_lane = -1

    def next_waiter(self):
        if self.active >= self.limit:
            return None
        for offset in range(1, len(self.lanes) + 1):
            lane = self.lanes[(self.last_lane + offset) % len(self.lanes)]
            for waiter in self.waiters:
                _, model, queued_lane, _ = waiter
                if queued_lane == lane and (model is None or
                        self.model_active[model] < self.model_limits[model]):
                    return waiter
        return None

    def queue_status(self):
        now = time.monotonic()
        return {model: {
            "queued": sum(row[1] == model for row in self.waiters),
            "oldest_queue_s": max((now - row[3] for row in self.waiters
                                    if row[1] == model), default=0),
        } for model in self.model_limits}

    async def acquire(self, model=None, channel="roles"):
        lane = "actor" if channel.startswith(("actor-", "wb-")) else (
            "grading" if channel.startswith("grade") else "roles")
        waiter = (object(), model, lane, time.monotonic())
        async with self.condition:
            self.waiters.append(waiter)
            try:
                await self.condition.wait_for(lambda: self.next_waiter() is waiter)
            except BaseException:
                self.waiters.remove(waiter)
                self.condition.notify_all()
                raise
            else:
                self.waiters.remove(waiter)
                self.active += 1
                if model is not None:
                    self.model_active[model] += 1
                self.last_lane = self.lanes.index(lane)
                self.condition.notify_all()

    async def release(self, model=None):
        async with self.condition:
            self.active -= 1
            if model is not None:
                self.model_active[model] -= 1
            self.condition.notify_all()

    async def resize(self, limit, model_limits=None):
        async with self.condition:
            self.limit = limit
            self.model_limits.update(model_limits or {})
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
    persisted_limits = read(limits_path) if limits_path.exists() else {}
    limit = persisted_limits.get("global_limit", config["model_concurrency"])
    ceiling = config.get("max_model_concurrency", 8)
    if type(ceiling) is not int or not 1 <= ceiling <= 128:
        raise ValueError("Invalid configured transport ceiling")
    if type(limit) is not int or not 1 <= limit <= ceiling:
        raise ValueError("Transport concurrency exceeds configured ceiling")
    secret = (
        backend_key
        or dotenv_values(config["credentials_file"])[
            config.get("backend_key_env", "OPENAI_API_KEY")
        ]
    )
    allowed = set(config["roles"].values())
    native_grader = (
        validate_native_grader(config["native_grader"])
        if "native_grader" in config
        else None
    )
    models = allowed | ({native_grader["model"]} if native_grader else set())
    if config.get("native_grader_model"):
        models.add(config["native_grader_model"])
    per_model_limit = config.get("per_model_concurrency", 2)
    if type(per_model_limit) is not int or not 1 <= per_model_limit <= ceiling:
        raise ValueError("Per-model concurrency exceeds configured ceiling")
    def model_limits(value):
        if not isinstance(value, dict) or any(
            model not in models or type(size) is not int
            or not 1 <= size <= per_model_limit
            for model, size in value.items()
        ):
            raise ValueError("model_limits exceeds configured per-model ceiling")
        return value

    initial_model_limits = {
        **dict.fromkeys(models, per_model_limit),
        **model_limits(persisted_limits.get("model_limits", {})),
    }
    gate = Capacity(limit, initial_model_limits)
    control_lock = asyncio.Lock()
    model_inflight = {model: 0 for model in models}
    counters = {"inflight": 0, "peak": 0, "requests": 0}
    role_stream_timeout_s = role_timeout_s(config, stream_timeout_s)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "limit": gate.limit,
            "per_model_limit": per_model_limit,
            "model_limits": dict(gate.model_limits),
            "model_inflight": model_inflight,
            "model_queues": gate.queue_status(),
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
        try:
            updates = model_limits(body.get("model_limits", {}))
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        async with control_lock:
            previous_models = dict(gate.model_limits)
            desired_models = {**previous_models, **updates}
            event = {
                "global_limit": limit,
                "previous_limit": gate.limit,
                "per_model_limit": per_model_limit,
                "model_limits": desired_models,
                "previous_model_limits": previous_models,
                "updated_at": time.time(),
            }
            write(
                Path(config["output"]) / "transport_limit_events" / (uuid4().hex + ".json"),
                event,
            )
            write(limits_path, event)
            await gate.resize(limit, desired_models)
        return await health()

    @app.post("/{channel}/v1/chat/completions")
    async def completion(channel: str, request: Request):
        actor_capability = None
        if not channel.replace("-", "").replace("_", "").isalnum():
            raise HTTPException(400, "Invalid channel")
        if channel.startswith("actor-"):
            from reward_engine.adapters.jiuwenswarm.gateway import authorize
            try:
                actor_capability = authorize(config, channel, request.headers.get("authorization", "").removeprefix("Bearer "))
            except (OSError, ValueError, KeyError):
                raise HTTPException(401, "Invalid actor capability") from None
        elif request.headers.get("authorization") != "Bearer " + auth(config):
            raise HTTPException(401, "Invalid campaign credential")
        body = await request.json()
        tool_audit = None
        if actor_capability is not None:
            from reward_engine.adapters.jiuwenswarm.policy import filter_request
            if body.get("model") != actor_capability["model"]:
                raise HTTPException(400, "Actor model differs from registered capability")
            try:
                body, tool_audit = filter_request(body, actor_capability["policy"])
            except (ValueError, TypeError, AttributeError, KeyError) as error:
                raise HTTPException(400, str(error)) from error
        if config.get("native_grader_model") and channel.startswith("grade"):
            if channel != "grade" or body.get("model") != config["native_grader_model"]:
                raise HTTPException(400, "Native grader channel or model differs")
        received_body = None
        if native_grader and channel.startswith("grade"):
            if (
                not channel.startswith("grade-wb-")
                or not channel.removeprefix("grade-wb-")
                or body.get("model") != native_grader["model"]
            ):
                raise HTTPException(400, "Native grader channel or model differs")
            parameters = {
                "max_tokens": native_grader["max_tokens"],
                **native_grader["request_options"],
            }
            if set(body) - {"model", "messages", *parameters} or any(
                key in body
                and (type(body[key]) is not type(value) or body[key] != value)
                for key, value in parameters.items()
            ):
                raise HTTPException(
                    400, "Native grader request differs from frozen parameters"
                )
            received_body = body
            body = {**body, **parameters}
        elif body.get("model") not in allowed and not (channel == "grade" and body.get("model") == config.get("native_grader_model")):
            raise HTTPException(400, "Unconfigured model")
        if channel.startswith("wb-"):
            body.setdefault("temperature", 0.95)
        root = Path(config["output"]) / (
            "official.private" if channel.startswith("grade") else "transport"
        )
        root = root / channel / uuid4().hex
        if received_body is not None:
            write(root / "received_request.json", received_body)
        if tool_audit is not None:
            write(root / "tool_policy.json", tool_audit)
        write(root / "request.json", body)
        single_attempt = channel == "roles" and config.get("role_input_projection") == "canonical_sources_v1"
        record = {
            "channel": channel,
            "model": body["model"],
            "queued_at": time.time(),
            "status": "queued",
        }
        if single_attempt:
            from reward_engine.adapters.workbuddy.role_requests import reserve

            record.update(gateway_calls=1, attempts=0, upstream_attempts=0,
                          logical_operation_id=request.headers.get("x-role-operation-id"),
                          physical_request_id=request.headers.get("x-role-request-id"))
            if config.get("role_contract_policy"):
                record.update(contract_policy=config["role_contract_policy"], config_identity=config["identity"])
            try:
                record.update(reserve(config, request.headers, root / "receipt.json"))
            except ValueError as error:
                record.update(status="rejected_before_execution", error_type=type(error).__name__, reason=str(error))
                write(root / "receipt.json", record)
                raise HTTPException(409, str(error), headers={"X-Role-Gateway-Receipt": str(root / "receipt.json")}) from error
        write(root / "receipt.json", record)
        global_acquired = False

        async def acquire_capacity():
            nonlocal global_acquired
            await gate.acquire(body["model"], channel)
            global_acquired = True

        async def wait_for_capacity():
            acquisition = asyncio.create_task(acquire_capacity())
            try:
                while True:
                    done, _ = await asyncio.wait({acquisition}, timeout=1)
                    if done:
                        await acquisition
                    # Uvicorn does not cancel a handler merely because its peer
                    # disconnected while awaiting response headers. Do not run
                    # an abandoned Actor request after its local timeout.
                    if await request.is_disconnected():
                        raise asyncio.CancelledError("Client disconnected while waiting for capacity")
                    if done:
                        return
            finally:
                if not acquisition.done():
                    acquisition.cancel()
                await asyncio.gather(acquisition, return_exceptions=True)

        try:
            if single_attempt:
                async with asyncio.timeout(max(0, record["deadline"] - time.time())):
                    await wait_for_capacity()
            else:
                await wait_for_capacity()
        except BaseException as error:
            if global_acquired:
                await gate.release(body["model"])
            record.update(status="rejected_before_execution", error_type=type(error).__name__,
                          reason="Operation ended while waiting for gateway capacity", finished_at=time.time())
            write(root / "receipt.json", record)
            if single_attempt:
                if not isinstance(error, asyncio.CancelledError):
                    raise HTTPException(504, record["reason"], headers={"X-Role-Gateway-Receipt": str(root / "receipt.json")}) from error
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
        client = client_factory(
            timeout=httpx.Timeout(840, read=120 if body.get("stream") else request_timeout_s),
            trust_env=False,
        )
        response = None

        async def open_upstream():
            """Open a response without exposing partial content to the caller."""
            nonlocal response
            waits = (5, 15)
            for retry in range(1 if single_attempt else 3):
                if single_attempt and time.time() >= record["deadline"]:
                    record["status"] = "rejected_before_execution"
                    raise TimeoutError("Operation deadline exhausted in gateway queue")
                record["attempts"] = record.get("attempts", 0) + 1
                if single_attempt:
                    record["upstream_attempts"] = record["attempts"]
                    record["upstream_started_at"] = time.time()
                    write(root / "receipt.json", record)
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
                    if single_attempt or retry == 2:
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
                if not single_attempt and response.status_code in {429, 500, 502, 503, 504} and retry < 2:
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
            await gate.release(body["model"])
            record["elapsed_s"] = time.monotonic() - started
            record["finished_at"] = time.time()
            write(root / "receipt.json", record)

        reply_headers = {"X-Role-Gateway-Receipt": str(root / "receipt.json")} if single_attempt else {}
        if single_attempt:
            request_timeout_s = min(request_timeout_s, max(0, record["deadline"] - time.time()))
        try:
            # Retry only before any response body is exposed to the caller.
            async with asyncio.timeout(request_timeout_s):
                await open_upstream()
                if not body.get("stream"):
                    raw = await response.aread()
                    (root / "response.json").write_bytes(raw)
                    data = json.loads(raw)
                    if data.get("model") and not model_matches(
                        config, body["model"], data["model"]
                    ):
                        raise ValueError("Provider model identity differs")
                    for choice in data.get("choices", []):
                        failure = completion_timeout(choice.get("message", {}).get("content"))
                        if failure:
                            record.setdefault("provider_errors", []).append(failure)
                            raise TimeoutError(failure["message"])
                    record.update(
                        status="completed",
                        usage=data.get("usage"),
                        actual_model=data.get("model"),
                    )
            if not body.get("stream"):
                await close()
                return Response(raw, media_type="application/json", headers=reply_headers)
        except BaseException as error:
            record.update(status="failed" if record.get("attempts", 0) else "rejected_before_execution", error_type=type(error).__name__)
            if isinstance(error, httpx.HTTPStatusError):
                record["upstream_http_status"] = error.response.status_code
                if single_attempt and "Retry-After" in error.response.headers:
                    reply_headers["Retry-After"] = error.response.headers["Retry-After"]
            await close()
            if isinstance(error, asyncio.CancelledError):
                raise
            raise HTTPException(
                (error.response.status_code if single_attempt and isinstance(error, httpx.HTTPStatusError)
                 else 502),
                "Upstream request failed; see campaign transport receipt", headers=reply_headers
            ) from None

        async def stream():
            nonlocal response
            try:
                async with asyncio.timeout(request_timeout_s):
                    for stream_attempt in range(1 if single_attempt else 6):
                        buffered, completed, finish, model, usage = (
                            [],
                            False,
                            None,
                            None,
                            None,
                        )
                        retry_reason = None
                        provider_error = None
                        content_prefix = ""
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
                                            provider_error = {
                                                "attempt": record["attempts"],
                                                "stream_attempt": stream_attempt + 1,
                                                "code": error.get("code"),
                                                "type": error.get("type"),
                                                "message": error.get("message"),
                                                "reason": "provider_stream_error",
                                            }
                                            record.setdefault(
                                                "provider_errors", []
                                            ).append(provider_error)
                                            if (
                                                error.get("code")
                                                == "LITELLM_UNAVAILABLE"
                                            ):
                                                retry_reason = (
                                                    "provider_litellm_unavailable"
                                                )
                                                break
                                            observed_error = (
                                                error.get("code"),
                                                error.get("type"),
                                                error.get("message"),
                                            )
                                            if observed_error == (
                                                None,
                                                "api_error",
                                                "Service temporarily unavailable",
                                            ):
                                                retry_reason = (
                                                    "provider_api_service_unavailable"
                                                )
                                                break
                                            if observed_error == (
                                                "error",
                                                "upstream_error",
                                                "Stream error occurred",
                                            ):
                                                retry_reason = (
                                                    "provider_upstream_stream_error"
                                                )
                                                break
                                            # Only the explicit code/observed tuples above
                                            # override rejection of coded provider errors.
                                            if error.get("code") is not None:
                                                raise RuntimeError(
                                                    "Provider stream error"
                                                )
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
                                        content = choice.get("delta", {}).get("content")
                                        if isinstance(content, str) and len(content_prefix) < 1024:
                                            content_prefix += content[:1024 - len(content_prefix)]
                        except (httpx.ReadTimeout, httpx.RemoteProtocolError):
                            retry_reason = "interrupted_stream"
                        if not retry_reason:
                            provider_error = completion_timeout(content_prefix)
                            if provider_error:
                                retry_reason = "provider_response_timeout"
                                record.setdefault("provider_errors", []).append(provider_error)
                        attempt_record = {
                            "attempt": record["attempts"],
                            "actual_model": model,
                            "usage": usage,
                            "finish_reason": finish,
                            "done": completed,
                        }
                        if provider_error is not None:
                            provider_error["reason"] = retry_reason
                            attempt_record["error_code"] = provider_error["code"]
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
                        if single_attempt or stream_attempt == 5:
                            raise RuntimeError("Provider transport retries exhausted")
                        record.setdefault("retries", []).append(
                            {
                                "reason": retry_reason,
                                "attempt": record["attempts"],
                                "stream_attempt": stream_attempt + 1,
                                **(
                                    {"error_code": provider_error["code"]}
                                    if provider_error is not None
                                    else {}
                                ),
                                "at": time.time(),
                            }
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

        return StreamingResponse(stream(), media_type="text/event-stream", headers=reply_headers)

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
