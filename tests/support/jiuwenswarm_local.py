"""Local deterministic Chat Completions server for real Process CLI witnesses."""

import asyncio
import json
import socket
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


@asynccontextmanager
async def local_model(respond):
    app = FastAPI()
    calls = []

    @app.post("/v1/chat/completions")
    async def completion(request: Request):
        body = await request.json()
        calls.append(body)
        message = await respond(body, len(calls))
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        if body.get("stream"):
            delta = dict(message)
            if delta.get("tool_calls"):
                delta["tool_calls"] = [
                    {"index": i, **c} for i, c in enumerate(delta["tool_calls"])
                ]
            chunk = {
                "id": "local-fixture",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
            return StreamingResponse(
                iter(["data: " + json.dumps(chunk) + "\n\n", "data: [DONE]\n\n"]),
                media_type="text/event-stream",
            )
        return {
            "id": "local-fixture",
            "model": body["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    async with serving(app) as port:
        yield port, calls


@asynccontextmanager
async def serving(app):
    sock = socket.socket()
    sock.bind(("0.0.0.0", 0))
    sock.listen()
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    job = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        for _ in range(100):
            if server.started:
                break
            if job.done():
                await job
            await asyncio.sleep(0.02)
        else:
            raise TimeoutError("Local fixture server failed to start")
        yield port
    finally:
        server.should_exit = True
        await asyncio.wait_for(job, 10)
        sock.close()


def tool(name, arguments, identifier="fixture-call"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": identifier,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }
