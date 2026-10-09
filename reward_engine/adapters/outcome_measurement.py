"""Wire real measurement tools into Builder and delivery review from trusted config."""

import importlib
import ipaddress
import os
from pathlib import Path
from urllib.parse import urlparse

from reward_engine.adapters.support import load_named_key, write_json
from reward_engine.adapters.tavily_review import TavilyReview
from reward_engine.core.outcome.harness import defaults
from reward_engine.core.outcome.measurement_tools import MeasurementTools


def measurement_registry(call, config, output):
    settings = config.get("measurement", {})
    for flag in ("external_documents", "cuda_isolated"):
        if flag in settings and type(settings[flag]) is not bool:
            raise ValueError(f"measurement.{flag} must be boolean")

    async def read_document(url):
        # Only the observed URL is sent to the reader, never the whole Task/artifact.
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if (
            parsed.scheme != "https"
            or parsed.username
            or parsed.password
            or not host
            or host == "localhost"
            or host.endswith((".local", ".internal"))
        ):
            raise ValueError("Source reader accepts public HTTPS documents only")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise ValueError("Private source addresses are not allowed")
        key = os.environ.get("TAVILY_API_KEY") or load_named_key(
            "TAVILY_API_KEY", config.get("credentials_file", "~/.pinchbench_env")
        )
        response = await TavilyReview(key).extract(url)
        rows = [
            r
            for r in response.get("results", [])
            if r.get("url") == url and r.get("raw_content")
        ]
        if len(rows) != 1:
            raise ValueError("Reader did not return the requested document body")
        return rows[0]

    cuda = None
    if settings.get("cuda_runner"):
        # Operator-owned adapter, never a module name proposed by the model.
        # Adapter owns isolation and real GPU execution; this is not a host-code sandbox.
        if settings.get("cuda_isolated") is not True:
            raise ValueError(
                "CUDA adapter requires an explicitly configured isolated backend"
            )
        module, name = settings["cuda_runner"].split(":", 1)
        cuda = getattr(importlib.import_module(module), name)
        if not callable(cuda):
            raise ValueError("CUDA runner must be a real callable")

    def record(name, value):
        directory = Path(output) / "measurements"
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / (name + ".json"), value)

    return defaults(
        call,
        tools=MeasurementTools(
            document_reader=read_document
            if settings.get("external_documents", False)
            else None,
            cuda_runner=cuda,
            record=record,
        ),
    )
