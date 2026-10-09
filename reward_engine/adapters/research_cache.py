"""Optional successful-public-IO cache. No Task cache or cached failure verdicts."""

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path


class ResearchCache:
    VERSION = "research-reader-v2"

    def __init__(self, directory=None, *, clock=time.time):
        self.directory = Path(directory).expanduser() if directory else None
        self.clock = clock

    async def call(self, kind, parameters, execute, *, ttl_s):
        key = hashlib.sha256(
            json.dumps([self.VERSION, kind, parameters], sort_keys=True).encode()
        ).hexdigest()
        path = self.directory / (key + ".json") if self.directory else None
        if path and path.exists():
            try:
                row = json.loads(path.read_text())
                age = self.clock() - row["timestamp"]
                if 0 <= age < ttl_s and self.cacheable(row["value"]):
                    value = row["value"]
                    return {
                        **value,
                        "origin_reader_receipts": value.get("reader_receipts", []),
                        "reader_receipts": [],
                        "physical_requests": 0,
                        "io_receipt": {
                            "kind": kind,
                            "cache_hit": True,
                            "timestamp": row["timestamp"],
                            "key": key,
                        },
                    }
            except (OSError, ValueError, KeyError, TypeError):
                pass  # Corrupt/old cache entries are misses, never successful evidence.
        value = await execute()
        timestamp = self.clock()
        cache_error = None
        if path and self.cacheable(value):
            temp = None
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=path.parent, delete=False
                ) as handle:
                    temp = handle.name
                    json.dump({"timestamp": timestamp, "value": value}, handle)
                os.replace(temp, path)
            except OSError as exc:
                cache_error = type(
                    exc
                ).__name__  # Optional cache cannot erase successful IO.
            finally:
                if temp:
                    try:
                        Path(temp).unlink(missing_ok=True)
                    except OSError:
                        pass  # Cleanup of an optional cache must not erase successful IO.
        return {
            **value,
            "io_receipt": {
                "kind": kind,
                "cache_hit": False,
                "timestamp": timestamp,
                "key": key,
                "cache_write_error": cache_error,
            },
        }

    @staticmethod
    def cacheable(value):
        origin = value.get("skill_origin") if isinstance(value, dict) else None
        return (
            isinstance(value, dict)
            and isinstance(value.get("results"), list)
            and bool(value["results"])
            and not value.get("failed_results")
            and value.get("cacheable", True)
            and value.get("read_kind") != "unavailable"
            and value.get("resolution_status") not in {"unresolved", "unavailable"}
            and (
                not isinstance(origin, dict)
                or origin.get("status") not in {"unresolved", "unavailable"}
            )
        )
