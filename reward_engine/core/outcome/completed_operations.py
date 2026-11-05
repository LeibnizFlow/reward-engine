"""Completed operation receipts bound to exact public inputs and implementation."""

from copy import deepcopy
from pathlib import Path

from reward_engine.core.outcome.construction import fingerprint


class CompletedOperations:
    def __init__(self, directory, identity, *, ancestors=()):
        self.directory = Path(directory)
        self.identity = deepcopy(identity)
        self.ancestors = list(ancestors)

    def key(self, operation, inputs):
        return fingerprint(
            {"identity": self.identity, "operation": operation, "inputs": inputs}
        )

    def load(self, operation, inputs, *, validate=None):
        import json

        key = self.key(operation, inputs)
        path = self.directory / (key + ".json")
        if not path.exists():
            if validate is None:
                return None
            for ancestor in self.ancestors:
                previous = CompletedOperations(self.directory, {**self.identity, "config_identity": ancestor})
                candidate = previous.load(operation, inputs)
                if candidate is None:
                    continue
                try:
                    result = validate(deepcopy(candidate))
                except (ValueError, TypeError, KeyError, AttributeError):
                    continue
                self.save(operation, inputs, result)
                proof = {"source_binding": previous.key(operation, inputs), "current_binding": key,
                         "source_result_sha256": fingerprint(candidate), "result_sha256": fingerprint(result),
                         "validation": "Full current contract on exact public inputs; zero model calls"}
                path.with_suffix(".reuse.json").write_text(json.dumps(proof, ensure_ascii=False))
                return result
            return None
        saved = json.loads(path.read_text())
        if (
            saved.get("binding") != key
            or saved.get("identity") != self.identity
            or fingerprint(saved["result"]) != saved.get("result_sha256")
        ):
            raise ValueError(
                "Completed operation checkpoint identity or content changed"
            )
        return deepcopy(saved["result"])

    def save(self, operation, inputs, result):
        import json
        from uuid import uuid4

        key = self.key(operation, inputs)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / (key + ".json")
        old = self.load(operation, inputs)
        if old is not None:
            if old != result:
                raise ValueError("Completed operation cannot be replaced")
            return
        data = {
            "binding": key,
            "identity": self.identity,
            "result": result,
            "result_sha256": fingerprint(result),
        }
        temp = self.directory / (key + "." + uuid4().hex + ".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False))
        temp.replace(path)
