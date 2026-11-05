"""Streaming byte verification with bounded I/O and observable progress.

An explicit local cache can reuse hashes of unchanged file identities. Every
verification still enumerates the full inventory and compares expected hashes.
"""

import json
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_progress = ContextVar("integrity_progress", default=None)
_cache = ContextVar("integrity_cache", default=None)


@contextmanager
def verification_cache(destination):
    """Share a metadata-validated digest cache across a recursive verification."""
    from reward_engine.adapters.integrity_cache import DigestCache

    if _cache.get() is not None:
        yield _cache.get()
        return
    cache = DigestCache(destination, reuse=os.environ.get('REWARD_ENGINE_VERIFY_FULL') != '1')
    token = _cache.set(cache)
    try:
        yield cache
    finally:
        verification_detail(cache_hits=cache.hits, files_read=cache.reads)
        _cache.reset(token)
        cache.close()


class Progress:
    def __init__(self, destination, phase):
        self.destination = Path(destination)
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.state = {"phase": phase, "status": "running", "pid": os.getpid(),
                      "started_at": time.time(), "steps": []}
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    def update(self, **values):
        with self.lock:
            self.state.update(values, last_progress_at=time.time())

    def step(self, root, total):
        with self.lock:
            self.state["steps"].append({"root": str(root), "status": "running",
                                        "files_hashed": 0, "expected_files": total})
        self.update(current_root=str(root), current_path=None)

    def tick(self, *, path=None, hashed=False, completed=False):
        with self.lock:
            if path is not None:
                self.state["current_path"] = str(path)
            if hashed:
                self.state["steps"][-1]["files_hashed"] += 1
            if completed:
                self.state["steps"][-1]["status"] = "completed"
            self.state["last_progress_at"] = time.time()

    def flush(self):
        with self.lock:
            value = {**self.state, "updated_at": time.time(),
                     "elapsed_s": time.monotonic() - self.started}
            text = json.dumps(value, indent=2) + "\n"
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.destination.with_name(self.destination.name + f".{os.getpid()}.tmp")
        temporary.write_text(text)
        temporary.replace(self.destination)

    def _heartbeat(self):
        while not self.stop.wait(2):
            self.flush()


@contextmanager
def verification_progress(destination, phase):
    """A heartbeat distinguishes a slow scan from task/model execution."""
    if _progress.get() is not None:
        yield
        return
    progress = Progress(destination, phase)
    token = _progress.set(progress)
    progress.flush()
    progress.thread.start()
    try:
        yield
    except BaseException as error:
        progress.update(status="failed", error_type=type(error).__name__)
        raise
    else:
        progress.update(status="completed")
    finally:
        progress.stop.set()
        progress.thread.join()
        try:
            progress.flush()
        finally:
            _progress.reset(token)


def verification_detail(**values):
    progress = _progress.get()
    if progress:
        progress.update(**values)


def files(root, *, ignore_bytecode=False):
    """One scandir per directory, no symlink traversal and no global glob."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Frozen tree cannot contain symlinks")
    pending = [root]
    progress = _progress.get()
    while pending:
        directory = pending.pop()
        if progress:
            progress.tick(path=directory)
        with os.scandir(directory) as entries:
            children = []
            for entry in entries:
                if ignore_bytecode and entry.name == "__pycache__":
                    continue
                if entry.is_symlink():
                    raise ValueError("Frozen tree cannot contain symlinks")
                if entry.is_dir(follow_symlinks=False):
                    children.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    yield Path(entry.path)
                else:
                    raise ValueError("Frozen tree contains a non-regular file")
        pending.extend(reversed(sorted(children)))


def tree_hashes(root, hasher, *, ignore_bytecode=False, exclude=(), expected_files=None,
                inspect_ignored=False):
    """Read every file once using at most two readers and eight queued files.

    The bounded queue avoids an unbounded prefetch load on shared storage. The
    resulting inventory includes additions, so matching old hashes cannot hide
    new files. The caller compares the complete result to its frozen manifest.
    """
    root = Path(root)
    from reward_engine.adapters.workbuddy.io import file_hash

    cache = _cache.get()
    # Callers may provide another hashing algorithm or an instrumented hasher.
    # Such calls never reuse SHA-256 results implicitly.
    if cache is not None and hasher is file_hash:
        hasher = cache.digest
    progress = _progress.get()
    if progress:
        progress.step(root, expected_files)
    result = {}
    pending = deque()

    def collect():
        name, future = pending.popleft()
        result[name] = future.result()
        if progress:
            progress.tick(path=root / name, hashed=True)
            if cache is not None:
                progress.update(cache_hits=cache.hits, files_read=cache.reads)

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="integrity") as pool:
        for path in files(root, ignore_bytecode=ignore_bytecode and not inspect_ignored):
            name = str(path.relative_to(root))
            if ignore_bytecode and "__pycache__" in path.relative_to(root).parts:
                continue
            if name in exclude:
                continue
            pending.append((name, pool.submit(hasher, path)))
            if len(pending) >= 8:
                collect()
        while pending:
            collect()
    if progress:
        progress.tick(completed=True)
    return dict(sorted(result.items()))
