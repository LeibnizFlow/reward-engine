"""Run the real CLI and reap its process group before returning its exit status."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    os.environ.pop("LD_LIBRARY_PATH", None)
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("entry.py")), *sys.argv[1:]],
        start_new_session=True,
    )

    def forward(signum, _frame):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    try:
        code = process.wait()
    finally:
        forward(signal.SIGTERM, None)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        forward(signal.SIGKILL, None)
        process.wait()
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main())
