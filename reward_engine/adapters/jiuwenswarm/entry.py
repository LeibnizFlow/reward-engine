"""Configure agent-core diagnostics outside task files, then enter the real CLI."""

import contextlib
import os
import sys
from pathlib import Path


def main():
    # Imports may initialize third-party logging; keep all startup diagnostics
    # off machine stdout and outside the public task workspace.
    os.chdir(os.environ["HOME"])
    with contextlib.redirect_stdout(sys.stderr):
        from openjiuwen.core.common.logging.log_config import log_config

        config = log_config.get_snapshot()
        config["log_path"] = str(Path(os.environ["JIUWENSWARM_DATA_DIR"]) / "core-logs")
        log_config.load_from_dict(config)
        from jiuwenswarm.channels.process_cli.main import main as process_main
    return process_main()


if __name__ == "__main__":
    sys.exit(main())
