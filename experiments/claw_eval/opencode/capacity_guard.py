"""Keep a running legacy campaign at its measured stable gateway limit."""

import argparse
import json
import time
import urllib.request
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def request_json(url, *, token=None, payload=None):
    headers = {}
    data = None
    method = "GET"
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
        method = "POST"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read())


def process_alive(root):
    process = root / "process.json"
    if not process.exists():
        return False
    pid = read(process).get("pid")
    cmdline = Path("/proc") / str(pid) / "cmdline"
    try:
        command = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return "experiments.claw_eval.opencode.campaign" in command and str(root) in command


def recent_real_failures(root, since):
    failures = []
    for section in ("transport", "official.private"):
        for path in (root / section).glob("*/*/receipt.json"):
            row = read(path)
            if (
                path.stat().st_mtime >= since
                and row.get("status") == "failed"
                and row.get("error_type") != "CancelledError"
            ):
                failures.append(
                    {"path": str(path), "error_type": row.get("error_type")}
                )
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stable-limit", type=int, default=3)
    parser.add_argument("--fallback-limit", type=int, default=2)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    root = args.output.resolve()
    config = read(root / "config.json")
    token = (root / "campaign.auth").read_text().strip()
    log = root / "capacity_guard.jsonl"

    while process_alive(root) and not (root / "acceptance.json").exists():
        health = request_json(config["gateway_local"] + "/health")
        if health["limit"] < args.stable_limit:
            events = sorted((root / "capacity_events").glob("*.json"))
            latest = read(events[-1])["at"] if events else time.time()
            failures = recent_real_failures(root, latest - 90)
            target = args.fallback_limit if failures else args.stable_limit
            if health["limit"] != target:
                result = request_json(
                    config["gateway_local"] + "/control/concurrency",
                    token=token,
                    payload={"global_limit": target},
                )
                with log.open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "at": time.time(),
                                "from": health["limit"],
                                "to": target,
                                "recent_real_failures": failures,
                                "result": result,
                            }
                        )
                        + "\n"
                    )
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
