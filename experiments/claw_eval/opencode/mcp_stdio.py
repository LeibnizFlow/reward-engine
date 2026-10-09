"""Small JSON-RPC stdio transport; execution and audit live in the trusted host."""

import json
import sys
import urllib.request

url, token = sys.argv[1:]
for line in sys.stdin:
    try:
        request = json.loads(line)
        if "id" not in request:
            continue
        method = request["method"]
        if method == "initialize":
            result = {
                "protocolVersion": request.get("params", {}).get(
                    "protocolVersion", "2024-11-05"
                ),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "claw", "version": "1.0"},
            }
        elif method == "ping":
            result = {}
        elif method in ("tools/list", "tools/call"):
            body = json.dumps(
                {"method": method, "params": request.get("params", {})}
            ).encode()
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + token,
                },
            )
            with urllib.request.urlopen(req, timeout=180) as response:
                result = json.load(response)
        else:
            raise ValueError("Unsupported method " + method)
        reply = {"jsonrpc": "2.0", "id": request["id"], "result": result}
    except Exception as error:
        reply = {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {
                "code": -32603,
                "message": type(error).__name__ + ": " + str(error),
            },
        }
    print(json.dumps(reply), flush=True)
