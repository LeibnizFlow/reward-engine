"""Synthetic protocol error frames, with no captured run identifiers."""

import json

SERVICE_ERRORS = []
for error_type, code, message, reason in [
    (
        "upstream_error",
        "error",
        "Stream error occurred",
        "provider_upstream_stream_error",
    ),
    (
        "api_error",
        None,
        "Service temporarily unavailable",
        "provider_api_service_unavailable",
    ),
]:
    event = {
        "error": {"type": error_type, "code": code, "message": message, "param": ""}
    }
    SERVICE_ERRORS.append(
        {
            "name": error_type,
            "event": event,
            "sse_line": "data: " + json.dumps(event),
            "reason": reason,
        }
    )
