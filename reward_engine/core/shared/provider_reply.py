"""Unwrap explicit provider presentation markers; never repair or invent JSON."""


def json_answer_text(text: str) -> str:
    raw = text.strip()
    # Some gateways split <think> into reasoning_content but leave its closing
    # marker plus the last reasoning fragment in content. Only an explicit
    # leading reasoning envelope may be removed; markers inside JSON are data.
    if raw.startswith("<think>") or (not raw.startswith(("{", "[", '"', "```")) and "</think>" in raw):
        _, marker, raw = raw.partition("</think>")
        if not marker:
            raise ValueError("Provider reasoning envelope is incomplete")
        raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) < 3 or lines[0] not in {"```", "```json"} or lines[-1] != "```":
            raise ValueError("Provider JSON fence is incomplete or ambiguous")
        raw = "\n".join(lines[1:-1]).strip()
    return raw
