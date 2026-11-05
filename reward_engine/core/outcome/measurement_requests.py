"""Versioned pre-execution contracts and exact numerical input provenance."""

import ast
import fnmatch
import json
from functools import lru_cache

from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.numeric_program import (
    MAX_INPUT_BYTES,
    preflight_program,
)

POLICY = "canonical_recovery_v1"


def enabled(owner):
    return getattr(owner, "role_contract_policy", None) == POLICY


class MeasurementRequestRejected(ValueError):
    """Invalid request rejected before the measurement executes."""

    def __init__(self, reason, *, field="arguments", actual=None, code="INVALID_TOOL_ARGUMENT"):
        super().__init__(reason)
        self.diagnostics = [{"code": code, "field": field, "actual": actual,
                             "rule": reason, "reason": reason}]


def _required_calculation_names(node):
    """Names necessarily read by an expression, without executing arithmetic.

    The calculator has no user variable environment. Conditional branches and
    empty comprehensions can legitimately leave names unread, so only reject
    names proven necessary. Function names are resolved by the numeric runner.
    """
    if isinstance(node, ast.Name):
        return {node.id} - {"inputs"}
    if isinstance(node, ast.Call):
        children = [*node.args, *(item.value for item in node.keywords)]
        if not isinstance(node.func, ast.Name):
            children.append(node.func)
    elif isinstance(node, ast.IfExp):
        return _required_calculation_names(node.test) | (
            _required_calculation_names(node.body) & _required_calculation_names(node.orelse)
        )
    elif isinstance(node, ast.BoolOp):
        children = node.values[:1]
    elif isinstance(node, ast.Compare):
        children = [node.left, *node.comparators[:1]]
    elif isinstance(node, ast.ListComp):
        # Its body/filters may never execute. The first iterable is evaluated
        # before binding the comprehension variable.
        children = [node.generators[0].iter] if node.generators else []
    else:
        children = ast.iter_child_nodes(node)
    return set().union(*(_required_calculation_names(child) for child in children))


def _literal_numeric_type(node):
    """Infer only unconditional arithmetic types; never compute a value."""
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        return type(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _literal_numeric_type(node.operand)
    if isinstance(node, ast.BinOp):
        left, right = _literal_numeric_type(node.left), _literal_numeric_type(node.right)
        if left not in {int, float} or right not in {int, float}:
            return None
        if isinstance(node.op, ast.Div):
            return float
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
            return float if float in {left, right} else int
        if isinstance(node.op, (ast.BitXor, ast.BitAnd, ast.BitOr, ast.LShift, ast.RShift)):
            return int if left is right is int else None
    return None


def _invalid_float_bitwise(node):
    """Reject proven operand errors without inspecting possibly skipped branches."""
    invalid = []
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.BitXor, ast.BitAnd, ast.BitOr, ast.LShift, ast.RShift)
    ) and float in {_literal_numeric_type(node.left), _literal_numeric_type(node.right)}:
        invalid.append(ast.unparse(node))
    if isinstance(node, ast.IfExp):
        children = [node.test]
    elif isinstance(node, ast.BoolOp):
        children = node.values[:1]
    elif isinstance(node, ast.Compare):
        children = [node.left, *node.comparators[:1]]
    elif isinstance(node, ast.ListComp):
        children = [node.generators[0].iter] if node.generators else []
    else:
        children = ast.iter_child_nodes(node)
    for child in children:
        invalid.extend(_invalid_float_bitwise(child))
    return invalid


def preflight(request, evidence=None, *, check_program=True):
    """Shared static recipe/runtime checks; never evaluate the program here."""
    def reject(reason, field="arguments", actual=None):
        raise MeasurementRequestRejected(reason, field=field, actual=actual)

    if not isinstance(request, dict):
        reject("Tool request must be an object", "request", request)
    tool, args = request.get("tool"), request.get("arguments", {})
    if not isinstance(args, dict):
        reject("Tool arguments must be an object", actual=args)
    if tool not in {"calculate", "code.run", "cuda.run", "document.read", "workspace.read", "workspace.inspect"}:
        reject("Unknown measurement tool", "tool", tool)
    for key in (["expression"] if tool == "calculate" else ["program"] if tool == "code.run"
                else ["source"] if tool == "cuda.run" else []):
        value = args.get(key)
        if not isinstance(value, str) or not value.strip():
            reject("Supply a nonempty string", "arguments." + key, value)
        if tool in {"calculate", "code.run"}:
            program = "result = (" + value + ")" if tool == "calculate" else value
            if len(program) > 16000:
                reject("Numerical program exceeds 16000 characters", "arguments." + key)
            try:
                ast.parse(value, mode="eval") if tool == "calculate" else ast.parse(value)
                ast.parse(program)
            except SyntaxError as error:
                reject("Supply one expression without assignment/semicolons" if tool == "calculate"
                       else "Supply syntactically valid numerical Python: " + error.msg,
                       "arguments." + key, value)
            if tool == "calculate" and evidence is not None and check_program:
                # Keep published symbolic recipes intact; reject their invalid
                # concrete invocation before execution and let the model author
                # a corrected request against observed evidence.
                expression = ast.parse(value, mode="eval")
                missing = sorted(_required_calculation_names(expression))
                if missing:
                    reject("calculate has no user-defined variables. Supply concrete observed numbers, "
                           "or use code.run with explicitly cited inputs; do not invent values.",
                           "arguments.expression", missing)
                invalid = _invalid_float_bitwise(expression)
                if invalid:
                    reject("Bitwise operators (^, &, |, <<, >>) require integer operands; "
                           "use ** for exponentiation.", "arguments.expression", invalid)
            if tool == "code.run" and evidence is not None and check_program and not any(
                isinstance(node, ast.Name) and node.id == "result" and isinstance(node.ctx, ast.Store)
                for node in ast.walk(ast.parse(value))
            ):
                reject("code.run must assign the computed value to result; print() only captures "
                       "stdout and does not supply a result.", "arguments.program")
            if tool == "code.run" and evidence is not None and check_program:
                try:
                    preflight_program(value)
                except ValueError as error:
                    reject(str(error), "arguments.program")
    bindings = []
    if tool in {"code.run", "cuda.run"}:
        refs, paths = args.get("input_refs", []), args.get("input_paths", {})
        if not isinstance(refs, list) or any(not isinstance(r, str) for r in refs) or len(set(refs)) != len(refs):
            reject("Supply distinct exact evidence references", "arguments.input_refs", refs)
        if not isinstance(paths, dict) or set(paths) - set(refs) or (tool == "cuda.run" and paths):
            reject("input_paths selects only cited code.run input_refs", "arguments.input_paths", paths)
        selected = {}
        for ref in refs:
            path = paths.get(ref, [])
            if not isinstance(path, list) or len(path) > 16 or any(type(p) not in {str, int} or type(p) is int and p < 0 for p in path):
                reject("Use at most 16 exact JSON keys or nonnegative array indices", "arguments.input_paths." + ref, path)
            if evidence is None:
                continue
            if ref not in evidence:
                reject("Reference is not supplied evidence", "arguments.input_refs", ref)
            value = evidence[ref]
            for part in path:
                if isinstance(value, dict) and isinstance(part, str) and part in value:
                    value = value[part]
                elif isinstance(value, list) and type(part) is int and 0 <= part < len(value):
                    value = value[part]
                else:
                    reject("Path must select an existing JSON key or nonnegative array index",
                           "arguments.input_paths." + ref, path)
            selected[ref] = value
            bindings.append({"ref": ref, "path": path, "sha256": fingerprint(value)})
        if evidence is not None and tool == "code.run" and len(json.dumps(selected).encode()) > MAX_INPUT_BYTES:
            reject("Selected numerical inputs exceed 256000 bytes; select relevant evidence", "arguments.input_paths")
    if tool in {"workspace.read", "workspace.inspect"}:
        paths = args.get("paths", ["**"])
        if not isinstance(paths, list) or not 1 <= len(paths) <= 32 or any(
            not isinstance(p, str) or not p or p.startswith("/") or ".." in p.split("/") for p in paths
        ):
            reject("Supply one to 32 relative path patterns", "arguments.paths", paths)
    if tool in {"document.read", "workspace.read", "workspace.inspect"}:
        offset = args.get("offset", 0)
        if type(offset) is not int or offset < 0:
            reject("Offset must be a nonnegative integer", "arguments.offset", offset)
    if tool == "document.read":
        key = "ref" if "ref" in args else "url"
        if not isinstance(args.get(key), str) or not args[key]:
            reject("Supply an exact source ref or an observed URL", "arguments." + key, args.get(key))
        if evidence is not None and key == "ref":
            if args[key] not in evidence:
                reject("Document ref is not supplied evidence", "arguments.ref", args[key])
            value = evidence[args[key]]
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            if args.get("offset", 0) > len(text):
                reject("Offset exceeds document length", "arguments.offset", args["offset"])
    return bindings


def comparable(old, replacements):
    """Same selected inputs, never just same source names or model assertions."""
    if old.get("contract_policy") != POLICY or any(r.get("contract_policy") != POLICY for r in replacements):
        return False
    if any(r["arguments"] == old["arguments"] and r.get("input_bindings") == old.get("input_bindings")
           for r in replacements):
        return True
    before = old.get("input_bindings", [])
    after = [b for r in replacements for b in r.get("input_bindings", [])]
    return old["tool"] == "code.run" and bool(before) and all(b in after for b in before)


def match_path(path, pattern):
    """POSIX evidence paths: ** is zero or more directories, * one segment."""
    parts, pattern_parts = path.split("/"), pattern.split("/")

    @lru_cache(None)
    def match(i, j):
        if j == len(pattern_parts):
            return i == len(parts)
        if pattern_parts[j] == "**":
            return match(i, j + 1) or i < len(parts) and match(i + 1, j)
        return i < len(parts) and fnmatch.fnmatchcase(parts[i], pattern_parts[j]) and match(i + 1, j + 1)

    return match(0, 0)
