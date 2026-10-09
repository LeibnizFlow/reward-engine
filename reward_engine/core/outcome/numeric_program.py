"""Small numerical Python subset, interpreted without eval/exec/import or host IO.

This is not a general Python/CUDA sandbox. Bounded arithmetic, JSON data, loops
and assertions support reproducible numerical checks on supplied evidence.
"""

import ast
import json
import math
import operator

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
       ast.Pow: operator.pow, ast.LShift: operator.lshift, ast.RShift: operator.rshift,
       ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor}
CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
       ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
DEFAULT_STEPS = 1_000_000
MAX_INPUT_BYTES = 256_000


def run_numeric(program, inputs=None, *, steps=DEFAULT_STEPS):
    if not isinstance(program, str) or len(program) > 16000:
        raise ValueError("Program must be bounded text")
    tree = ast.parse(program)
    encoded_inputs = json.dumps(inputs or {})
    if len(encoded_inputs.encode()) > MAX_INPUT_BYTES:
        raise ValueError("Program inputs too large; select relevant evidence")
    env = {"inputs": json.loads(encoded_inputs)}
    remaining = steps

    def tick(cost=1):
        nonlocal remaining
        remaining -= cost
        if remaining < 0:
            raise ValueError("Numerical program step budget exceeded")

    def bound(value):
        if isinstance(value, int) and value.bit_length() > 4096:
            raise ValueError("Integer too large")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite result")
        if isinstance(value, (str, list, tuple, dict)) and len(value) > 10000:
            raise ValueError(f"Value too large: {type(value).__name__} length {len(value)} exceeds 10000; "
                             "select or summarize the value before constructing it")
        if isinstance(value, (list, tuple, dict)):
            pending = [value]
            while pending:
                tick()  # Count expanded structure, including repeated nested references.
                item = pending.pop()
                if isinstance(item, dict):
                    pending.extend(item.values())
                elif isinstance(item, (list, tuple)):
                    pending.extend(item)
        return value

    def expr(node):
        tick()
        if isinstance(node, ast.Constant) and type(node.value) in {str, int, float, bool, type(None)}:
            return bound(node.value)
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return bound([expr(x) for x in node.elts])
        if isinstance(node, ast.Dict):
            return bound({expr(k): expr(v) for k, v in zip(node.keys, node.values, strict=True)})
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Slice):
                value = expr(node.value)
                if not isinstance(value, (str, list, tuple)):
                    raise ValueError("Slice requires a bounded sequence")
                parts = [expr(part) if part is not None else None
                         for part in (node.slice.lower, node.slice.upper, node.slice.step)]
                if any(part is not None and type(part) is not int for part in parts) or parts[2] == 0:
                    raise ValueError("Slice bounds must be integers and step must be nonzero")
                return bound(value[slice(*parts)])
            return expr(node.value)[expr(node.slice)]
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            a, b = expr(node.left), expr(node.right)
            if isinstance(node.op, ast.Add) and type(a) is type(b) and isinstance(a, (str, list, tuple)):
                if len(a) + len(b) > 10000:
                    raise ValueError(f"Value too large at line {node.lineno}: {type(a).__name__} "
                                     f"concatenation length {len(a)} + {len(b)} exceeds 10000; "
                                     "inspect each part directly or aggregate small results instead. "
                                     "Shrinking the final result does not fix this intermediate operation")
                return bound(a + b)
            if type(a) not in {int, float} or type(b) not in {int, float}:
                raise ValueError("Arithmetic operands must be numbers")
            if isinstance(node.op, (ast.Pow, ast.LShift, ast.RShift)) and not 0 <= b <= 128:
                raise ValueError("Exponent/shift out of range")
            return bound(OPS[type(node.op)](a, b))
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            for part in node.values:
                value = expr(part)
                if isinstance(node.op, ast.And) and not value or isinstance(node.op, ast.Or) and value:
                    return value
            return value
        if isinstance(node, ast.UnaryOp):
            value = expr(node.operand)
            if isinstance(node.op, ast.Not):
                return not value
            if type(value) in {int, float} and isinstance(node.op, (ast.USub, ast.UAdd)):
                return bound(-value if isinstance(node.op, ast.USub) else value)
        if isinstance(node, ast.Compare):
            a = expr(node.left)
            for op, right in zip(node.ops, node.comparators, strict=True):
                b = expr(right)
                if isinstance(op, (ast.Is, ast.IsNot)):
                    if a is not None and b is not None:
                        raise ValueError("Identity comparison supports None checks only")
                    if (a is b) != isinstance(op, ast.Is):
                        return False
                    a = b
                    continue
                if isinstance(op, (ast.In, ast.NotIn)):
                    if not isinstance(b, (str, list, tuple, dict)):
                        raise ValueError("Membership requires JSON container or string")
                    tick(len(b))
                    matched = a in b
                    if matched == isinstance(op, ast.NotIn):
                        return False
                    a = b
                    continue
                if type(op) not in CMP:
                    raise ValueError("Unsupported comparison")
                if not CMP[type(op)](a, b):
                    return False
                a = b
            return True
        if isinstance(node, ast.ListComp) and len(node.generators) == 1:
            generator = node.generators[0]
            if (generator.is_async or not isinstance(generator.target, ast.Name)
                    or generator.target.id == "inputs"
                    or (generator.target.id.startswith("_") and generator.target.id != "_")):
                raise ValueError("Use a non-reserved name in a bounded list comprehension")
            values = expr(generator.iter)
            if not isinstance(values, (list, tuple)):
                raise ValueError("Comprehension requires bounded sequence")
            name = generator.target.id
            existed, old = name in env, env.get(name)
            result = []
            try:
                for value in values:
                    tick()
                    env[name] = value
                    if all(expr(condition) for condition in generator.ifs):
                        result.append(expr(node.elt))
                return bound(result)
            finally:
                if existed:
                    env[name] = old
                else:
                    env.pop(name, None)
        if isinstance(node, ast.IfExp):
            return expr(node.body if expr(node.test) else node.orelse)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            args = [expr(a) for a in node.args]
            name = node.func.id
            if name == "str":
                if len(args) != 1:
                    raise ValueError("str requires one JSON value")
                parts, active = [], set()
                size = 0

                def emit(text):
                    nonlocal size
                    size += len(text)
                    if size > 10000:
                        raise ValueError("Value too large: str conversion exceeds 10000 characters")
                    tick(len(text))
                    parts.append(text)

                def render(value, nested=False):
                    tick()
                    bound(value)
                    if type(value) in {str, int, float, bool, type(None)}:
                        emit(repr(value) if nested else str(value))
                        return
                    if type(value) not in {list, tuple, dict}:
                        raise ValueError("str requires one JSON value")
                    if id(value) in active or len(active) >= 64:
                        raise ValueError("str conversion requires acyclic data with depth below 64")
                    active.add(id(value))
                    try:
                        mapping = isinstance(value, dict)
                        emit("{" if mapping else "[" if isinstance(value, list) else "(")
                        for index, item in enumerate(value):
                            if index:
                                emit(", ")
                            render(item, True)
                            if mapping:
                                emit(": ")
                                render(value[item], True)
                        if isinstance(value, tuple) and len(value) == 1:
                            emit(",")
                        emit("}" if mapping else "]" if isinstance(value, list) else ")")
                    finally:
                        active.remove(id(value))

                render(args[0])
                return "".join(parts)
            if name == "range":
                value = range(*args)
                if len(value) > 10000:
                    raise ValueError("Range too large")
                return list(value)
            if name in {'find', 'rfind'}:
                if (not 2 <= len(args) <= 4 or not all(type(x) is str for x in args[:2])
                        or any(type(x) is not int for x in args[2:])):
                    raise ValueError('find/rfind require text, substring and optional integer start/end')
                text, needle = args[:2]
                start = args[2] if len(args) > 2 else 0
                end = args[3] if len(args) > 3 else len(text)
                lo, hi, _ = slice(start, end).indices(len(text))
                tick(max(0, hi - lo) + len(needle))
                return (text.find if name == 'find' else text.rfind)(needle, start, end)
            if name == 'sorted' and len(args) == 1:
                values = args[0]
                if (not isinstance(values, (list, tuple)) or len(values) > 10000
                        or any(type(value) not in {str, int, float, bool, type(None)} for value in values)):
                    raise ValueError('sorted requires a bounded list of JSON scalars')
                tick(len(values) * max(1, len(values).bit_length()))
                return bound(sorted(values))
            funcs = {"abs": abs, "min": min, "max": max, "len": len, "round": round,
                     "int": int, "float": float, "sqrt": math.sqrt, "floor": math.floor}
            if name in funcs:
                return bound(funcs[name](*args))
            raise ValueError(f"Unsupported numerical function {name!r} at line {node.lineno}; "
                             "use functions listed in the code.run catalog")
        raise ValueError(f"Unsupported numerical syntax: {type(node).__name__}")

    def statements(nodes, *, in_loop=False):
        for node in nodes:
            tick()
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name == "inputs" or (name.startswith("_") and name != "_"):
                    raise ValueError("Reserved variable")
                env[name] = expr(node.value)
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], ast.Subscript)):
                value = expr(node.value)
                target = node.targets[0]
                container = expr(target.value)
                if isinstance(target.slice, ast.Slice):
                    raise ValueError('Indexed assignment requires an existing list index, not a slice')
                index = expr(target.slice)
                if (not isinstance(container, list) or type(index) is not int
                        or type(value) not in {str, int, float, bool, type(None)}):
                    raise ValueError('Indexed assignment supports existing list slots and JSON scalar values only')
                # Inputs were deep-copied on entry. Existing slots only: no
                # resizing, container aliases/cycles, or host-object mutation.
                container[index] = value
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                raise ValueError(f"Unsupported assignment at line {node.lineno}: use one plain variable target "
                                 "(name = expression) or an existing list slot with a JSON scalar; chained, destructured, annotated and augmented "
                                 "assignments are unavailable. Use supported list.append or a fresh result value.")
            elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == 'append' and len(node.value.args) == 1
                    and not node.value.keywords):
                # Interpret this one list operation; never dispatch arbitrary
                # Python attributes/methods or expose objects from the host.
                target = expr(node.value.func.value)
                if not isinstance(target, list):
                    raise ValueError('append requires a bounded JSON list')
                value = expr(node.value.args[0])
                if len(target) >= 10000:
                    raise ValueError(f'Value too large at line {node.lineno}: list append exceeds 10000 items; '
                                     'aggregate or select items instead of collecting all of them')
                pending = [value]
                while pending:
                    tick()
                    part = pending.pop()
                    if part is target:
                        raise ValueError('Cyclic list append is not allowed')
                    if isinstance(part, dict):
                        pending.extend(part.values())
                    elif isinstance(part, (list, tuple)):
                        pending.extend(part)
                target.append(value)
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name) and not node.orelse:
                if node.target.id == "inputs" or (node.target.id.startswith("_") and node.target.id != "_"):
                    raise ValueError("Reserved loop variable")
                values = expr(node.iter)
                if isinstance(values, str):
                    bound(values)
                if not isinstance(values, (str, list, tuple)):
                    raise ValueError("Loop requires bounded sequence")
                for value in values:
                    tick()
                    env[node.target.id] = value
                    control = statements(node.body, in_loop=True)
                    if control == 'break':
                        break
                    # continue ends only this body's statements; the enclosing
                    # bounded for still charges its next iteration normally.
            elif isinstance(node, ast.If):
                control = statements(node.body if expr(node.test) else node.orelse, in_loop=in_loop)
                if control is not None:
                    return control
            elif isinstance(node, (ast.Break, ast.Continue)):
                if not in_loop:
                    raise ValueError('break/continue require an enclosing for loop')
                return 'break' if isinstance(node, ast.Break) else 'continue'
            elif isinstance(node, ast.Assert):
                if not expr(node.test):
                    raise AssertionError("Numerical assertion failed")
            else:
                raise ValueError(f"Unsupported statement: {type(node).__name__}")

    statements(tree.body)
    if "result" not in env:
        raise ValueError("Program must assign result")
    encoded = json.dumps(env["result"], allow_nan=False)
    if len(encoded.encode()) > 16000:
        raise ValueError("Numerical result too large")
    return json.loads(encoded)
