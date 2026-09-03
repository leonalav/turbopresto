"""Calculator / code-execution tool for inference.

Why this exists
───────────────
Per /ipho-physicist: a 50M-param model cannot reliably perform exact
multi-digit arithmetic in its head.  The carry propagates right-to-left
while the tokenizer reads digits left-to-right, so by the time the
model emits a digit for a column, it does not yet know whether a carry
is coming from the right.

This is the same failure mode observed in the addition task of the
Neural Tangent Kernel / recursion-paper family: small models fail at
multi-digit arithmetic not because they cannot learn the algorithm,
but because the supervision signal is too sparse for the carry
recurrence to be discovered end-to-end at this parameter count.

TinyGSM and Orca-Math both attack this by offloading arithmetic to a
Python interpreter.  At 50M params we are firmly in that regime.

This module exposes a tiny sandboxed expression evaluator that the
generator can call when the model emits a <TOOL>...</TOOL> block.
The result is inserted back into the generation stream.

Safety
──────
The evaluator is restricted to:
  • Numeric literals (integers, floats, scientific notation)
  • The arithmetic operators  +  -  *  /  //  %  **
  • Parentheses for grouping
  • The math functions abs, round, min, max, sqrt, log, exp, sin,
    cos, tan, floor, ceil (whitelisted via the `math` module)
  • Comparisons  ==  !=  <  >  <=  >=  (return Python bools)

It rejects anything else (no attribute access, no imports, no
subscript, no calls except to the whitelist, no comprehensions).
This is **not** a complete Python sandbox — it is a regex/AST-level
filter that is appropriate for short arithmetic expressions produced
by the model.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Optional, Tuple

# Whitelisted AST visitors
_BIN_OPS: dict = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CMP_OPS: dict = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_UNARY_OPS: dict = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_WHITELIST_FUNCS: dict = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": math.pow,
    "factorial": math.factorial,
    "gcd": math.gcd,
}


class CalculatorError(ValueError):
    """Raised when an expression fails safety or evaluation checks."""


def safe_eval(expr: str) -> object:
    """Evaluate a whitelisted arithmetic expression.

    Args:
        expr: A string like "(2 + 3) * 4" or "sqrt(2)".

    Returns:
        A number (int / float / bool).

    Raises:
        CalculatorError: if the expression is unsafe or invalid.
    """
    if not isinstance(expr, str):
        raise CalculatorError("Expression must be a string.")
    expr = expr.strip()
    if not expr:
        raise CalculatorError("Empty expression.")
    if len(expr) > 512:
        raise CalculatorError("Expression too long.")
    # Disallow obviously dangerous characters
    if any(ch in expr for ch in (";", "_", "import", "lambda", "class",
                                  "@", "\\", "\n", "\r", "$")):
        raise CalculatorError("Disallowed character or keyword.")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Syntax error: {exc}") from exc

    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> object:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, bool)):
            raise CalculatorError(
                f"Only numeric literals allowed (got {type(node.value).__name__})"
            )
        return node.value

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise CalculatorError(f"Operator {op_type.__name__} not allowed.")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(left, bool) or isinstance(right, bool):
            # bool is a subclass of int, but arithmetic on bools is sketchy
            if not (isinstance(left, bool) and isinstance(right, bool)):
                # mixed bool+num: allow (Python does)
                pass
        return _BIN_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise CalculatorError(f"Unary {op_type.__name__} not allowed.")
        return _UNARY_OPS[op_type](_eval_node(node.operand))

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise CalculatorError("Chained comparisons not allowed.")
        op_type = type(node.ops[0])
        if op_type not in _CMP_OPS:
            raise CalculatorError(f"Comparison {op_type.__name__} not allowed.")
        left = _eval_node(node.left)
        right = _eval_node(node.comparators[0])
        return _CMP_OPS[op_type](left, right)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise CalculatorError("Only direct function calls allowed.")
        if node.func.id not in _WHITELIST_FUNCS:
            raise CalculatorError(f"Function {node.func.id!r} not in whitelist.")
        if node.keywords:
            raise CalculatorError("Keyword args not allowed.")
        args = [_eval_node(a) for a in node.args]
        return _WHITELIST_FUNCS[node.func.id](*args)

    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(elt) for elt in node.elts)

    raise CalculatorError(f"Node type {type(node).__name__} not allowed.")


# ---------------------------------------------------------------------------
# Tool-call extraction
# ---------------------------------------------------------------------------

_TOOL_RE = re.compile(
    r"<TOOL>\s*(calc|exec)\s*\((?P<expr>[^)]*)\)\s*</TOOL>",
    re.IGNORECASE | re.DOTALL,
)


def extract_tool_call(text: str) -> Optional[Tuple[str, str, int, int]]:
    """Return (tool, expr, start, end) for the first <TOOL>...</TOOL> in text.

    Returns None if no tool call is present.
    """
    m = _TOOL_RE.search(text)
    if m is None:
        return None
    tool = m.group(1).lower()
    expr = m.group("expr").strip()
    return tool, expr, m.start(), m.end()


def call_calculator(expr: str) -> str:
    """Evaluate `expr` and return the result formatted as a string.

    Args:
        expr: The expression inside <TOOL>calc(...)</TOOL>.

    Returns:
        Stringified result, e.g. "579" or "1.4142135623730951".

    Raises:
        CalculatorError: if the expression is unsafe or fails to evaluate.
    """
    value = safe_eval(expr)
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        # Trim trailing zeros for readability
        if value.is_integer():
            return str(int(value))
        return repr(value)
    return str(value)


def try_call(text: str) -> Tuple[str, str]:
    """If `text` contains a tool call, replace it with the result.

    Iterates over every ``<TOOL>calc(...)</TOOL>`` block in ``text`` and
    resolves them in left-to-right order.  Each iteration evaluates one
    call and splices its numeric result in; we then rescan the spliced
    text so a subsequent call (whose expression may itself contain a
    numeral produced by a previous call, in principle) can be caught.

    A bounded number of iterations (``MAX_TOOL_CALLS_PER_PASS = 64``)
    prevents a pathological malformed string from pinning generation on
    an infinite loop.  We also defend against a stuck iteration: if two
    consecutive passes splice identical text (e.g. the regex finds an
    always-present tool call for some reason), we bail with status
    ``"stuck"`` rather than spinning forever.

    This is the simpler of two viable strategies and is what ``RWKVGenerator``
    uses today (see ``src/inference/generation.py``).  The other strategy --
    pause generation when ``</TOOL>`` closes, evaluate, inject the result as
    real tokens, and resume the recurrent state -- is currently out of scope
    and noted as a known limitation in that module.  Iterating over all
    calls rather than only the first already fixes the most common
    failure mode: a multi-step CoT that ends with a tool call no longer
    drops intermediate calls.

    Args:
        text: Generated text possibly containing ``<TOOL>calc(...)</TOOL>``.

    Returns:
        (new_text, status) where status is one of:
          - "no_call"  -- no ``<TOOL>`` block was ever present
          - "ok"       -- every ``<TOOL>`` block was found and executed
          - "error"    -- a tool call was found but evaluation failed;
                          earlier successful calls (if any) are kept,
                          the failing text is returned as-is, and an
                          evaluation error is signalled
          - "stuck"    -- the bounded loop hit ``MAX_TOOL_CALLS_PER_PASS``
                          without making further progress on the final
                          pass (should never happen on real text but
                          guards against adversarial input)
    """
    MAX_TOOL_CALLS_PER_PASS = 64
    current = text
    # Fast path: no tool call at all.
    if extract_tool_call(current) is None:
        return current, "no_call"

    for _ in range(MAX_TOOL_CALLS_PER_PASS):
        found = extract_tool_call(current)
        if found is None:
            return current, "ok"
        _tool, _expr, start, end = found
        try:
            result = call_calculator(_expr)
        except CalculatorError:
            # We had at least one tool call (we checked above), but this
            # particular one failed.  Return whatever ``current`` looks
            # like now (which may include earlier successful resolutions)
            # with status ``"error"``.  For the very-first-iteration case
            # this is unchanged from the legacy contract.
            return current, "error"
        new_current = current[:start] + result + current[end:]
        if new_current == current:
            # Made no progress -- guard against infinite loop.
            return current, "stuck"
        current = new_current
    # Hit the bounded maximum without running out of calls -- defend
    # against adversarial inputs.
    return current, "stuck"


# ---------------------------------------------------------------------------
# SFT data helpers — column-by-column traces
# ---------------------------------------------------------------------------

def column_cot_addition(a: int, b: int) -> str:
    """Generate a column-by-column CoT for a + b, suitable for SFT.

    The trace spells out each column with carry state explicitly, so the
    model never has to do multi-digit arithmetic internally.

    Columns are emitted in the model's natural reading order
    (ones → tens → hundreds → ...), so the model can learn the carry
    recurrence by attending to its own previous column's carry state.

    Example for 567 + 489:

        ones:        7 + 9 = 16, write 6, carry 1
        tens:        6 + 8 + 1 = 15, write 5, carry 1
        hundreds:    5 + 4 + 1 = 10, write 0, carry 1
        thousands:   0 + 0 + 1 = 1,   write 1
        Result: 1056

    Example for 12345 + 678:

        ones:        1 + 0 = 1,   write 1
        tens:        4 + 7 = 11,  write 1, carry 1
        hundreds:    3 + 6 + 1 = 10, write 0, carry 1
        thousands:   2 + 0 + 1 = 3,   write 3
        ten-thousands: 1 + 0 = 1,   write 1
        Result: 13023
    """
    sa, sb = str(a), str(b)
    n = max(len(sa), len(sb))
    sa = sa.zfill(n)
    sb = sb.zfill(n)
    place_names = ["ones", "tens", "hundreds", "thousands",
                   "ten-thousands", "hundred-thousands",
                   "millions", "ten-millions", "hundred-millions"]
    lines = []
    carry = 0
    digits_out: list[str] = []
    # Iterate least-significant column first.
    # String index `i = n-1` is the ones column; `i = n-2` is tens; etc.
    # `place_idx = n-1-i` so the place label matches the column we read.
    for i in range(n - 1, -1, -1):
        place_idx = n - 1 - i
        da = int(sa[i])
        db = int(sb[i])
        incoming_carry = carry
        s = da + db + incoming_carry
        write = s % 10
        carry = s // 10
        place = place_names[place_idx] if place_idx < len(place_names) \
                else f"10^{place_idx}"
        # Show incoming carry if nonzero; outgoing carry in carry_note.
        expr = f"{da}+{db}" + (f"+{incoming_carry}" if incoming_carry else "")
        carry_note = f", carry {carry}" if carry else ""
        lines.append(f"{place:<14} {expr} = {s}, write {write}{carry_note}")
        digits_out.append(str(write))
    if carry:
        # implicit 0+0+carry column at the next-higher place
        place_idx = n
        place = place_names[place_idx] if place_idx < len(place_names) \
                else f"10^{place_idx}"
        lines.append(f"{place:<14} 0+0+{carry} = {carry}, write {carry}")
        digits_out.append(str(carry))
    digits_out.reverse()
    return "\n".join(lines) + "\nResult: " + "".join(digits_out)


def column_cot_multiplication(a: int, b: int) -> str:
    """Column-by-column CoT for a * b, one row per digit of b.

    Iterates over the digits of `b` from rightmost (ones) to leftmost,
    matching the model's natural reading order.
    """
    sb = str(b)
    lines = []
    partials: list[int] = []
    for i, ch in enumerate(reversed(sb)):
        d = int(ch)
        row = a * d * (10 ** i)
        partials.append(row)
        shift_note = f" (×10^{i})" if i else ""
        lines.append(f"{a} × {d}{shift_note} = {row}")
    if len(partials) > 1:
        lines.append(" + ".join(str(p) for p in partials) + f" = {sum(partials)}")
    return "\n".join(lines) + f"\nResult: {a * b}"


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. safe_eval
    assert safe_eval("2 + 3") == 5
    assert safe_eval("(2 + 3) * 4") == 20
    assert safe_eval("sqrt(16)") == 4.0
    assert safe_eval("abs(-7)") == 7
    assert safe_eval("2 ** 10") == 1024
    assert safe_eval("3 == 3") is True
    try:
        safe_eval("__import__('os')")
        raise AssertionError("should have raised")
    except CalculatorError:
        pass
    try:
        safe_eval("open('x')")
        raise AssertionError("should have raised")
    except CalculatorError:
        pass
    print("[safe_eval] all sanity tests passed.")

    # 2. try_call
    txt = "The answer is <TOOL>calc(567 + 489)</TOOL>."
    new, status = try_call(txt)
    print(f"[try_call] {txt!r} -> ({status!r}) {new!r}")
    assert status == "ok" and "1056" in new

    # 3. column_cot_addition
    print("\n[column_cot_addition(567, 489)]:")
    print(column_cot_addition(567, 489))
    print("\n[column_cot_addition(12345, 678)]:")
    print(column_cot_addition(12345, 678))

    # 4. column_cot_multiplication
    print("\n[column_cot_multiplication(23, 47)]:")
    print(column_cot_multiplication(23, 47))
