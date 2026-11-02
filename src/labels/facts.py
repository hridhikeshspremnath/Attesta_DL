"""Per-file static facts. Python via ast, JS via esprima; ok=False means 'parse failed, use regex'."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

import esprima

MAX_JS_AST_BYTES = 120_000


@dataclass
class Facts:
    path: str
    lang: str                      # py | js | other
    text: str
    ok: bool = False               # AST parse succeeded
    imports: set = field(default_factory=set)
    calls: set = field(default_factory=set)         # dotted, alias-resolved
    toplevel_calls: set = field(default_factory=set)  # py: calls outside def/class bodies
    strings: list = field(default_factory=list)
    env_keys: list = field(default_factory=list)
    env_bulk: bool = False         # whole environment passed/copied
    class_bases: list = field(default_factory=list)
    kwargs: set = field(default_factory=set)       # keyword names seen in calls (py)


def lang_of(path: str) -> str:
    p = path.lower()
    return "py" if p.endswith(".py") else "js" if p.endswith((".js", ".mjs", ".cjs")) else "other"


# ---- python ----------------------------------------------------------------
def _pydotted(n: ast.AST, alias: dict) -> str:
    if isinstance(n, ast.Name):
        return alias.get(n.id, n.id)
    if isinstance(n, ast.Attribute):
        base = _pydotted(n.value, alias)
        return f"{base}.{n.attr}" if base else ""
    return ""


def _py_walk(node, f: Facts, alias: dict, in_def: bool) -> None:
    for child in ast.iter_child_nodes(node):
        deeper = in_def or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
        if isinstance(child, ast.Call):
            d = _pydotted(child.func, alias)
            if d:
                f.calls.add(d)
                if not in_def:
                    f.toplevel_calls.add(d)
            f.kwargs.update(k.arg for k in child.keywords if k.arg)
            if d in ("os.environ.get", "os.getenv") and child.args and isinstance(child.args[0], ast.Constant):
                f.env_keys.append(str(child.args[0].value))
            if any(_pydotted(a, alias) == "os.environ" for a in child.args):
                f.env_bulk = True
            if d == "os.environ.copy":
                f.env_bulk = True
        elif isinstance(child, ast.Subscript) and _pydotted(child.value, alias) == "os.environ":
            if isinstance(child.slice, ast.Constant):
                f.env_keys.append(str(child.slice.value))
        elif isinstance(child, ast.Constant) and isinstance(child.value, str) and len(f.strings) < 4000:
            f.strings.append(child.value)
        elif isinstance(child, ast.ClassDef):
            f.class_bases += [_pydotted(b, alias) or getattr(b, "id", "") for b in child.bases]
        _py_walk(child, f, alias, deeper)


def py_facts(path: str, text: str) -> Facts:
    f = Facts(path, "py", text)
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return f
    alias: dict = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                f.imports.add(a.name)
                alias[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom) and n.module:
            f.imports.add(n.module)
            for a in n.names:
                alias[a.asname or a.name] = f"{n.module}.{a.name}"
    _py_walk(tree, f, alias, False)
    f.ok = True
    return f


# ---- javascript ------------------------------------------------------------
def _children(n):
    for v in vars(n).values():
        if isinstance(v, list):
            yield from (x for x in v if hasattr(x, "type"))
        elif hasattr(v, "type"):
            yield v


def _jsdotted(n, alias: dict) -> str:
    t = getattr(n, "type", "")
    if t == "Identifier":
        return alias.get(n.name, n.name)
    if t == "MemberExpression":
        base = _jsdotted(n.object, alias)
        if not base:
            return ""
        if n.computed:
            return f"{base}.{n.property.value}" if getattr(n.property, "type", "") == "Literal" else base
        return f"{base}.{n.property.name}"
    return ""


def _is_require(n) -> str:
    if getattr(n, "type", "") == "CallExpression" and getattr(n.callee, "type", "") == "Identifier" \
            and n.callee.name == "require" and n.arguments and getattr(n.arguments[0], "type", "") == "Literal":
        return str(n.arguments[0].value)
    return ""


def _js_aliases(root, f: Facts) -> dict:
    alias, stack = {}, [root]
    while stack:
        n = stack.pop()
        t = n.type
        if t == "VariableDeclarator" and n.init is not None:
            mod = _is_require(n.init)
            if mod:
                f.imports.add(mod)
                if n.id.type == "Identifier":
                    alias[n.id.name] = mod
                elif n.id.type == "ObjectPattern":
                    for p in n.id.properties:
                        if getattr(p.key, "name", None) and getattr(p.value, "type", "") == "Identifier":
                            alias[p.value.name] = f"{mod}.{p.key.name}"
        elif t == "ImportDeclaration":
            f.imports.add(str(n.source.value))
            for s in n.specifiers:
                alias[s.local.name] = str(n.source.value)
        stack.extend(_children(n))
    return alias


def js_facts(path: str, text: str) -> Facts:
    f = Facts(path, "js", text)
    if len(text) > MAX_JS_AST_BYTES:
        return f
    try:
        try:
            root = esprima.parseScript(text, {"tolerant": True})
        except Exception:
            root = esprima.parseModule(text, {"tolerant": True})
    except Exception:
        return f
    alias = _js_aliases(root, f)
    stack = [root]
    while stack:
        n = stack.pop()
        t = n.type
        if t == "CallExpression":
            d = _jsdotted(n.callee, alias)
            if d:
                f.calls.add(d)
            mod = _is_require(n)
            if mod:
                f.imports.add(mod)
            if any(_jsdotted(a, alias) == "process.env" for a in n.arguments):
                f.env_bulk = True
        elif t == "NewExpression":
            d = _jsdotted(n.callee, alias)
            if d:
                f.calls.add("new " + d)
        elif t == "MemberExpression":
            d = _jsdotted(n, alias)
            if d.startswith("process.env.") or (d == "process.env" and False):
                f.env_keys.append(d.split(".", 2)[2])
        elif t == "Literal" and isinstance(n.value, str) and len(f.strings) < 4000:
            f.strings.append(n.value)
        stack.extend(_children(n))
    f.ok = True
    return f


def get_facts(path: str, text: str) -> Facts:
    lang = lang_of(path)
    return py_facts(path, text) if lang == "py" else js_facts(path, text) if lang == "js" else Facts(path, lang, text)
