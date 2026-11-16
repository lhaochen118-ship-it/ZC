#!/usr/bin/env python3
"""Remove # comments and docstrings from Python files in place (code unchanged: AST compared).
Docstrings are kept in files listed with --keep-doc (their __doc__ is read at run time)."""
import ast, io, sys, tokenize


def _norm(src):
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            b0 = n.body[0]
            if isinstance(b0, ast.Expr) and isinstance(getattr(b0, "value", None), ast.Constant) and isinstance(b0.value.value, str):
                n.body = n.body[1:] or [ast.Pass()]
    return ast.dump(tree)


def strip(src, keep_doc=False):
    lines = src.splitlines(keepends=True)
    drop = set()
    repl = {}
    if not keep_doc:
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
                b0 = n.body[0]
                if isinstance(b0, ast.Expr) and isinstance(getattr(b0, "value", None), ast.Constant) and isinstance(b0.value.value, str):
                    a, z = b0.lineno - 1, b0.end_lineno - 1
                    tail = lines[z][b0.end_col_offset:].strip()
                    if lines[a][:b0.col_offset].strip() or (tail and not tail.startswith("#")):
                        continue
                    if len(n.body) == 1:
                        repl[a] = " " * b0.col_offset + "pass\n"
                        drop.update(range(a + 1, z + 1))
                    else:
                        drop.update(range(a, z + 1))
    out = []
    comments = {}
    for tk in tokenize.generate_tokens(io.StringIO(src).readline):
        if tk.type == tokenize.COMMENT and not (tk.start[0] == 1 and tk.string.startswith("#!")):
            comments.setdefault(tk.start[0] - 1, []).append(tk.start[1])
    for i, l in enumerate(lines):
        if i in drop: continue
        if i in repl: out.append(repl[i]); continue
        if i in comments:
            c = min(comments[i])
            body = l[:c].rstrip()
            if not body.strip(): continue
            l = body + "\n"
        out.append(l)
    res = "".join(out)
    while "\n\n\n\n" in res: res = res.replace("\n\n\n\n", "\n\n\n")
    return res


if __name__ == "__main__":
    keep = set(sys.argv[sys.argv.index("--keep-doc") + 1].split(",")) if "--keep-doc" in sys.argv else set()
    args = sys.argv[1:]
    if "--keep-doc" in args: i = args.index("--keep-doc"); args = args[:i] + args[i + 2:]
    files = [f for f in args if f.endswith(".py")]
    bad = 0
    for f in files:
        src = open(f, encoding="utf-8").read()
        kd = any(f.endswith("/" + k) for k in keep)
        new = strip(src, kd)
        try:
            same = _norm(src) == _norm(new)
        except SyntaxError as e:
            print("SYNTAX", f, e); bad += 1; continue
        if not same: print("AST DIFF", f); bad += 1; continue
        open(f, "w", encoding="utf-8").write(new)
    print(f"stripped {len(files) - bad} files, {bad} refused")
    sys.exit(1 if bad else 0)
