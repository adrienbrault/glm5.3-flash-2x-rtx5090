"""CPU-only checks for mtp-fast.patch: the patched module compiles, and _join_host picks only an
unstarted, same-checkpoint, same-thread-count host with uniform expert dims (else None, which
keeps the unpatched one-worker-per-component behaviour). Run: python3 -I test_join_host.py <patched block_sparse_mlp_cpu.py>"""
import ast, sys, types, py_compile

path = sys.argv[1]
py_compile.compile(path, doraise = True)
src = open(path).read()
tree = ast.parse(src)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_join_host")
ns = {}
exec(compile(ast.Module(body = [fn], type_ignores = []), path, "exec"), ns)
join = ns["_join_host"]

def host(started = False, model_dir = "/m", threads = 8, specs = None):
    return types.SimpleNamespace(started = started, model_dir = model_dir, threads = threads,
                                 specs = specs if specs is not None else [dict(hi = 4096, ho = 4096, topk = 8)])

h = host()
assert join({"mtp": h}, "/m", 8, 4096, 4096, 8) == ("mtp", h)                 # TabbyAPI order: MTP first
assert join({}, "/m", 8, 4096, 4096, 8) == (None, None)                      # nothing to join
assert join({"text": host(started = True)}, "/m", 8, 4096, 4096, 8) == (None, None)  # started: fallback
assert join({"mtp": host(model_dir = "/other")}, "/m", 8, 4096, 4096, 8) == (None, None)
assert join({"mtp": host(threads = 4)}, "/m", 8, 4096, 4096, 8) == (None, None)
assert join({"mtp": host()}, "/m", 8, 4096, 2048, 8) == (None, None)        # non-uniform dims
assert join({"mtp": host()}, "/m", 8, 4096, 4096, 6) == (None, None)        # non-uniform top-k
e = host(specs = [])
assert join({"mtp": e}, "/m", 8, 4096, 4096, 8) == ("mtp", e)               # empty host is joinable
s, u = host(started = True), host()
assert join({"text": s, "mtp": u}, "/m", 8, 4096, 4096, 8) == ("mtp", u)    # skips the started one
print("test_join_host: 9/9 OK")
