#!/usr/bin/env python3
"""Plan 4 W (docs/17): executable rule units written by the model, on CPU with a fake model.
 1. src/lrl_gen.py --mode unit on a mini public split (dev_012, repair shop: its rulebook industry section): one prompt per row,
    3 samples; the fake writer returns a working unit, a unit that raises, a syntax error and a forbidden import -> only working
    units are kept (one per row), each verified on the shop's own data;
 2. src/units.py: load / bind / verify / block / spec entries; an unsafe unit is refused;
 3. the flow: agent --units -> the unit appears in # SHOP FUNCTIONS, precheck knows its name, and a program calling it runs on the
    CPU workers (answer = the unit's value);
 4. predict_flow.prepare_units: a writer over the config's units_max_s is killed and the run goes on without units.
Usage: test_units.py <public_split> <scratch>"""
import os, sys, json, shutil, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)
import units

split, scratch = sys.argv[1], sys.argv[2]
mini = os.path.join(scratch, "mini_units"); shutil.rmtree(mini, ignore_errors=True); os.makedirs(os.path.join(mini, "data"))
for f in ("rulebook.md", "calendar.csv"): shutil.copy(os.path.join(split, f), mini)
os.symlink(os.path.abspath(os.path.join(split, "data", "dev_012")), os.path.join(mini, "data", "dev_012"))
qs = [l for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8") if json.loads(l)["business_id"] == "dev_012"][:6]
open(os.path.join(mini, "questions.jsonl"), "w", encoding="utf-8").writelines(qs)

WRITER = os.path.join(scratch, "fake_writer.py")
open(WRITER, "w").write('''import json, re, zlib
def responder(conv):
    u = conv[1]["content"]                                         # the row's prompt (a repair conversation ends with the error)
    term = re.findall(r"# DÒNG RULEBOOK CẦN VIẾT HÀM.*?\\n\\| (.*?) \\|", u, re.S)[0]
    h = zlib.crc32((term + str(len(conv))).encode()) % 4
    name = "u_" + str(zlib.crc32(term.encode()) % 100000)
    codes = ["def %s(p):\\n    return n_orders(p) + 1" % name,
             "def %s(p):\\n    raise ValueError('broken')" % name,
             "def %s(p)\\n    return 1" % name,
             "def %s(p):\\n    import os\\n    return 1" % name]
    return json.dumps({"name": name, "signature": name + "(p)", "doc": "test", "triggers": [term], "code": codes[h]})
''')
# the writer is called with the same conversation for the 3 samples: make the variants differ by sample index (temperature)
out = os.path.join(scratch, "units_out"); shutil.rmtree(out, ignore_errors=True)
r = subprocess.run([sys.executable, os.path.join(SRC, "lrl_gen.py"), mini, out, "--mode", "unit", "--fake-responder", WRITER, "--samples", "3"],
                   capture_output=True, text=True, timeout=900)
assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
U = units.load(os.path.join(out, "cards_unit.jsonl"))
raw = [json.loads(l) for l in open(os.path.join(out, "raw_unit.jsonl"), encoding="utf-8")]
assert raw, "no unit prompt was written for the repair shop's industry rows"
assert all("return n_orders(p) + 1" in u["code"] for u in U), [u["code"] for u in U]
assert len({(u["bid"], u["term"]) for u in U}) == len(U)
print(f"[1] {len(raw)} unit samples -> {len(U)} verified units (only the working variant kept)", flush=True)

# 2. the library API
ok, why = units.safe_code("def f(p):\n    import os\n    return 1")
assert not ok and "import" in why
ok, why = units.safe_code("def f(p):\n    return open('x').read()")
assert not ok
spec = units.spec_entries(U)
assert spec and spec[0]["name"] == U[0]["name"] and spec[0]["params"][0]["name"] == "p"
assert U[0]["signature"] in units.block(U)
g = {"n_orders": lambda p: 41}
assert units.bind(g, U[:1]) == [U[0]["name"]] and g[U[0]["name"]](None) == 42
print("[2] load / bind / verify / block / spec OK", flush=True)

# 3. the flow with the units: a fake solver calls the first unit
name = U[0]["name"]
SOLVER = os.path.join(scratch, "fake_solver.py")
open(SOLVER, "w").write(f'''def responder(conv):
    u = conv[-1]["content"]
    assert "# SHOP FUNCTIONS" in conv[1]["content"] or "# SHOP FUNCTIONS" in u, "units block missing"
    return '<plan>unit | - | -</plan>\\n<code>\\nANSWER = {{"type": "number", "answer": {name}(this_month(TODAY))}}\\n</code>'
''')
res = os.path.join(scratch, "units_flow"); shutil.rmtree(res, ignore_errors=True)
cmd = [sys.executable, os.path.join(SRC, "agent.py"), mini, res, "--flow", "hybrid", "--pipeline", "--pure", "1", "--hints", "0", "--grammar", "0",
       "--k-fast", "1", "--k-max", "1", "--units", os.path.join(out, "cards_unit.jsonl"), "--fake-responder", SOLVER, "--workers", "2",
       "--prefer-map", "fmt:automap", "--cache", os.path.join(scratch, "units_cache")]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
logs = [json.loads(l) for l in open(os.path.join(res, "logs.jsonl"), encoding="utf-8")]
assert logs and all(x["final"].get("type") == "number" for x in logs), [x["final"] for x in logs][:3]
pre = [t.get("precheck") for x in logs for s in x["samples"]["samples"] for t in s["turns"]]
assert not any(p and any(str(i).startswith("error:name") for i in p) for p in pre), pre[:3]
print(f"[3] flow: {len(logs)} questions answered by a program calling the unit {name}", flush=True)

# 4. the container path (predict_flow.prepare_units): a writer slower than the config's cap is killed (process group) and the run
#    goes on without units
import time, predict_flow
SLOW = os.path.join(scratch, "slow_writer.py"); open(SLOW, "w").write("import time\ndef responder(conv):\n    time.sleep(120)\n    return '{}'\n")
cfg = os.path.join(scratch, "cfg_units_cap.json"); json.dump({"units_gen": 1, "units_samples": 1, "units_max_s": 4}, open(cfg, "w"))
t = time.time(); uv = predict_flow.prepare_units(cfg, mini, os.path.join(scratch, "units_cap"), "/fake/model", lambda *m: None, SLOW)
assert uv is None and time.time() - t < 60, (uv, time.time() - t)
print(f"[4] writer over its time cap killed after {time.time() - t:.0f}s, run continues without units", flush=True)
print("test_units PASS")
