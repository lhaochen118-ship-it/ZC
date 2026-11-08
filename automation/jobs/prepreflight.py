"""run automation/preflight.py ONCE for the jobs of one or more jobs files and, on success, write fanout.py's cache mark of
each file (same hash as fanout.py: automation/codehash.py). Usage: prepreflight.py <jobs file or string> [<jobs file> ...]"""
import sys, os, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "automation")); import codehash
specs = sys.argv[1:]
per = [codehash.read_jobs(sp) for sp in specs]
marks = [codehash.mark_path(js) for js in per]                         # hashed BEFORE the run: a file edited meanwhile is not marked
alljobs = []
for js in per: alljobs += [j for j in js if j.startswith("run_kaggle.py") and j not in alljobs]
r = subprocess.run([sys.executable, os.path.join(ROOT, "automation", "preflight.py"), ";".join(alljobs)], capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-1500:], flush=True)
if r.returncode == 0:
    for sp, js, m in zip(specs, per, marks):
        if codehash.mark_path(js) != m: print("CODE CHANGED during the preflight: no mark for", sp); continue
        os.makedirs(os.path.dirname(m), exist_ok=True); open(m, "w").write("\n".join(js)); print("MARK", sp, m)
else: print("PREFLIGHT FAILED")
