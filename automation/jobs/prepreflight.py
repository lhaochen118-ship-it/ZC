"""run automation/preflight.py for a jobs file and write fanout.py's cache mark on success (same hash as fanout.py)."""
import sys, os, glob, hashlib, subprocess
ROOT = '/home/user/ZC'; jobs = sys.argv[1]
jobs = [j.strip() for j in (open(jobs).read().replace("\n", ";") if os.path.isfile(jobs) else jobs).split(";") if j.strip()]
h = hashlib.sha256(";".join(jobs).encode())
for f in sorted(f for f in glob.glob(os.path.join(ROOT, "src", "**", "*"), recursive=True) + glob.glob(os.path.join(ROOT, "automation", "*.py")) +
                glob.glob(os.path.join(ROOT, "tests", "*")) if os.path.isfile(f) and "__pycache__" not in f):
    h.update(f.encode()); h.update(open(f, "rb").read())
mark = os.path.expanduser(f"~/.cache/zc_preflight/{h.hexdigest()[:16]}")
pf = [j for j in jobs if j.startswith("run_kaggle.py")]
r = subprocess.run([sys.executable, os.path.join(ROOT, "automation", "preflight.py"), ";".join(pf)], capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-800:], flush=True)
if r.returncode == 0: os.makedirs(os.path.dirname(mark), exist_ok=True); open(mark, "w").write("\n".join(jobs)); print("MARK", mark)
else: print("PREFLIGHT FAILED")
