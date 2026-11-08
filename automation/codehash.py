"""The code + jobs hash of the pre-run gate (fanout.py refuses to launch without a preflight mark of this hash)."""
import os, glob, hashlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def code_files():
    fs = glob.glob(os.path.join(ROOT, "src", "**", "*"), recursive=True) + glob.glob(os.path.join(ROOT, "automation", "*.py")) + \
         glob.glob(os.path.join(ROOT, "tests", "*")) + glob.glob(os.path.join(ROOT, "training", "**", "*.py"), recursive=True) + \
         glob.glob(os.path.join(ROOT, "training", "*.yaml")) + glob.glob(os.path.join(ROOT, "configs", "*"))
    return sorted(f for f in fs if os.path.isfile(f) and "__pycache__" not in f)


def mark_path(jobs):
    h = hashlib.sha256(";".join(jobs).encode())
    for f in code_files(): h.update(f.encode()); h.update(open(f, "rb").read())
    return os.path.expanduser(f"~/.cache/zc_preflight/{h.hexdigest()[:16]}")


def read_jobs(spec):
    return [j.strip() for j in (open(spec).read().replace("\n", ";") if os.path.isfile(spec) else spec).split(";") if j.strip()]
