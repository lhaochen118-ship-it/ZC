"""grid_report.py <runs dir> [run ...]: per v2 public run (results/<run>/<cfg>/submission.jsonl): container PA, ensemble PA, pub_ind,
units, paired diff (container) vs H0a and H0a2 ensemble files. Sorted by container PA."""
import os, sys, json, glob, math, re
sys.path.insert(0, "src"); sys.path.insert(0, "Process/tools")
import score
R = score.load("data/raw/public_gt.jsonl")
ind = set(l.strip() for l in open("tests/splits/pub_ind.txt") if l.strip())
def sc(f):
    out = {}
    for l in open(f):
        if not l.strip(): continue
        a = json.loads(l)
        if a.get("type") == "abstain": out[a["id"]] = 0; continue
        try: out[a["id"]] = 1 if score.judge(a, R[a["id"]]) == "correct" else -1
        except Exception: out[a["id"]] = -1
    return out
D = sys.argv[1]
base = {}
for b in ("h0a", "h0a2"):
    f = glob.glob(f"{D}/{b}/*_out/results/{b}/*/judge/ensemble/submission.jsonl")
    if f: base[b] = sc(f[0])
rows = []
runs = sys.argv[2:] or sorted(os.listdir(D))
for r in runs:
    fs = glob.glob(f"{D}/{r}/*_out/results/{r}/*/submission.jsonl")
    if not fs: continue
    c = sc(fs[0]); e = glob.glob(os.path.dirname(fs[0]) + "/judge/ensemble/submission.jsonl")
    ens = sc(e[0]) if e else None
    pa = sum(c.values()) / 1000; pe = sum(ens.values()) / 1000 if ens else float("nan")
    pi = sum(v for k, v in c.items() if k in ind) / len(ind)
    u = ""
    lg = os.path.dirname(fs[0]) + "/eval.log"
    if os.path.isfile(lg):
        m = re.findall(r"units: rc=.*?(\d+) verified units|units: reused", open(lg, errors="replace").read()); u = m[-1] if m else ""
        v = re.findall(r"unit verify .*?kept (\d+) / (\d+)", open(lg, errors="replace").read()); u += f" v{v[-1][0]}/{v[-1][1]}" if v else ""
    pd = []
    for b, bs in base.items():
        d = [c.get(q, 0) - bs.get(q, 0) for q in R]; m = sum(d) / len(d); sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** .5
        pd.append(f"{m:+.3f}(z{m / (sd / math.sqrt(len(d))) if sd else 0:+.1f})")
    rows.append((pa, r, os.path.basename(os.path.dirname(fs[0])), pe, pi, u, " ".join(pd)))
print(f"{'run':10s} {'config':16s} {'cont':>6s} {'ens':>6s} {'ind':>6s} units      paired vs H0a / H0a2 (ensemble)")
for pa, r, cfg, pe, pi, u, pd in sorted(rows, reverse=True):
    print(f"{r:10s} {cfg[:16]:16s} {pa:6.3f} {pe:6.3f} {pi:6.3f} {u:10s} {pd}")
