#!/usr/bin/env python3
"""P5 zero-shot plan report: every run under Process/runs/p5/<run>/*_out/results/<run>/<cfg>/ -> container PA (file the Docker image
writes), ensemble PA (judge file), c/w/a, pub_ind / pub_noind PA, paired difference vs B00-S (container vs container: mean of the seeds
present, z = d / SE with SE = sqrt(b + c) / n per seed file), wall time, GPU util / peak memory. Writes Process/scores_p5zs.csv.
Usage: python3 Process/tools/p5/p5_report.py [runs dir]"""
import os, sys, json, glob, math, csv
ROOT = "/home/user/ZC"; sys.path.insert(0, os.path.join(ROOT, "src"))
import score
R = score.load(os.path.join(ROOT, "data/raw/public_gt.jsonl"))
IND = set(l.strip() for l in open(os.path.join(ROOT, "tests/splits/pub_ind.txt")) if l.strip())


def sc(f):
    out = {k: 0 for k in R}
    for l in open(f, encoding="utf-8"):
        if not l.strip(): continue
        a = json.loads(l)
        if a["id"] not in R or a.get("type") in (None, "abstain"): continue
        try: out[a["id"]] = 1 if score.judge(a, R[a["id"]]) == "correct" else -1
        except Exception: out[a["id"]] = -1
    return out


def paired(x, y):
    d = [x[k] - y[k] for k in R]; n = len(d); m = sum(d) / n
    b = sum(1 for v in d if v > 0); c = sum(1 for v in d if v < 0)
    se = math.sqrt(b + c) / n if b + c else float("nan")
    return m, (m / se if se == se and se > 0 else float("nan")), b, c


def main():
    D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "Process/runs/p5")
    rows = []
    for f in sorted(glob.glob(f"{D}/*/*_out/results/*/*/submission.jsonl")):
        if "/judge/" in f: continue
        cfgd = os.path.dirname(f); run = f.split("/results/")[1].split("/")[0]; cfg = os.path.basename(cfgd)
        c = sc(f); e = os.path.join(cfgd, "judge", "ensemble", "submission.jsonl"); ens = sc(e) if os.path.isfile(e) else None
        res = json.load(open(os.path.join(cfgd, "result.json"))) if os.path.isfile(os.path.join(cfgd, "result.json")) else {}
        gs = glob.glob(os.path.join(os.path.dirname(os.path.dirname(cfgd)), "..", "gpu_summary.json")) + glob.glob(os.path.join(os.path.dirname(cfgd), "gpu_summary.json"))
        g = json.load(open(gs[0])) if gs else {}
        util = [v["mean_util"] for k, v in g.items() if k.startswith("gpu")]; mem = [v["peak_mem_gb"] for k, v in g.items() if k.startswith("gpu")]
        rows.append({"run": run, "config": cfg, "container": c, "ensemble": ens, "wall_s": g.get("wall_s") or res.get("eval_s"),
                     "util": round(sum(util) / len(util), 1) if util else None, "peak_gb": max(mem) if mem else None})
    base = [r["container"] for r in rows if r["run"] in ("b00s0", "b00s1")]
    out = []
    for r in rows:
        c = r["container"]; n = len(R)
        pa = sum(c.values()) / n; cw = (sum(v == 1 for v in c.values()), sum(v == -1 for v in c.values()), sum(v == 0 for v in c.values()))
        pe = sum(r["ensemble"].values()) / n if r["ensemble"] else float("nan")
        pi = sum(v for k, v in c.items() if k in IND) / len(IND); pn = sum(v for k, v in c.items() if k not in IND) / (n - len(IND))
        dz = [paired(c, b) for b in base if b is not c]
        d = sum(x[0] for x in dz) / len(dz) if dz else float("nan"); z = sum(x[1] for x in dz) / len(dz) if dz else float("nan")
        out.append({"run": r["run"], "config": r["config"], "PA_container": round(pa, 4), "PA_ensemble": round(pe, 4),
                    "correct": cw[0], "wrong": cw[1], "abstain": cw[2], "PA_pub_ind": round(pi, 4), "PA_pub_noind": round(pn, 4),
                    "dZS_vs_B00S": round(d, 4) if d == d else "", "z": round(z, 2) if z == z else "", "wall_min": round((r["wall_s"] or 0) / 60),
                    "gpu_util": r["util"], "peak_gb": r["peak_gb"]})
    out.sort(key=lambda x: -x["PA_container"])
    if out:
        with open(os.path.join(ROOT, "Process/scores_p5zs.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("| run | config | PA container | PA ensemble | c/w/a | pub_ind | pub_noind | ΔZS vs B00-S | z | wall min | util % | peak GB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for o in out:
        print(f"| {o['run']} | {o['config']} | {o['PA_container']} | {o['PA_ensemble']} | {o['correct']}/{o['wrong']}/{o['abstain']} | {o['PA_pub_ind']} | "
              f"{o['PA_pub_noind']} | {o['dZS_vs_B00S']} | {o['z']} | {o['wall_min']} | {o['gpu_util']} | {o['peak_gb']} |")


if __name__ == "__main__":
    main()
