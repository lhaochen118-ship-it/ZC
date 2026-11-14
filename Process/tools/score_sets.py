"""PA per evaluation set of one or more runs' submission.jsonl (several shards of one run are merged), PA over the set's own ids
(a missing answer counts as an abstain). Sets: pub_full, pub_ho400c, tr_ho400c, pub_noind, tr_newterm (+ pub_ho, tr_ho, pub_ho400,
tr_ho400 when present). Public sets need data/raw/public_gt.jsonl (scoring only); without it they are skipped.
Held-out (plan 3 §5.4, subsets): mean of pub_ho400c and tr_ho400c; 'combined4' = mean of pub_ho400c, tr_ho400c, pub_noind, tr_newterm.
Usage: score_sets.py <run_dir_or_submission>[,<more>...] [--name N] [--csv]   (--csv appends a row to Process/scores_p3.csv)"""
import os, sys, json, glob, csv, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.join(ROOT, "src"))
import score

SETS = ("pub_full", "pub_ho400c", "tr_ho400c", "pub_noind", "tr_newterm", "pub_ind", "pub_hoind", "pub_ho", "tr_ho", "pub_ho400", "tr_ho400", "lrl_ab0",
        "tr_dev_para_a", "tr_dev_para_b", "tr_dev_all", "d3sel")


def load_subs(paths):
    subs = {}
    for p in paths:
        fs = sorted(glob.glob(os.path.join(p, "**", "submission.jsonl"), recursive=True)) if os.path.isdir(p) else [p]
        for f in fs:
            for l in open(f, encoding="utf-8"):
                if l.strip(): r = json.loads(l); subs[r["id"]] = r
    return subs


def refs():
    out, have_pub = {}, False
    for f in ("data/raw/public_gt.jsonl", "data/raw/train/train/answers.jsonl", "tests/splits/d3sel.answers.jsonl"):
        p = os.path.join(ROOT, f)
        if os.path.isfile(p):
            out.update(score.load(p)); have_pub |= "public_gt" in f
    return out, have_pub


def score_sets(subs, R):
    out = {}
    for s in SETS:
        p = os.path.join(ROOT, "tests", "splits", s + ".txt")
        if not os.path.isfile(p): continue
        ids = [l.strip() for l in open(p) if l.strip()]
        have = [i for i in ids if i in subs]
        if len(have) < 0.9 * len(ids): continue                            # a set this run did not cover
        if any((i in R or i.split("~")[0] in R) is False for i in ids): continue      # no reference for this set (public w/o gt)
        tot = 0; c = {"right": 0, "wrong": 0, "abstain": len(ids) - len(have)}
        for i in have:
            a = subs[i]; ref = R[i] if i in R else R[i.split("~")[0]]
            if a.get("type") == "abstain": c["abstain"] += 1; continue
            try: j = score.judge(a, ref)
            except Exception: j = "wrong"
            if j == "correct": c["right"] += 1; tot += 1
            else: c["wrong"] += 1; tot -= 1
        out[s] = {"pa": round(tot / len(ids), 4), "answered": len(have), "n": len(ids), **c}
    if "pub_ho400c" in out and "tr_ho400c" in out:
        out["ho_mean"] = round((out["pub_ho400c"]["pa"] + out["tr_ho400c"]["pa"]) / 2, 4)
        if "pub_noind" in out and "tr_newterm" in out:
            out["combined4"] = round(sum(out[k]["pa"] for k in ("pub_ho400c", "tr_ho400c", "pub_noind", "tr_newterm")) / 4, 4)
    return out


def main():
    paths = sys.argv[1].split(",")
    name = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else os.path.basename(os.path.normpath(paths[0]))
    R, have_pub = refs()
    if not have_pub: print("(no data/raw/public_gt.jsonl: public sets skipped)")
    out = score_sets(load_subs(paths), R)
    for s, v in out.items(): print(f"{s}: {v}")
    if "--csv" in sys.argv:
        f = os.path.join(ROOT, "Process", "scores_p3.csv"); new = not os.path.isfile(f)
        cols = ["run", "time"] + [f"{s}{k}" for s in SETS for k in ("", "_n")] + ["ho_mean", "combined4"]
        row = {"run": name, "time": time.strftime("%Y-%m-%d %H:%M")}
        for s in SETS:
            if s in out: row[s] = out[s]["pa"]; row[s + "_n"] = out[s]["answered"]
        for k in ("ho_mean", "combined4"):
            if k in out: row[k] = out[k]
        if not new:                                                     # one row per run name: a re-score replaces it
            rows = [r for r in csv.DictReader(open(f)) if r["run"] != name]
        else: rows = []
        rows.append(row)
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]


if __name__ == "__main__":
    main()
