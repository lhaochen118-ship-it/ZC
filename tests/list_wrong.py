#!/usr/bin/env python3
"""Every wrong or abstained question of the ORIGINAL public test (pub_full, 1000 questions) in a run, with what the analyst
needs (docs/13_TASK_AFTER_PLAN1.md §1): id, question, shop, reference answer (data/raw/public_gt.jsonl, scoring only), the
model's final answer, confidence + votes, the hints, and every sample's plan / code / precheck / execution / postcheck.
Held-out questions (pub_ho, tr_ho, ids with '~') are never shown.

The reference answers are read here only to judge and to show the analyst; nothing of this goes to src/ or a prompt.
Usage: list_wrong.py <run_dir>[,<run_dir>...] [--md out.md] [--jsonl out.jsonl] [--set pub_full] [--max-code 1200]
       several runs: one section per run plus the ids wrong in every run (stable errors)."""
import os, sys, json
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import score


def ids_of(name):
    p = os.path.join(ROOT, "tests", "splits", name + ".txt")
    return {l.strip() for l in open(p) if l.strip()} if os.path.isfile(p) else set()


def load_logs(run):
    p = os.path.join(run, "logs.jsonl")
    if not os.path.isfile(p):
        c = [os.path.join(dp, "logs.jsonl") for dp, _, fs in os.walk(run) if "logs.jsonl" in fs]
        if not c: sys.exit(f"no logs.jsonl under {run}")
        p = c[0]
    out = {}
    for l in open(p, encoding="utf-8"):
        r = json.loads(l); out[r["id"]] = r                       # the last line of an id wins (resumed runs)
    return out


def verdict(fin, ref):
    if not fin or fin.get("type") == "abstain": return "abstain"
    try: return score.judge(fin, ref)
    except Exception as e: return f"judge_error:{type(e).__name__}"


def rows_of(run, keep, R, max_code):
    L = load_logs(run); rows = []; n = right = wrong = abst = 0
    for i in sorted(keep):
        r = L.get(i)
        if r is None or i not in R: continue
        n += 1; v = verdict(r.get("final"), R[i])
        if v == "correct": right += 1; continue
        if v == "abstain": abst += 1
        else: wrong += 1
        lg = r.get("samples") if isinstance(r.get("samples"), dict) else {}
        S = lg.get("samples") or []
        samples = []
        for s in S:
            turns = []
            for t in s.get("turns") or []:
                turns.append({"plan": t.get("plan"), "code": (t.get("code") or "")[:max_code], "precheck": t.get("precheck"),
                              "exec": t.get("exec"), "post": t.get("post"), "ntok": t.get("ntok")})
            ans = s.get("answer")
            samples.append({"j": s.get("j"), "variant": s.get("variant"), "temp": s.get("temp"), "ok": s.get("ok"),
                            "answer": ans, "right": (verdict(ans, R[i]) == "correct") if ans else False,
                            "flags": s.get("flags"), "turns": turns})
        rows.append({"id": i, "business_id": r.get("business_id"), "as_of_date": r.get("as_of_date"), "question": r.get("question"),
                     "gt": R[i], "final": r.get("final"), "verdict": v, "conf": r.get("conf"), "votes": r.get("votes"),
                     "stop": lg.get("stop"), "error": lg.get("error"), "hints": lg.get("hints"), "examples": lg.get("examples"),
                     "any_sample_right": any(s["right"] for s in samples), "samples": samples, "elapsed_s": r.get("elapsed_s")})
    pa = (right - wrong) / max(1, n)
    return rows, {"run": run, "n": n, "right": right, "wrong": wrong, "abstain": abst, "pa": round(pa, 4), "acc": round(right / max(1, n), 4)}


def j(x): return json.dumps(x, ensure_ascii=False)


def md_row(r):
    out = [f"### {r['id']} ({r['business_id']}, as_of {r['as_of_date']}) — {r['verdict']}",
           f"- **Question:** {r['question']}", f"- **Reference:** `{j(r['gt'])}`", f"- **Model:** `{j(r['final'])}` (conf {r['conf']}, stop {r['stop']})",
           f"- **Votes:** `{j(r['votes'])}`; a sample was right: {r['any_sample_right']}", f"- **Hints:** `{j(r['hints'])}`"]
    if r.get("error"): out.append(f"- **Error:** `{r['error']}`")
    for s in r["samples"]:
        out.append(f"- sample {s['j']} [{s['variant']}, t={s['temp']}] ok={s['ok']} right={s['right']} answer=`{j(s['answer'])}` flags={s['flags']}")
        for k, t in enumerate(s["turns"]):
            ex = t.get("exec") or {}
            res = ex.get("answer") if ex.get("ok") else ex.get("error")
            out.append(f"  - turn {k}: plan `{t['plan']}`; precheck {t['precheck']}; exec `{j(res)[:300]}`; post {t['post']}")
            if t["code"]: out.append("\n```python\n" + t["code"].strip() + "\n```\n")
    return "\n".join(out)


def main():
    runs = sys.argv[1].split(",")
    arg = lambda k, d="": sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    keep = ids_of(arg("--set", "pub_full")) - ids_of("pub_ho") - ids_of("tr_ho")
    keep = {i for i in keep if "~" not in i}
    R = score.load(os.path.join(ROOT, "data", "raw", "public_gt.jsonl"))
    max_code = int(arg("--max-code", "1200"))
    allrows, stats = [], []
    for run in runs:
        rows, st = rows_of(run, keep, R, max_code); allrows.append(rows); stats.append(st)
        print(f"{run}: n {st['n']} right {st['right']} wrong {st['wrong']} abstain {st['abstain']} acc {st['acc']} PA {st['pa']}")
    stable = set.intersection(*[{r["id"] for r in rows} for rows in allrows]) if allrows else set()
    if len(runs) > 1: print(f"not right in every run: {len(stable)}; in at least one: {len(set().union(*[{r['id'] for r in rows} for rows in allrows]))}")
    if arg("--jsonl"):
        with open(arg("--jsonl"), "w", encoding="utf-8") as f:
            for k, rows in enumerate(allrows):
                for r in rows: f.write(j(dict(r, run=runs[k], stable=r["id"] in stable)) + "\n")
    if arg("--md"):
        with open(arg("--md"), "w", encoding="utf-8") as f:
            f.write("# Wrong / abstained questions of the original public test\n\n")
            f.write("Reference answers from `data/raw/public_gt.jsonl` (scoring and analysis only; never in src/ or a prompt).\n\n")
            for st in stats: f.write(f"- `{st['run']}`: n {st['n']}, right {st['right']}, wrong {st['wrong']}, abstain {st['abstain']}, accuracy {st['acc']}, PA {st['pa']}\n")
            if len(runs) > 1: f.write(f"\nNot right in every run: {len(stable)} (marked *stable*).\n")
            rows = allrows[0]
            kinds = Counter(r["verdict"] for r in rows)
            f.write(f"\n## Run 1: {len(rows)} questions ({dict(kinds)})\n\n")
            for r in rows: f.write(md_row(r) + ("\n- *stable* (not right in every run)\n" if r["id"] in stable and len(runs) > 1 else "\n") + "\n")
            for k in range(1, len(runs)):
                extra = [r for r in allrows[k] if r["id"] not in {x["id"] for x in rows}]
                f.write(f"\n## Run {k + 1}: wrong only here ({len(extra)})\n\n")
                for r in extra: f.write(md_row(r) + "\n\n")


if __name__ == "__main__":
    main()
