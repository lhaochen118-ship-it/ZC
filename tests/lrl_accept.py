#!/usr/bin/env python3
"""Acceptance of induced rule cards (plan 3, docs/16 §2.6) — automatic, on TRAINING data only, never by reading cards one by one:

  scan   (CPU) a card with a check expression: over every logged sample of training runs (logs.jsonl of tr_dev_* / d3sel, never
         a held-out shop) whose question the card's triggers match, count the samples the check flags among the WRONG ones and
         among the CORRECT ones. Accept when it flags >= --min-wrong wrong samples, precision >= --min-prec and <= --max-correct
         share of the correct samples it matches ("clean signal", as the checks of round 17 were accepted — now computed, not read).
         A card's own shop does not count (a rule must hold on OTHER shops).
  abset  (CPU) the ids of training questions (other shops than each card's source) that the content cards match -> a split file
         for an A/B run (eval_flow --sets <name>: one run without cards, one with --lrl <candidates>).
  ab     (CPU) per card: correct / wrong with vs without the cards on the questions it matched (other shops); accept when
         fixed - broken >= --min-gain and broken <= --max-broken.
Usage: lrl_accept.py scan <cards.jsonl> <logs.jsonl,...> <out_accepted.jsonl> [--report F]
       lrl_accept.py abset <cards.jsonl> <name> [--sets tr_dev_para_a,d3sel] [--max 600]
       lrl_accept.py ab <cards.jsonl> <logs_without.jsonl> <logs_with.jsonl> <out_accepted.jsonl>"""
import os, sys, json, argparse
from collections import Counter, defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import lrl, score

TR_HO = {"biz_105", "biz_109", "biz_114", "biz_123", "biz_127", "biz_130"}


def refs():
    out = score.load(os.path.join(ROOT, "data", "raw", "train", "train", "answers.jsonl"))
    p = os.path.join(ROOT, "tests", "splits", "d3sel.answers.jsonl")
    if os.path.isfile(p): out.update(score.load(p))
    return out


def src_shop(c): return (c.get("case") or {}).get("business_id")


def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def scan(a):
    cards = [c for c in jl(a.cards) if c.get("check")]
    R = refs(); stats = defaultdict(Counter)
    lib = lrl.Library(cards)
    for f in a.logs.split(","):
        for r in jl(f):
            base = r["id"].split("~")[0]
            if r.get("business_id") in TR_HO or base not in R: continue
            m = lib.match(r["question"], k=len(cards))
            if not m: continue
            for s in ((r.get("samples") or {}).get("samples") or []):
                code = (s.get("turns") or [{}])[-1].get("code")
                if not code or not s.get("ok"): continue
                v = score.judge(s.get("answer"), R[base])
                for c in m:
                    if src_shop(c) == r.get("business_id"): continue
                    fired = bool(lrl.Library.violations([c], r["question"], code))
                    stats[c["id"]][("w" if v != "correct" else "c") + ("_fire" if fired else "")] += 1
    acc, rep = [], []
    for c in cards:
        st = stats[c["id"]]; fw, fc = st["w_fire"], st["c_fire"]; nc = st["c"] + fc
        prec = fw / max(1, fw + fc); share_c = fc / max(1, nc)
        ok = fw >= a.min_wrong and prec >= a.min_prec and share_c <= a.max_correct
        rep.append({"id": c["id"], "content": c["content"][:120], "check": c["check"], "wrong_flagged": fw, "correct_flagged": fc,
                    "correct_matched": nc, "precision": round(prec, 3), "accepted": ok})
        if ok: acc.append(dict(c, stats={"wrong_flagged": fw, "correct_flagged": fc, "precision": round(prec, 3)}))
    with open(a.out, "w", encoding="utf-8") as f:
        for c in acc: f.write(json.dumps(c, ensure_ascii=False) + "\n")
    if a.report: json.dump(rep, open(a.report, "w"), ensure_ascii=False, indent=1)
    print(f"scan: {len(cards)} check cards, {len(acc)} accepted -> {a.out}", flush=True)


def abset(a):
    cards = [c for c in jl(a.cards) if not c.get("disabled")]
    lib = lrl.Library(cards)
    qs = {}
    for f in ("questions.jsonl",):
        for q in jl(os.path.join(ROOT, "data", "raw", "train", "train", f)): qs[q["id"]] = q
    p = os.path.join(ROOT, "tests", "splits", "d3sel.questions.jsonl")
    if os.path.isfile(p):
        for q in jl(p): qs.setdefault(q["id"], q)
    para = {}
    pp = os.environ.get("ZC_PARA_DIR", "")
    if pp and os.path.isfile(os.path.join(pp, "para_train.jsonl")):
        for r in jl(os.path.join(pp, "para_train.jsonl")):
            if r.get("ok"): para[f"{r['id']}~p{r['k']}"] = r["paraphrase"]
    ids = []
    for s in a.sets.split(","):
        ids += [l.strip() for l in open(os.path.join(ROOT, "tests", "splits", s + ".txt")) if l.strip()]
    keep, per = [], Counter()
    for i in dict.fromkeys(ids):
        base = i.split("~")[0]; q = qs.get(base)
        if not q or q["business_id"] in TR_HO: continue
        text = para.get(i) if "~" in i else q["question"]
        if not text: continue
        m = [c for c in lib.match(text, bid=q["business_id"], k=len(cards)) if src_shop(c) != q["business_id"]]
        if m and all(per[c["id"]] < 25 for c in m[:1]):
            keep.append(i); per.update(c["id"] for c in m[:2])
        if len(keep) >= a.max: break
    out = os.path.join(ROOT, "tests", "splits", a.name + ".txt")
    open(out, "w").write("\n".join(keep) + "\n")
    print(f"abset: {len(keep)} questions matched by {len(per)} cards -> {out}", flush=True)


def ab(a):
    cards = {c["id"]: c for c in jl(a.cards)}
    R = refs()
    def verdicts(p):
        out = {}
        for r in jl(p):
            base = r["id"].split("~")[0]
            if base in R: out[r["id"]] = (score.judge(r.get("final"), R[base]), r)
        return out
    A, B = verdicts(a.without), verdicts(a.with_)
    st = defaultdict(Counter)
    for i, (vb, rb) in B.items():
        if i not in A: continue
        va = A[i][0]
        for cid in (rb.get("samples") or {}).get("rules") or rb.get("rules") or []:
            c = cards.get(cid)
            if not c or src_shop(c) == rb.get("business_id"): continue
            st[cid]["n"] += 1
            if va != "correct" and vb == "correct": st[cid]["fixed"] += 1
            if va == "correct" and vb != "correct": st[cid]["broken"] += 1
    acc = []
    for cid, s in st.items():
        if s["fixed"] - s["broken"] >= a.min_gain and s["broken"] <= a.max_broken:
            acc.append(dict(cards[cid], stats={"ab_n": s["n"], "fixed": s["fixed"], "broken": s["broken"]}))
    with open(a.out, "w", encoding="utf-8") as f:
        for c in acc: f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tot = Counter(); [tot.update(s) for s in st.values()]
    print(f"ab: {len(st)} cards matched, {len(acc)} accepted (fixed {tot['fixed']}, broken {tot['broken']}) -> {a.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan"); p.add_argument("cards"); p.add_argument("logs"); p.add_argument("out"); p.add_argument("--report", default="")
    p.add_argument("--min-wrong", type=int, default=3); p.add_argument("--min-prec", type=float, default=0.85); p.add_argument("--max-correct", type=float, default=0.02)
    p = sub.add_parser("abset"); p.add_argument("cards"); p.add_argument("name"); p.add_argument("--sets", default="tr_dev_para_a,d3sel"); p.add_argument("--max", type=int, default=600)
    p = sub.add_parser("ab"); p.add_argument("cards"); p.add_argument("without"); p.add_argument("with_"); p.add_argument("out")
    p.add_argument("--min-gain", type=int, default=1); p.add_argument("--max-broken", type=int, default=0)
    a = ap.parse_args()
    {"scan": scan, "abset": abset, "ab": ab}[a.cmd](a)


if __name__ == "__main__":
    main()
