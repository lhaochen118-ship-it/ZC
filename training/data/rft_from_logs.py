#!/usr/bin/env python3
"""M2 (docs/12_PLAN2.md §6, STaR / rejection-sampling fine-tuning): SFT records made of the model's OWN correct outputs.
Input = hybrid-flow logs (logs.jsonl) of the model on training-split questions only (training shops not held out, their
paraphrases, D3 teacher questions); a sample is kept when its first turn already gave the answer the reference gives (checked by
execution, src/score.judge), its program is complete and unflagged. The record's prompt is rebuilt exactly as at inference with
the CODE THIS SCRIPT RUNS WITH (src/shopctx.py; run it from the worktree whose model wrote the logs): same hints, same retrieved
examples (checked against the ids the log recorded), same variant; the label is the sample's own text (its short reasoning,
<plan>, <code>), so the fine-tune keeps the model's output distribution and only reinforces its correct choices; a correct
program the model reached only by reasoning first ('think' samples), by a repair turn or in an answer longer than --max-ntok
is written as the direct answer <plan>…</plan><code>…</code> (the data-factory format): get it right at the first, short try.
Hard questions (the flow's vote was not unanimous, wrong or abstained) keep up to --per-hard samples; easy ones one sample, a
seeded share --easy-frac of them. Public / private test questions are never read (refs = training answers + D3 answers only).

Usage: rft_from_logs.py <train_split> <ingest_cache> <canon_dir> <out.jsonl> --logs a/logs.jsonl[,b/logs.jsonl]
       --refs answers.jsonl[,...] [--questions q.jsonl,...] [--mode full] [--per-hard 2] [--easy-frac 0.3] [--max-ntok 480]"""
import os, sys, json, random, argparse, time
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("cache"); ap.add_argument("canon_dir"); ap.add_argument("out")
    ap.add_argument("--logs", required=True); ap.add_argument("--refs", required=True); ap.add_argument("--questions", default="")
    ap.add_argument("--mode", default="full"); ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--per-hard", type=int, default=2); ap.add_argument("--easy-frac", type=float, default=0.3)
    ap.add_argument("--max-ntok", type=int, default=480, help="drop samples longer than this (truncation risk at 512)")
    ap.add_argument("--seed", type=int, default=20261110); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import score, retrieval, grammar
    t0 = time.time()
    refs = {}
    for p in a.refs.split(","): refs.update(score.load(p))
    bank = retrieval.Bank.load(); by_id = {e.get("id"): e for e in bank.ex}
    recs = []
    for p in a.logs.split(","):
        recs += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    rnd = random.Random(a.seed); stats = Counter(); rows = []
    seen = set()
    for r in recs[: a.limit or None]:
        if r["id"] in seen: continue
        seen.add(r["id"])
        if r.get("business_id") in common.TR_HO: stats["held_out_skipped"] += 1; continue
        ref = refs.get(r["id"].split("~")[0]) if r["id"] not in refs else refs[r["id"]]
        if ref is None: stats["no_ref"] += 1; continue
        lg = r.get("samples") if isinstance(r.get("samples"), dict) else {}
        S = lg.get("samples") or []
        good, other, n_ok = [], [], 0
        for s in S:
            if not s.get("ok"): continue
            try: ok = score.judge(s["answer"], ref) == "correct"
            except Exception: ok = False
            n_ok += ok
            t = s.get("turns") or [{}]
            if not ok or not (t[-1].get("code") or "").strip(): continue
            txt = t[0].get("text") or ""
            try: ntok = int(t[0].get("ntok") or 0)
            except (TypeError, ValueError): ntok = 0
            if (len(t) == 1 and not s.get("flags") and s.get("variant") == "direct" and "</code>" in txt
                    and not (a.max_ntok and ntok > a.max_ntok)):
                good.append((ntok, s, txt, "A"))                             # A: the sample's own text as it was written
            else:                                                       # B: a correct program reached by reasoning first,
                code = t[-1]["code"]; plan = t[-1].get("plan")          # by a repair turn or by a long answer: written as
                other.append((len(code), s, grammar.wrap(code, plan or grammar.auto_plan(code)), "B"))   # the direct answer
        if not good and not other: stats["no_correct"] += 1; continue
        try: fin_ok = score.judge(r.get("final") or {"type": "abstain"}, ref) == "correct"
        except Exception: fin_ok = False
        hard = not fin_ok or n_ok < len(S)
        if not hard and rnd.random() >= a.easy_frac: stats["easy_dropped"] += 1; continue
        if not hard and not good: stats["easy_no_direct"] += 1; continue
        take, codes = [], set()
        for _, s, lab, kind in sorted(good, key=lambda x: x[0]) + sorted(other, key=lambda x: x[0]):   # own short text first, distinct programs
            c = "".join((s["turns"][-1].get("code") or "").split())
            if c in codes: continue
            codes.add(c); take.append((s, lab, kind))
            if len(take) >= (a.per_hard if hard else 1): break
        c = common.ctx(a.split, a.cache, a.canon_dir, r["business_id"], a.mode)
        q = {"id": r["id"], "business_id": r["business_id"], "question": r["question"], "as_of_date": r["as_of_date"]}
        h = c.hints(q["question"], q["as_of_date"])
        if json.dumps(h, sort_keys=True, default=str) != json.dumps(lg.get("hints") or {}, sort_keys=True, default=str): stats["hints_differ"] += 1
        exs = bank.retrieve(q["question"], h, k=a.shots, exclude_bids={r["business_id"]}, exclude_ids={q["id"]})
        want = list(lg.get("examples") or [])
        if [e.get("id") for e in exs] != want:
            stats["examples_differ"] += 1
            if all(i in by_id for i in want): exs = [by_id[i] for i in want]  # the logged examples win
        msgs = c.messages(q["question"], q["as_of_date"], exs, h, "direct")
        for s, lab, kind in take:
            rows.append({"id": q["id"], "business_id": q["business_id"], "source": ("RFT_hard" if hard else "RFT_easy") + "_" + kind,
                         "messages": msgs + [{"role": "assistant", "content": lab}],
                         "program": s["turns"][-1].get("code"), "answer": s.get("answer"), "as_of_date": q["as_of_date"],
                         "question": q["question"]})
            stats[("rows_hard_" if hard else "rows_easy_") + kind] += 1
        stats["questions_hard" if hard else "questions_easy"] += 1
    rnd.shuffle(rows)
    common.write_jl(a.out, rows)
    print(f"RFT: {len(rows)} records from {len(seen)} question logs -> {a.out} ({time.time() - t0:.0f}s) {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
