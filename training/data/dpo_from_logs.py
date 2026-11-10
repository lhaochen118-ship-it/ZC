#!/usr/bin/env python3
"""M4 (docs/12_PLAN2.md §6; ExCoT): DPO pairs from the model's own samples on training-split questions only (training shops
not held out, their paraphrases, D3 teacher questions; refs = training / D3 answers, judged by execution).
  chosen   = a correct answer: the model's own short correct text (A), else a correct program it reached only by reasoning
             first / a repair turn / a long answer, written as the direct answer (B), else - for a question that has its own
             checked program (a training question: src/refs/answer_examples.jsonl; a D3 question: d3_programs.jsonl; a
             paraphrase: its original's program when the paraphrase keeps the original's period, as D2 does) - that program
             as the direct answer (C);
  rejected = the model's own direct first-turn text that ran and gave a WRONG answer (most voted wrong answers first), or
             with --truncated a text cut at the token cap without a program (teaches the short format).
Prompts rebuilt exactly as at inference (rft_from_logs.prompt_of: same code, hints, examples). Up to --per-q pairs per
question (distinct rejected answers). Public / private test questions are never read.
Usage: dpo_from_logs.py <split> <ingest_cache> <canon_dir> <out.jsonl> --logs a/logs.jsonl[,...] --refs answers.jsonl[,...]
       [--d3-programs training/data/out/d3/d3_programs.jsonl] [--per-q 2] [--truncated 1] [--max-ntok 480]"""
import os, sys, json, random, argparse, time
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from rft_from_logs import ref_of, correct_samples, final_ok, prompt_of, load_logs


def same_period(o, r):
    """D2's rule (training/data/d2_paraphrase.py): a paraphrase whose period differs from its original's is another question."""
    if not o: return False
    import qparse
    try:
        p0, _ = qparse.parse_period(o["question"], o["as_of_date"]); p1, _ = qparse.parse_period(r["question"], o["as_of_date"])
    except Exception: return False
    return (p0 is None) == (p1 is None) and (p0 is None or (p0.start, p0.end) == (p1.start, p1.end))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("cache"); ap.add_argument("canon_dir"); ap.add_argument("out")
    ap.add_argument("--logs", required=True); ap.add_argument("--refs", required=True)
    ap.add_argument("--d3-programs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "d3", "d3_programs.jsonl"))
    ap.add_argument("--mode", default="full"); ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--per-q", type=int, default=2); ap.add_argument("--truncated", type=int, default=1)
    ap.add_argument("--max-ntok", type=int, default=480); ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import score, retrieval, grammar, agent
    t0 = time.time()
    refs = {}
    for p in a.refs.split(","): refs.update(score.load(p))
    progs = {r["id"]: r["code"] for r in common.jl(os.path.join(common.SRC, "refs", "answer_examples.jsonl")) if r.get("code")}
    if os.path.isfile(a.d3_programs):
        progs.update({r["id"]: r["code"] for r in common.jl(a.d3_programs) if r.get("code")})
    orig = {r["id"]: r for r in common.jl(os.path.join(common.SRC, "refs", "answer_examples.jsonl"))}
    bank = retrieval.Bank.load(); by_id = {e.get("id"): e for e in bank.ex}
    recs = load_logs(a.logs)[: a.limit or None]
    rnd = random.Random(a.seed); stats = Counter(); rows = []
    for r in recs:
        if r.get("business_id") in common.TR_HO: stats["held_out_skipped"] += 1; continue
        ref = ref_of(r, refs)
        if ref is None: stats["no_ref"] += 1; continue
        lg = r.get("samples") if isinstance(r.get("samples"), dict) else {}
        S = lg.get("samples") or []
        wrong, trunc, votes = {}, [], Counter()
        for s in S:
            t = s.get("turns") or [{}]; txt = t[0].get("text") or ""
            if s.get("variant") != "direct" or not txt: continue
            if s.get("ok") and len(t) == 1:
                try: ok = score.judge(s["answer"], ref) == "correct"
                except Exception: ok = False
                if not ok and "</code>" in txt:
                    k = agent.canon_of(s["answer"]); votes[k] += 1; wrong.setdefault(k, txt)
            elif not s.get("ok") and "</code>" not in txt and a.truncated:
                trunc.append(txt)                                       # cut at the cap before any program
        rejected = [wrong[k] for k, _ in votes.most_common()] + trunc[:1]
        if not rejected: stats["no_rejected"] += 1; continue
        good, other, n_ok = correct_samples(S, ref, a.max_ntok)
        if good: chosen, kind = min(good, key=lambda x: x[0])[2], "A"
        elif other: chosen, kind = min(other, key=lambda x: x[0])[2], "B"
        else:
            code = progs.get(r["id"]) or progs.get(r["id"].rsplit("_t", 1)[0] if r["id"].startswith("d3_") else "")
            if not code and "~p" in r["id"] and same_period(orig.get(r["id"].split("~")[0]), r): code = progs.get(r["id"].split("~")[0])
            if not code: stats["no_chosen"] += 1; continue
            chosen, kind = grammar.wrap(code, grammar.auto_plan(code)), "C"
        msgs = prompt_of(r, a, bank, by_id, stats)
        for rej in rejected[: a.per_q]:
            rows.append({"id": r["id"], "business_id": r["business_id"], "source": f"DPO_{kind}" + ("_trunc" if rej in trunc else ""),
                         "messages": msgs, "chosen": chosen, "rejected": rej, "final_ok": final_ok(r, ref)})
            stats[f"pairs_{kind}" + ("_trunc" if rej in trunc else "")] += 1
        stats["questions"] += 1
    rnd.shuffle(rows)
    common.write_jl(a.out, rows)
    print(f"DPO: {len(rows)} pairs from {len(recs)} question logs -> {a.out} ({time.time() - t0:.0f}s) {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
