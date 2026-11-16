#!/usr/bin/env python3
import os, sys, json, random, argparse, time
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def ref_of(r, refs):
    return refs[r["id"]] if r["id"] in refs else refs.get(r["id"].split("~")[0])


def correct_samples(S, ref, max_ntok):
    import score, grammar
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
                and not (max_ntok and ntok > max_ntok)):
            good.append((ntok, s, txt, "A"))
        else:
            code = t[-1]["code"]; plan = t[-1].get("plan")
            other.append((len(code), s, grammar.wrap(code, plan or grammar.auto_plan(code)), "B"))
    return good, other, n_ok


def final_ok(r, ref):
    import score
    try: return score.judge(r.get("final") or {"type": "abstain"}, ref) == "correct"
    except Exception: return False


def prompt_of(r, a, bank, by_id, stats):
    lg = r.get("samples") if isinstance(r.get("samples"), dict) else {}
    c = common.ctx(a.split, a.cache, a.canon_dir, r["business_id"], a.mode)
    h = {} if a.pure else c.hints(r["question"], r["as_of_date"])
    if json.dumps(h, sort_keys=True, default=str) != json.dumps(lg.get("hints") or {}, sort_keys=True, default=str): stats["hints_differ"] += 1
    exs = bank.retrieve(r["question"], h, k=a.shots, exclude_bids={r["business_id"]}, exclude_ids={r["id"]})
    want = list(lg.get("examples") or [])
    if [e.get("id") for e in exs] != want:
        stats["examples_differ"] += 1
        if all(i in by_id for i in want): exs = [by_id[i] for i in want]
    return c.messages(r["question"], r["as_of_date"], exs, None if a.pure else h, "direct")


def load_logs(paths):
    recs, seen = [], set()
    for p in paths.split(","):
        for l in open(p, encoding="utf-8"):
            if not l.strip(): continue
            r = json.loads(l)
            if r["id"] not in seen: seen.add(r["id"]); recs.append(r)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("cache"); ap.add_argument("canon_dir"); ap.add_argument("out")
    ap.add_argument("--logs", required=True); ap.add_argument("--refs", required=True)
    ap.add_argument("--mode", default="full"); ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--per-hard", type=int, default=2); ap.add_argument("--easy-frac", type=float, default=0.3)
    ap.add_argument("--max-ntok", type=int, default=480, help="drop samples longer than this (truncation risk at 512)")
    ap.add_argument("--seed", type=int, default=20261110); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pure", action="store_true", help="logs of the pure flow (src/hybrid.py --pure 1): prompts without hints, retrieval by the question text")
    a = ap.parse_args()
    import score, retrieval
    t0 = time.time()
    refs = {}
    for p in a.refs.split(","): refs.update(score.load(p))
    bank = retrieval.Bank.load(); by_id = {e.get("id"): e for e in bank.ex}
    recs = load_logs(a.logs)[: a.limit or None]
    rnd = random.Random(a.seed); stats = Counter(); rows = []
    for r in recs:
        if r.get("business_id") in common.TR_HO: stats["held_out_skipped"] += 1; continue
        ref = ref_of(r, refs)
        if ref is None: stats["no_ref"] += 1; continue
        lg = r.get("samples") if isinstance(r.get("samples"), dict) else {}
        S = lg.get("samples") or []
        good, other, n_ok = correct_samples(S, ref, a.max_ntok)
        if not good and not other: stats["no_correct"] += 1; continue
        hard = not final_ok(r, ref) or n_ok < len(S)
        if not hard and rnd.random() >= a.easy_frac: stats["easy_dropped"] += 1; continue
        if not hard and not good: stats["easy_no_direct"] += 1; continue
        take, codes = [], set()
        for _, s, lab, kind in sorted(good, key=lambda x: x[0]) + sorted(other, key=lambda x: x[0]):
            c = "".join((s["turns"][-1].get("code") or "").split())
            if c in codes: continue
            codes.add(c); take.append((s, lab, kind))
            if len(take) >= (a.per_hard if hard else 1): break
        msgs = prompt_of(r, a, bank, by_id, stats)
        for s, lab, kind in take:
            rows.append({"id": r["id"], "business_id": r["business_id"], "source": ("RFT_hard" if hard else "RFT_easy") + "_" + kind,
                         "messages": msgs + [{"role": "assistant", "content": lab}],
                         "program": s["turns"][-1].get("code"), "answer": s.get("answer"), "as_of_date": r["as_of_date"],
                         "question": r["question"]})
            stats[("rows_hard_" if hard else "rows_easy_") + kind] += 1
        stats["questions_hard" if hard else "questions_easy"] += 1
    rnd.shuffle(rows)
    common.write_jl(a.out, rows)
    print(f"RFT: {len(rows)} records from {len(recs)} question logs -> {a.out} ({time.time() - t0:.0f}s) {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
