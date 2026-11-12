#!/usr/bin/env python3
"""Pure-model training records (user 12/11: the model writes the program from the question, the shop context and retrieved
checked examples only — no rule-read hints). Re-renders existing SFT records (D1 / D2 / D2b ...: same question, same checked
program, same label) in the prompt of the pure flow (src/hybrid.py --pure 1): no # HINTS block, examples retrieved by the question
text alone (retrieval.Bank.retrieve(question, {}): never the same shop, never the question itself or its paraphrases).
--self-hints: the label starts with <hints>{...}</hints> = what the rule hint module (src/hints.py, counts on) reads from THIS
question (clean question, period, entities -> data values, answer type, measure words, counted object, thresholds; not the long
rulebook definitions, already in the system prompt), then <plan>/<code>: the model learns to write its own hints first, so the
pure flow (no rule at inference) keeps what the hints gave (distillation of the hint module; prompting.parse_output ignores
the <hints> block).
Usage: pure_records.py <train_split> <ingest_cache> <canon_dir> <in.jsonl>[,<in.jsonl>...] <out_dir> [--mode full] [--k 3] [--self-hints]
       writes <out_dir>/<name of each input>p.jsonl (d1_full.jsonl -> d1_fullp.jsonl; --self-hints: ...ph.jsonl)"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


SELF_KEYS = ("clean", "period", "periods", "entities", "answer_type", "measure_words", "counted", "thresholds", "missing_data")


def self_hints(h):
    """the <hints> block the model writes first: the rule hints without the rulebook definitions, entity mentions as
    'text -> data value (kind)', the period as its toolkit expression."""
    k = {}
    for key in SELF_KEYS:
        v = (h or {}).get(key)
        if v in (None, [], {}, ""): continue
        if key == "period" and isinstance(v, dict): v = v.get("code")
        if key == "entities": v = [f"{e.get('text')} -> {e.get('data_value')} ({e.get('kind')})" for e in v if e.get("data_value")]
        if v in (None, [], {}, ""): continue
        k[key] = v
    return "<hints>" + json.dumps(k, ensure_ascii=False, separators=(", ", ": "), default=str) + "</hints>\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("cache"); ap.add_argument("canon_dir"); ap.add_argument("inputs"); ap.add_argument("out_dir")
    ap.add_argument("--mode", default="full"); ap.add_argument("--k", type=int, default=3); ap.add_argument("--self-hints", action="store_true")
    a = ap.parse_args()
    import retrieval
    bank = retrieval.Bank.load()
    os.makedirs(a.out_dir, exist_ok=True)
    for f in a.inputs.split(","):
        t0 = time.time(); rows = []
        for r in common.jl(f):
            q = {"id": r["id"], "business_id": r["business_id"], "question": r["question"], "as_of_date": r["as_of_date"]}
            c = common.ctx(a.split, a.cache, a.canon_dir, r["business_id"], a.mode)
            exs = bank.retrieve(q["question"], {}, k=a.k, exclude_bids={r["business_id"]}, exclude_ids={r["id"].split("~")[0]})
            s = common.sample(c, q, r["program"], exs, None, r.get("source", "D?") + ("ph" if a.self_hints else "p"), r.get("answer"))
            if a.self_hints:
                c.hint_opts = {"counts_hints": True}
                s["messages"][-1]["content"] = self_hints(c.hints(q["question"], q["as_of_date"])) + s["messages"][-1]["content"]
            rows.append(s)
        out = os.path.join(a.out_dir, os.path.basename(f).replace(".jsonl", "ph.jsonl" if a.self_hints else "p.jsonl"))
        common.write_jl(out, rows)
        print(f"{f}: {len(rows)} pure records -> {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
