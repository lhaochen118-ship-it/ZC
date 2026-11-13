#!/usr/bin/env python3
"""DR + tr_newterm (plan 3, docs/16 §3, §5.1) from src/lrl_gen.py --mode newterm outputs (a checked program, a NEW business term
for what it computes, the rulebook row defining it, the question rewritten with the term):
  dev shops  -> pure training records whose shop rulebook carries the new row (it shows in the prompt's 'Rulebook of this test'
               block, as a new industry's rows of the private test would), label = <spec> + the program: the model learns to READ
               new rulebook definitions instead of knowing every term;
  held-out shops -> the measurement set tests/splits/tr_newterm{.txt,.questions.jsonl,.rulebook_rows.jsonl} (<= --per-shop terms
               per shop: the prompt shows at most 12 new rows), scored against the original question's answer.
Usage: dr_newterm.py <train_split> <ingest_cache> <canon_dir> <cards_newterm*.jsonl,...> <out_train.jsonl> [--per-shop 10] [--k 3]"""
import os, sys, json, time, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from spec_from_program import spec_of, spec_block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("cache"); ap.add_argument("canon_dir"); ap.add_argument("inputs"); ap.add_argument("out")
    ap.add_argument("--per-shop", type=int, default=10); ap.add_argument("--k", type=int, default=3); ap.add_argument("--mode", default="full")
    ap.add_argument("--seed", type=int, default=20261113)
    a = ap.parse_args(); t0 = time.time(); rng = random.Random(a.seed)
    import retrieval, shopctx
    bank = retrieval.Bank.load()
    items = []
    for f in a.inputs.split(","):
        if os.path.isfile(f): items += common.jl(f)
    rb0 = open(os.path.join(a.split, "rulebook.md"), encoding="utf-8").read()
    def with_rows(rows):
        return rb0.rstrip("\n") + "\n\n## Thuật ngữ riêng của cửa hàng\n| Thuật ngữ | Định nghĩa |\n|---|---|\n" + "\n".join(rows) + "\n"
    train, ho = [], {}
    for it in items:
        (ho.setdefault(it["business_id"], []) if it["business_id"] in common.TR_HO else train).append(it)
    rows = []
    import copy, prompting
    for it in train:
        base_c = common.ctx(a.split, a.cache, a.canon_dir, it["business_id"], a.mode)   # cached per shop; only the business block
        c = copy.copy(base_c); rb = with_rows([it["row"]]); c.rulebook = rb             # depends on the rulebook's new rows
        rc = 0 if a.mode == "compact" else 1500
        c.business = prompting.business_block(c.card, rule_changes=shopctx.rule_changes(rb), industry_doc=c.industry_doc,
                                              readme=(c.biz.readme or "")[:rc] if rc else "")
        q = {"id": it["id"] + "~n0", "business_id": it["business_id"], "question": it["question"], "as_of_date": it["as_of_date"]}
        exs = bank.retrieve(it["question"], {}, k=a.k, exclude_bids={it["business_id"]}, exclude_ids={it["id"]})
        s = common.sample(c, q, it["program"], exs, None, "DRph", it.get("answer"))
        sp = spec_of(it["program"])
        if sp: s["messages"][-1]["content"] = spec_block(sp) + s["messages"][-1]["content"]
        s["new_term"] = it["term"]; rows.append(s)
    common.write_jl(a.out, rows)
    sd = os.path.join(common.ROOT, "tests", "splits"); ids, qs, rr = [], [], []
    for b, lst in sorted(ho.items()):
        rng.shuffle(lst); seen = set()
        for it in lst:
            if it["term"].lower() in seen: continue
            seen.add(it["term"].lower())
            ids.append(it["id"] + "~n0"); rr.append({"business_id": b, "row": it["row"]})
            qs.append({"id": it["id"] + "~n0", "business_id": b, "question": it["question"], "as_of_date": it["as_of_date"]})
            if len(seen) >= a.per_shop: break
    if ids:
        open(os.path.join(sd, "tr_newterm.txt"), "w").write("\n".join(ids) + "\n")
        for name, data in (("tr_newterm.questions.jsonl", qs), ("tr_newterm.rulebook_rows.jsonl", rr)):
            with open(os.path.join(sd, name), "w", encoding="utf-8") as f:
                for x in data: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"DR: {len(rows)} training records -> {a.out}; tr_newterm: {len(ids)} questions of {len(ho)} held-out shops ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
