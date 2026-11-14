#!/usr/bin/env python3
"""v2 C: training records of S' (a LoRA on V3A x 0.70) that teach the solver to READ the rulebook layer it is given at inference
(# RULES rows linked to the question, # SHOP FUNCTIONS written by the tool writer) instead of recalling a fixed rulebook.
Training shops only (v3a_mix already leaves the 6 held-out shops out). Every record is the inference prompt itself (shopctx
messages: business [+ shop functions block] + examples + [rules] + question), changed by text edits of a v3a record:

  replay      every v3a record; half of them also carry a block of the shop's model-written functions that the question does not
              need (the solver learns to ignore what does not apply) — target unchanged
  term        a NEW term (TG0: model-invented, its meaning fixed by a checked program) used in a reworded question, its rulebook
              row in # RULES next to a distractor row -> the checked program (right against the organisers' answer)
  term_fn     the same with the tool writer's function for the term in # SHOP FUNCTIONS (teacher, self-consistent) -> the call of
              that function on the question's period, kept only when it returns the checked answer (execution)
  deleted     (off by default, --deleted-share 0) the reworded question with the term's row absent -> raise Unanswerable. Checked
              14/11 on the built records: the model-invented terms are mostly transparent ("doanh thu thuần", "doanh thu trung
              bình đơn"; S already answers tr_newterm 0.85 without the rows), so the label would teach wrong refusals
  contrast    systematic contrast / style pairs (training/contrast_job.py: executed + back-checked) -> their program
Usage: s2_build.py <out_dir> --units <tg1 dirs> --newterm training/data/out/tg0 --contrast <contrast.jsonl> [--mix ...]
       [--deleted-share 0] [--seed 5]"""
import os, re, sys, ast, json, glob, gzip, zlib, random, argparse
from collections import Counter, defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))


def label(code):
    import grammar, spec_from_program as sp
    return sp.spec_block(sp.spec_of(code)) + grammar.wrap(code, grammar.auto_plan(code))


UNANS = '<spec>{"measure": ["Unanswerable"]}</spec>\n<plan>| - | -</plan>\n<code>\nraise Unanswerable("the rulebook does not define this term")\n</code>'


def split_user(u):
    """(business, middle (examples), question header + text) of a solver user turn."""
    qi = u.rfind("\n\n# QUESTION (today = ")
    ei = u.find("\n\n# EXAMPLES")
    bi = ei if 0 < ei < qi else qi
    return u[:bi], u[bi:qi], u[qi:]


def with_parts(u, block=None, rules=None, question=None, as_of=None):
    biz, mid, q = split_user(u)
    if block: biz = biz + "\n\n" + block
    if rules:                                                          # # RULES goes after the examples, before the question
        mid = mid + "\n\n" + rules.strip()
    if question is not None:
        head = q.split("\n", 3)
        q = f"\n\n# QUESTION (today = {as_of})\n{question.strip()}"
    return biz + mid + q


def unit_call(u, code):
    """the call of unit u on the period expression of a checked program (None when u needs more than a period)."""
    import contrast_job
    req = [p for p in re.findall(r"def\s+\w+\s*\(([^)]*)\)", u["code"])[:1] for p in [p]]
    params = [x.strip() for x in (req[0].split(",") if req else []) if x.strip()]
    required = [x for x in params if "=" not in x and not x.startswith("*")]
    if len(required) != 1: return None
    per = [v for k, v in contrast_job.slots(code) if k == "period"]
    if not per: return None
    m = re.search(r'"type":\s*"(\w+)"', code); t = m.group(1) if m else "number"
    return f'ANSWER = {{"type": "{t}", "answer": {u["name"]}({per[0]})}}'


def run_with_unit(split, b, as_of, unit_code, prog):
    import contrast_job
    return contrast_job.run_program(split, b, as_of, unit_code + "\n" + prog)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--units", default=""); ap.add_argument("--newterm", default="")
    ap.add_argument("--contrast", default=""); ap.add_argument("--mix", default=os.path.join(ROOT, "Process", "train_data", "p3", "v3a_mix.jsonl.gz"))
    ap.add_argument("--split", default=os.path.join(ROOT, "data", "raw", "train", "train"))
    ap.add_argument("--deleted-share", type=float, default=0.0); ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--max-units", type=int, default=8)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rnd = random.Random(a.seed)
    import units as unitsmod, lrl, judge, wt_newterm_in
    ho = set(wt_newterm_in.holdout_shops())
    mix = [json.loads(l) for l in gzip.open(a.mix, "rt", encoding="utf-8")]
    mix = [r for r in mix if r["business_id"] not in ho]
    by_id = {}
    for r in mix: by_id.setdefault(r["id"], r)
    U = defaultdict(list)                                              # shop -> model-written functions of NEW rows (as at inference)
    for d in [x for x in a.units.split(",") if x]:
        for f in glob.glob(os.path.join(d, "**", "units_teacher.jsonl"), recursive=True):
            for l in open(f, encoding="utf-8"):
                if not l.strip(): continue
                u = json.loads(l)
                if u.get("src") in ("synth_new", "synth_variant", "newterm") and u["bid"] not in ho: U[u["bid"]].append(u)
    NT = defaultdict(list)
    for f in glob.glob(os.path.join(a.newterm, "**", "cards_newterm*.jsonl"), recursive=True) if a.newterm else []:
        for l in open(f, encoding="utf-8"):
            if l.strip():
                c = json.loads(l)
                if c["business_id"] not in ho: NT[c["business_id"]].append(c)
    recs, st = [], Counter()

    def rec(base, user, target, source, q=None):
        m = base["messages"]
        recs.append({"id": f"{base['id']}~{source}~{len(recs)}", "business_id": base["business_id"], "source": "S2:" + source,
                     "messages": [m[0], {"role": "user", "content": user}, {"role": "assistant", "content": target}]})
        st[source] += 1

    for r in mix:                                                      # replay (half with irrelevant shop functions)
        u = r["messages"][1]["content"]
        us = [x for x in U.get(r["business_id"], []) if x["term"].lower() not in r["question"].lower()]
        blk = unitsmod.block(rnd.sample(us, min(len(us), a.max_units))) if us and rnd.random() < 0.5 else None
        rec(r, with_parts(u, block=blk), r["messages"][2]["content"], "replay_fn" if blk else "replay")
    for b, cards in NT.items():
        for c in cards:
            base = by_id.get(c["id"])
            if base is None: st["term_no_base"] += 1; continue
            row = {"content": f"{c['term']}: {c['definition']}"}
            others = [x for x in cards if x["term"] != c["term"]]
            dis = [{"content": f"{x['term']}: {x['definition']}"} for x in rnd.sample(others, min(1, len(others)))]
            rows = [row] + dis; rnd.shuffle(rows)
            u0 = base["messages"][1]["content"]
            rec(base, with_parts(u0, rules=lrl.render(rows), question=c["question"], as_of=c["as_of_date"]), label(c["program"]), "term")
            fn = next((x for x in U.get(b, []) if x.get("src") == "newterm" and x["term"] == c["term"]), None)
            if fn is not None:
                call = unit_call(fn, c["program"])
                if call:
                    ok, ans = run_with_unit(a.split, b, c["as_of_date"], fn["code"], call)
                    if ok and judge.canon(ans) == judge.canon(c.get("answer")):
                        rest = [x for x in U[b] if x is not fn]
                        blk = unitsmod.block([fn] + rnd.sample(rest, min(len(rest), a.max_units - 1)))
                        rec(base, with_parts(u0, block=blk, rules=lrl.render(rows), question=c["question"], as_of=c["as_of_date"]), label(call), "term_fn")
                    else: st["term_fn_mismatch"] += 1
            if rnd.random() < a.deleted_share * 4:                     # ~deleted-share of all records
                rest = [x for x in U.get(b, []) if x["term"] != c["term"]]
                blk = unitsmod.block(rnd.sample(rest, min(len(rest), a.max_units))) if rest else None
                rec(base, with_parts(u0, block=blk, rules=lrl.render(dis) if dis else None, question=c["question"], as_of=c["as_of_date"]), UNANS, "deleted")
    if a.contrast:
        for l in open(a.contrast, encoding="utf-8"):
            if not l.strip(): continue
            c = json.loads(l); base = by_id.get(c["id"].split("~")[0])
            if base is None: st["contrast_no_base"] += 1; continue
            rec(base, with_parts(base["messages"][1]["content"], question=c["question"], as_of=c["as_of_date"]), label(c["program"]),
                "style" if c["slot"][0] == "style" else "contrast")
    rnd.shuffle(recs)
    with open(os.path.join(a.out, "s2_mix.jsonl"), "w", encoding="utf-8") as w:
        for r in recs: w.write(json.dumps(r, ensure_ascii=False) + "\n")
    chars = sorted(sum(len(m["content"]) for m in r["messages"]) for r in recs) or [0]
    summ = {"records": len(recs), "by_source": dict(st), "chars_p99": chars[int(0.99 * (len(chars) - 1))], "chars_max": chars[-1],
            "est_tokens_M": round(sum(chars) * 0.33 / 1e6, 1)}
    json.dump(summ, open(os.path.join(a.out, "s2_summary.json"), "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
