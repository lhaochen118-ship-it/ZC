#!/usr/bin/env python3
"""New7 offline scoring after the GPU pass (CPU): N7-A / N7-B (certificates), N3-A (counter-example choice), N6-A (DuckDB second
engine), each as a change of the run's vote, scored paired against the run's own vote (same candidates).

  n7a_drop   candidate groups whose certificate fails the algebra (new7_dimensions.validate_certificate) or whose result unit
             differs from the unit the question asks (both written by Qwen) are dropped from the vote (never all: if every group
             fails the vote stands)
  n3a_pick   on disagreement questions, the program chosen by the majority of Qwen's 3 choices wins; 0 (none) -> abstain only when
             >= 2 of 3 say none; no valid choice -> vote stands
  n6a_vote   the answers of Qwen's DuckDB queries (executed on the canonical tables) join the vote as extra samples
  n6a_agree  on disagreement questions, the pandas group that a DuckDB answer equals wins; none equal -> vote stands
Usage: new7_score.py <solver_logs.jsonl> <gen_dir> <canon_dir> --tag _b00s0"""
import os, sys, json, glob, pickle, re, argparse, html
from collections import Counter
sys.path.insert(0, "/home/user/ZC/src")


def jparse(t):
    t = html.unescape(t or ""); t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    m = re.search(r"\{.*\}", t, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None


def vote(smp, drop=None, extra=(), theta=0.5):
    import agent
    votes, rep, n = Counter(), {}, 0
    for s in smp:
        n += 1
        if not s.get("ok") or not isinstance(s.get("answer"), dict): continue
        c = agent.canon_of(s["answer"])
        if drop and c in drop: continue
        votes[c] += 0.5 if s.get("flags") else 1.0; rep.setdefault(c, s["answer"])
    for a in extra:
        n += 1; c = agent.canon_of(a); votes[c] += 1.0; rep.setdefault(c, a)
    if not votes: return {"type": "abstain"}
    top, v = votes.most_common(1)[0]
    return rep[top] if v / max(1, n) >= theta else {"type": "abstain"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("logs"); ap.add_argument("gen"); ap.add_argument("canon"); ap.add_argument("--tag", default="")
    ap.add_argument("--gt", default="/home/user/ZC/data/raw/public_gt.jsonl"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    import agent, score, sqlesc, new7_dimensions as nd
    gen = {}
    for f in glob.glob(os.path.join(a.gen, "**", "gen_*.jsonl"), recursive=True):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); gen[r["id"]] = r["texts"]
    items = {}
    for m in ("n3a",):
        p = os.path.join("/home/user/ZC/training/data/out/p5n7", f"{m}{a.tag}.jsonl")
        if os.path.isfile(p):
            for l in open(p, encoding="utf-8"): it = json.loads(l); items[(m, it["id"])] = it
    rows = [json.loads(l) for l in open(a.logs, encoding="utf-8") if l.strip()]
    gt = score.load(a.gt); C = {}
    pol = {k: {} for k in ("base", "n7a_drop", "n3a_pick", "n6a_vote", "n6a_agree")}; st = Counter()
    for r in rows:
        qid, smp = r["id"], (r.get("samples") or {}).get("samples") or []
        groups = {}
        for s in smp:
            if s.get("ok") and isinstance(s.get("answer"), dict): groups.setdefault(agent.canon_of(s["answer"]), s["answer"])
        pol["base"][qid] = vote(smp)
        # N7-A / N7-B
        bad = set()
        for c in groups:
            t = gen.get(f"n7a{a.tag}|{qid}|{c}") or gen.get(f"{qid}|{c}")
            if not t: continue
            cert = jparse(t[0]); st["n7a_cert"] += 1
            if not isinstance(cert, dict): st["n7a_unparsed"] += 1; continue
            ok = nd.validate_certificate(cert).valid if all(k in cert for k in ("variables", "expression", "result_unit")) else False
            try: same = nd.parse_dimension(str(cert.get("question_unit", ""))) == nd.parse_dimension(str(cert.get("result_unit", "")))
            except Exception: same = False
            if not ok or not same: bad.add(c); st["n7a_flag"] += 1
        pol["n7a_drop"][qid] = vote(smp, drop=bad) if bad and len(bad) < len(groups) else pol["base"][qid]
        # N3-A
        it = items.get(("n3a", qid)); t = gen.get(f"n3a{a.tag}|{qid}") if it else None
        choice = None
        if it and t:
            ch = Counter()
            for x in t:
                j = jparse(x)
                try: ch[int((j or {}).get("choice"))] += 1
                except Exception: pass
            if ch:
                k, v = ch.most_common(1)[0]
                if k == 0 and v >= 2: choice = "none"
                elif 1 <= k <= len(it["order"]): choice = it["order"][k - 1]
            st["n3a_q"] += 1
        pol["n3a_pick"][qid] = ({"type": "abstain"} if choice == "none" else groups.get(choice)) if choice else pol["base"][qid]
        if pol["n3a_pick"][qid] is None: pol["n3a_pick"][qid] = pol["base"][qid]
        # N6-A
        sq = gen.get(f"sql{a.tag}|{qid}") or ([] if not it else None)
        sql_answers = []
        if len(groups) >= 2:
            sq = gen.get(f"n6a{a.tag}|{qid}") or gen.get(f"sql|{qid}") or []
            b = r["business_id"]
            if b not in C: C[b] = pickle.load(open(os.path.join(a.canon, b + ".pkl"), "rb"))["C"]
            for x in sq:
                j = jparse(x)
                if not j or not j.get("sql"): continue
                try:
                    ans = sqlesc.make_sql_answer(C[b])(str(j["sql"]))
                    if isinstance(ans, dict) and ans.get("type") not in (None, "abstain") and ans.get("answer") is not None:
                        agent.canon_of(ans); sql_answers.append(ans); st["n6a_ok"] += 1
                except Exception: st["n6a_err"] += 1
        pol["n6a_vote"][qid] = vote(smp, extra=sql_answers) if sql_answers else pol["base"][qid]
        agree = [c for c in groups if any(agent.canon_of(x) == c for x in sql_answers)]
        pol["n6a_agree"][qid] = groups[agree[0]] if len(agree) == 1 else pol["base"][qid]
    s = {"correct": 1, "abstain": 0, "wrong": -1}; out = {"stats": dict(st)}
    _, vb = score.score(gt, pol["base"])
    for k, p in pol.items():
        rr, va = score.score(gt, p); d = [s[va[q]] - s[vb[q]] for q in gt]; B = sum(x > 0 for x in d); Cn = sum(x < 0 for x in d)
        out[k] = {"PA": rr["PA"], "c/w/a": f"{rr.get('correct', 0)}/{rr.get('wrong', 0)}/{rr.get('abstain', 0)}", "better": B, "worse": Cn,
                  "z": round((sum(d) / len(d)) / ((B + Cn) ** 0.5 / len(d)), 2) if B + Cn else 0.0}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.out: json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
