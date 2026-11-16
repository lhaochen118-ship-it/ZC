#!/usr/bin/env python3
import os, sys, json, glob, gzip, random, hashlib, argparse
from collections import Counter, defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import judge, score, grammar


def is_val(base_id):
    return int(hashlib.md5(base_id.encode()).hexdigest(), 16) % 10 == 0


def prefixes(mix):
    out = {}
    op = gzip.open if mix.endswith(".gz") else open
    with op(mix, "rt", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l); b = r.get("business_id")
            if not b or b in out: continue
            m = r["messages"]; u = m[1]["content"]
            cuts = [c for c in (u.find(k) for k in ("\n\n# EXAMPLES", "\n\n# RULES", "\n\n# HINTS", "\n\n# QUESTION")) if c > 0]
            if not cuts or not u.startswith("# BUSINESS"): continue
            out[b] = (m[0]["content"], u[:min(cuts)])
    return out


def programs(mix):
    out = {}
    op = gzip.open if mix.endswith(".gz") else open
    with op(mix, "rt", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            if r.get("program") and r.get("id"): out.setdefault(r["id"], r["program"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="c1a,c1b,rq5_tr_dev_para_a,rq5_tr_dev_para_b,v2v4s1a,v2v4s1b,v2v4s2a,v2v4s2b,rq5_tr_dev_all")
    ap.add_argument("--runs-dir", default=os.path.join(ROOT, "Process", "runs"))
    ap.add_argument("--mix", default=os.path.join(ROOT, "Process", "train_data", "p3", "v3a_mix.jsonl.gz"))
    ap.add_argument("--answers", default=os.path.join(ROOT, "data", "raw", "train", "train", "answers.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "training", "data", "out", "wj"))
    ap.add_argument("--n-right", type=int, default=1300, help="list: all-right questions sampled (the rest of the mix is every "
                    "mixed / all-wrong question)"); ap.add_argument("--cap", type=int, default=3, help="records per question id and kind")
    ap.add_argument("--max-chars", type=int, default=13000); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-run", default="c1a")
    ap.add_argument("--aug", type=int, default=0, help="1: also the right candidates dropped from mixed / unanswerable questions "
                    "(the judge sees only wrong programs: label 0, or U when the reference is unanswerable) - teaches rejection")
    ap.add_argument("--tag", default="")
    ap.add_argument("--rows", default="", help="p_rows.jsonl: the question's rulebook rows go into the judge prompt (as W2 links at inference)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rnd = random.Random(a.seed)
    A = score.load(a.answers); P = prefixes(a.mix); D1 = programs(a.mix)
    ROWS = {}
    if a.rows:
        for l in open(a.rows, encoding="utf-8"):
            if l.strip(): x = json.loads(l); ROWS[x["id"]] = x.get("cards") or []
        print(f"rulebook rows of {len(ROWS)} questions ({sum(1 for v in ROWS.values() if v)} with rows)", flush=True)
    print(f"answers {len(A)}, shop prefixes {len(P)}, checked programs {len(D1)}", flush=True)
    rows = []
    good = {}
    for run in [x for x in a.runs.split(",") if x]:
        fs = sorted(glob.glob(os.path.join(a.runs_dir, run, "**", "logs.jsonl.gz"), recursive=True))
        if not fs: print(f"  {run}: no logs", flush=True); continue
        L = judge.load_logs(fs[0]); k = 0
        for i, r in L.items():
            ref = A.get(i.split("~")[0])
            if ref is None or r.get("business_id") not in P: continue
            cands, n = judge.candidates(r)
            if not cands: continue
            ok = [score.judge(c["answer"], ref) == "correct" for c in cands]
            rows.append((run, r, cands, n, ok)); k += 1
            for c, o in zip(cands, ok):
                if o and c["answer"].get("type") != "unanswerable": good.setdefault(i, c["code"])
        print(f"  {run}: {k} questions with candidates", flush=True)
    answered_right = {r["id"] for _, r, _, _, ok in rows if any(ok)}

    def plan_of(i, cands, ok, ref):
        if ref["type"] == "unanswerable": return "Unanswerable | - | -"
        code = next((c["code"] for c, o in zip(cands, ok) if o), None) or good.get(i) or D1.get(i) or D1.get(i.split("~")[0])
        return grammar.auto_plan(code) if code else None

    def rec(kind, run, r, cands, n, plan, choice, mode):
        sysm, biz = P[r["business_id"]]
        u = biz + "\n\n" + judge.question_block(r["question"], r["as_of_date"], cands, mode, n, ROWS.get(r["id"]) if ROWS else None)
        if len(sysm) + len(u) > a.max_chars: return None
        return {"id": r["id"], "business_id": r["business_id"], "source": f"WJ_{mode}_{kind}", "run": run, "choice": choice,
                "messages": [{"role": "system", "content": sysm}, {"role": "user", "content": u},
                             {"role": "assistant", "content": judge.label_text(plan, choice)}]}

    L_mixed, L_wrong, L_right, PT_pos, PT_neg, L_aug = [], [], [], [], [], []
    seen = Counter(); val_rows = []; st = Counter()
    order = list(range(len(rows))); rnd.shuffle(order)
    for idx in order:
        run, r, cands, n, ok = rows[idx]; i = r["id"]; base = i.split("~")[0]; ref = A[base]
        if is_val(base):
            if run == a.val_run: val_rows.append(r)
            st["val"] += 1; continue
        plan = plan_of(i, cands, ok, ref)
        if plan is None: st["no_plan"] += 1; continue
        unans = ref["type"] == "unanswerable"
        kind = "right" if all(ok) else ("mixed" if any(ok) else "wrong")
        if kind == "wrong" and "~" in i and i not in answered_right: st["wrong_dropped(noise guard)"] += 1; continue
        st[kind] += 1
        if seen[(i, kind, "l")] < a.cap:
            cs, okk = list(cands), list(ok)
            if len(cs) > 1 and rnd.random() < 0.5:
                z = list(zip(cs, okk)); rnd.shuffle(z); cs, okk = map(list, zip(*z))
            choice = (okk.index(True) + 1) if any(okk) else ("U" if unans else 0)
            x = rec(kind, run, r, cs, n, plan, choice, "list")
            if x is not None:
                seen[(i, kind, "l")] += 1
                (L_right if kind == "right" else L_mixed if kind == "mixed" else L_wrong).append(x)
        if a.aug and any(ok) and not all(ok) and seen[(i, "aug")] < 1:
            wc = [c for c, o in zip(cands, ok) if not o]
            x = rec("aug", run, r, wc, n, plan, "U" if unans else 0, "list")
            if x is not None: seen[(i, "aug")] += 1; L_aug.append(x)
        for c, o in zip(cands, ok):
            key = (i, "p", o, c["key"])
            if seen[key] or seen[(i, "p", o)] >= a.cap: continue
            x = rec("pos" if o else "neg", run, r, [c], n, plan, 1 if o else ("U" if unans else 0), "point")
            if x is None: continue
            seen[key] += 1; seen[(i, "p", o)] += 1
            (PT_pos if o else PT_neg).append(x)
    if a.aug:
        pool = {}
        for run, r, cands, n, ok in rows:
            if is_val(r["id"].split("~")[0]): continue
            for c, o in zip(cands, ok):
                if o and c["answer"].get("type") in ("string", "number", "list"): pool.setdefault(r["business_id"], []).append(c)
        done = Counter()
        for run, r, cands, n, ok in rows:
            i = r["id"]; base = i.split("~")[0]
            if is_val(base) or A[base]["type"] != "unanswerable" or done[i] or not pool.get(r["business_id"]): continue
            want = (A[base].get("answer_type") or "")
            src = pool[r["business_id"]]
            k = 1 if rnd.random() < 0.6 else 2
            cs = [dict(x, n=max(1, n - j)) for j, x in enumerate(rnd.sample(src, min(k, len(src))))]
            x = rec("augU", run, r, cs, n, "Unanswerable | - | -", "U", "list")
            if x is not None: done[i] += 1; L_aug.append(x)
        print(f"augmentation: {sum(1 for x in L_aug if x['choice'] == 'U')} unanswerable records with borrowed programs", flush=True)
    rnd.shuffle(L_right); L_right_s = L_right[:a.n_right]
    lst = L_mixed + L_wrong + L_right_s + L_aug; rnd.shuffle(lst)
    rnd.shuffle(PT_pos); pt = PT_neg + PT_pos[:max(len(PT_neg), 600)]; rnd.shuffle(pt)
    mixd = lst[:] + pt[:len(pt) // 2]; rnd.shuffle(mixd)

    def dump(name, recs):
        p = os.path.join(a.out, name)
        with open(p, "w", encoding="utf-8") as f:
            for x in recs: f.write(json.dumps(x, ensure_ascii=False) + "\n")
        ch = Counter(str(x["choice"]) for x in recs); src = Counter(x["source"] for x in recs)
        chars = sum(len(m["content"]) for x in recs for m in x["messages"])
        print(f"{name}: {len(recs)} records, ~{chars / 3.1 / 1e6:.1f}M tokens (chars / 3.1); choices {dict(ch.most_common(8))}; {dict(src)}", flush=True)
    dump(f"wj_list{a.tag}.jsonl", lst); dump(f"wj_point{a.tag}.jsonl", pt); dump(f"wj_both{a.tag}.jsonl", mixd)
    with gzip.open(os.path.join(a.out, "tr_val.logs.jsonl.gz"), "wt", encoding="utf-8") as f:
        for r in val_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"stats {dict(st)}; list: mixed {len(L_mixed)}, wrong {len(L_wrong)}, right {len(L_right_s)} of {len(L_right)}; "
          f"point: pos {len(PT_pos)}, neg {len(PT_neg)}; validation rows ({a.val_run}) {len(val_rows)}", flush=True)


if __name__ == "__main__":
    main()
