#!/usr/bin/env python3
"""Paraphrase equivalence judge (plan 3, docs/16 §5.1): a held-out paraphrase is scored against its ORIGINAL question's
reference answer, so a paraphrase that changed the period, the quantity or a filter is a wrong label (docs/15 §4.3). The
solution's own model reads each (original, paraphrase) pair with two independent prompts; a pair is kept only when both say
it asks exactly the same thing. Nobody reads the held-out questions: the kept ids become <set>_clean.txt.
GPU job: para_judge.py <split> <out> --model M --sets pub_ho,tr_ho [--para DIR] [--quant fp8] [--fake-responder f.py]
writes <out>/judge.jsonl and <out>/<set>_clean.txt"""
import os, re, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

P1 = ("Hai câu hỏi dưới đây của chủ cửa hàng có hỏi CÙNG MỘT điều không: cùng khoảng thời gian, cùng đại lượng cần tính, cùng "
      "điều kiện lọc (mặt hàng, khách, kênh, cách trả…), cùng kiểu đáp án? Chỉ trả lời CÓ hoặc KHÔNG.\nCâu 1 (hôm nay = {d}): {a}\nCâu 2 (hôm nay = {d}): {b}")
P2 = ("Một câu trả lời đúng cho câu A có chắc chắn cũng là câu trả lời đúng cho câu B không (cùng ngày hỏi {d})? Xét kỹ mốc thời gian, "
      "con số, đơn vị, tên hàng, cách tính. Chỉ trả lời YES hoặc NO.\nA: {a}\nB: {b}")


def yes(t):
    t = (t or "").split("</think>")[-1].strip().lower()
    return t.startswith(("có", "co", "yes"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out"); ap.add_argument("--model", default=""); ap.add_argument("--sets", default="pub_ho,tr_ho")
    ap.add_argument("--para", default=""); ap.add_argument("--quant", default=""); ap.add_argument("--fake-responder", default="")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import eval_flow
    Q = {}
    for name in ("public_test", "train"):
        sp = eval_flow.find_split(name, a.split)
        if sp:
            for l in open(os.path.join(sp, "questions.jsonl"), encoding="utf-8"):
                if l.strip(): q = json.loads(l); Q[q["id"]] = q
    para = {}
    for f in ("para_pub.jsonl", "para_train.jsonl"):
        p = os.path.join(a.para, f) if a.para else next(iter(sorted(glob.glob("/kaggle/input/**/" + f, recursive=True))), "")
        if p and os.path.isfile(p):
            for r in map(json.loads, open(p, encoding="utf-8")):
                if r.get("ok"): para[f"{r['id']}~p{r['k']}"] = r["paraphrase"]
    items = []
    for s in a.sets.split(","):
        for i in eval_flow.split_ids(s):
            if "~p" not in i or i not in para or i.split("~")[0] not in Q: continue
            q = Q[i.split("~")[0]]; items.append((s, i, q["question"], para[i], q["as_of_date"]))
    if os.environ.get("ZC_SMOKE") == "1": items = items[:20]
    convs = []
    for _, _, x, y, d in items:
        convs.append([{"role": "user", "content": P1.format(a=x, b=y, d=d)}]); convs.append([{"role": "user", "content": P2.format(a=x, b=y, d=d)}])
    if a.fake_responder:
        g = {}; exec(compile(open(a.fake_responder, encoding="utf-8").read(), a.fake_responder, "exec"), g)
        texts = [g["responder"](c) for c in convs]
    else:
        sys.argv = [sys.argv[0]]; import agent
        eng = agent.Engine(argparse.Namespace(model=a.model, tp=1, max_len=4096, think=False, eager=False, lora="", quant=a.quant,
                                              gpu_util=0.9, max_num_seqs=0, max_batched_tokens=0, spec_mtp=0, prefix_budget=0))
        texts, _ = eng.gen(convs, [0.0] * len(convs), [7 + i for i in range(len(convs))], 8, think=False)
    keep = {}
    with open(os.path.join(a.out, "judge.jsonl"), "w", encoding="utf-8") as f:
        for k, (s, i, x, y, d) in enumerate(items):
            v1, v2 = yes(texts[2 * k]), yes(texts[2 * k + 1])
            f.write(json.dumps({"set": s, "id": i, "same_1": v1, "same_2": v2}, ensure_ascii=False) + "\n")
            keep.setdefault(s, [])
            if v1 and v2: keep[s].append(i)
    summ = {}
    for s in a.sets.split(","):
        orig = [i for i in eval_flow.split_ids(s) if "~p" not in i]          # original questions of the set stay (their label is exact)
        ids = orig + keep.get(s, [])
        open(os.path.join(a.out, f"{s}_clean.txt"), "w").write("\n".join(ids) + "\n")
        summ[s] = {"paraphrases": sum(1 for it in items if it[0] == s), "kept": len(keep.get(s, [])), "originals": len(orig)}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("PARA_JUDGE", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
