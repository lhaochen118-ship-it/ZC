#!/usr/bin/env python3
import os, re, sys, json, argparse, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qnorm

SYSTEM = ("Bạn là công cụ thêm dấu tiếng Việt. Nhận một câu hỏi tiếng Việt viết thiếu dấu hoặc không dấu; trả lại ĐÚNG câu đó "
          "với dấu tiếng Việt chuẩn theo ngữ cảnh. Không thêm, bớt, đổi thứ tự hay thay từ nào; giữ nguyên số, ký hiệu, mã, tên "
          "riêng, từ tiếng Anh và các từ đã có dấu. Chỉ in câu kết quả, không giải thích.")
TOK = re.compile(r"[^\W_]+", re.U)


def toks(s): return [qnorm.strip_accents(t).lower() for t in TOK.findall(unicodedata.normalize("NFC", s))]


def check(src, out):
    if not out or "\n" in out.strip(): return False
    a, b = TOK.findall(unicodedata.normalize("NFC", src)), TOK.findall(unicodedata.normalize("NFC", out.strip()))
    if len(a) != len(b) or toks(src) != toks(out): return False
    return all(x.lower() == y.lower() for x, y in zip(a, b) if not x.isascii())


def typed_with_accents(question):
    ws = qnorm.WORD.findall(unicodedata.normalize("NFC", str(question or "")))
    return bool(ws) and sum(not w.isascii() for w in ws) / len(ws) >= 0.15


def merge(rule_text, llm_text, question=None):
    if question is not None and typed_with_accents(question): return rule_text
    if toks(rule_text) != toks(llm_text): return rule_text
    parts = re.split(r"([^\W_]+)", unicodedata.normalize("NFC", rule_text)); lw = TOK.findall(unicodedata.normalize("NFC", llm_text.strip()))
    for k, i in enumerate(range(1, len(parts), 2)):
        w = parts[i]; o = lw[k]
        if not w.isascii() or any(c.isdigit() for c in w): continue
        if k > 0 and w[:1].isupper(): continue
        if o.lower() != w.lower(): parts[i] = qnorm._case_like(w, o.lower())
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out"); ap.add_argument("--model", default="")
    ap.add_argument("--max-len", type=int, default=4096); ap.add_argument("--quant", default=""); ap.add_argument("--fake-responder", default="")
    ap.add_argument("--cache", default="")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import ingest, pickle
    qs = [json.loads(l) for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8")]
    has_ans = os.path.isfile(os.path.join(a.split, "answers.jsonl"))
    rows, norms = [], {}
    for q in qs:
        b = q["business_id"]
        if b not in norms:
            p = os.path.join(a.cache, b + ".pkl") if a.cache else ""
            biz = pickle.load(open(p, "rb")) if p and os.path.isfile(p) else ingest.load_business(os.path.join(a.split, "data", b))
            norms[b] = qnorm.Normalizer.for_split(a.split, exclude_business=b if has_ans else None, extra_texts=qnorm.Normalizer.entity_texts(biz))
        r = norms[b].clean(q["question"], q["as_of_date"]); rows.append((q, r))
    prompts = [(("q", q["id"]), r["clean"]) for q, r in rows]
    prompts += [(("acc", q["id"]), qnorm.strip_accents(q["question"])) for q, r in rows if not q["question"].isascii()]
    convs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": t}] for _, t in prompts]
    if a.fake_responder:
        g = {}; exec(compile(open(a.fake_responder, encoding="utf-8").read(), a.fake_responder, "exec"), g)
        texts = [g["responder"](c) for c in convs]
    else:
        sys.argv = [sys.argv[0]]; import agent
        eng = agent.Engine(argparse.Namespace(model=a.model, tp=1, max_len=a.max_len, think=False, eager=False, lora="", quant=a.quant,
                                              gpu_util=0.9, max_num_seqs=0, max_batched_tokens=0, spec_mtp=0, prefix_budget=0))
        texts, _ = eng.gen(convs, [0.0] * len(convs), [1] * len(convs), 256, think=False)
    out = dict(zip([k for k, _ in prompts], texts))
    n = agree = ok = 0; aw = ac = bc = qa = qb = nq = 0
    with open(os.path.join(a.out, "questions_clean.jsonl"), "w", encoding="utf-8") as f:
        for q, r in rows:
            t = (out.get(("q", q["id"])) or "").strip()
            good = check(r["clean"], t)
            final = merge(r["clean"], t, q["question"]) if good else r["clean"]
            n += 1; ok += good; agree += final == r["clean"]
            f.write(json.dumps({"id": q["id"], "business_id": q["business_id"], "as_of_date": q["as_of_date"], "question": q["question"],
                                "clean_rule": r["clean"], "clean_llm": t, "llm_ok": good, "clean": final, "agree": final == r["clean"],
                                "period": r["period"], "changes": r["changes"]}, ensure_ascii=False) + "\n")
            if not q["question"].isascii():
                ref = [x.lower() for x in TOK.findall(unicodedata.normalize("NFC", q["question"]))]
                plain = qnorm.strip_accents(q["question"]); lt = (out.get(("acc", q["id"])) or "").strip()
                hyp_l = [x.lower() for x in TOK.findall(lt if check(plain, lt) else plain)]
                hyp_b = [x.lower() for x in TOK.findall(norms[q["business_id"]].restore_only(plain))]
                if len(hyp_l) == len(ref) == len(hyp_b):
                    nq += 1; aw += len(ref)
                    ac += sum(x == y for x, y in zip(ref, hyp_l)); qa += hyp_l == ref
                    bc += sum(x == y for x, y in zip(ref, hyp_b)); qb += hyp_b == ref
    acc = {"questions": n, "llm_output_accepted": ok, "final_equals_rules": agree, "accented_questions": nq,
           "llm_word_acc": round(ac / max(aw, 1), 4), "llm_question_acc": round(qa / max(nq, 1), 4),
           "bigram_word_acc": round(bc / max(aw, 1), 4), "bigram_question_acc": round(qb / max(nq, 1), 4)}
    json.dump(acc, open(os.path.join(a.out, "accuracy.json"), "w"), indent=1)
    print("QNORM_LLM", json.dumps(acc), flush=True)


if __name__ == "__main__":
    main()
