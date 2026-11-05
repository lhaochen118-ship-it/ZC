#!/usr/bin/env python3
"""Paraphrases of a split's questions by the solution's own LLM (Qwen3.5-9B, no extra model), for a generalization stress
test of the rule reader (src/qparse.py): the same question in other natural words, as another shop owner would ask it,
every name / number / date / code kept. NOT used to fit anything: tests/eval_split.py --perturb-q para:<file> answers the
paraphrases and scores them against the ORIGINAL question's reference answer (a paraphrase must keep the meaning; the
check below rejects one that lost a number, a date, a code or a name, and the errors are read before any fix).

Job (GPU, Kaggle): paraphrase_llm.py <split> <out> --model <path> [--n 3] [--quant fp8] [--fake-responder f.py]
writes <out>/paraphrases.jsonl: {id, business_id, k, question, paraphrase, ok} and <out>/summary.json."""
import os, re, sys, json, argparse, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SYSTEM = ("Bạn giúp kiểm thử một trợ lý dữ liệu cho cửa hàng. Nhận MỘT câu hỏi của chủ cửa hàng; viết lại câu hỏi đó theo một "
          "cách hỏi KHÁC mà một chủ cửa hàng khác ở Việt Nam có thể dùng (đổi từ ngữ, đổi trật tự, văn nói hoặc văn viết, có thể "
          "dùng từ địa phương), nhưng GIỮ NGUYÊN Ý NGHĨA: cùng con số cần tính, cùng khoảng thời gian, cùng điều kiện lọc. Giữ "
          "nguyên CHÍNH XÁC mọi tên hàng, tên khách, tên người, mã, số, ngày tháng, đơn vị và mốc thời gian như trong câu gốc. "
          "Không thêm điều kiện mới, không bỏ điều kiện nào, không trả lời câu hỏi. Chỉ in đúng một câu hỏi mới.")
STYLES = ["", "Hãy hỏi ngắn gọn kiểu nhắn tin.", "Hãy hỏi lịch sự, đầy đủ.", "Hãy đổi cấu trúc câu thật khác câu gốc."]
TOK = re.compile(r"[^\W_]+", re.U)


def keep_tokens(q):
    """what a paraphrase must keep: every token with a digit (numbers, dates, codes, sizes) and every capitalised word
    inside the sentence (names), case-insensitively."""
    ws = TOK.findall(unicodedata.normalize("NFC", q)); out = []
    for i, w in enumerate(ws):
        if re.search(r"\d", w) or (i > 0 and w[:1].isupper()): out.append(w.lower())
    return out


def check(q, p):
    if not p or "\n" in p.strip() or len(p) > 3 * len(q) + 40: return False
    have = set(t.lower() for t in TOK.findall(unicodedata.normalize("NFC", p)))
    return all(t in have for t in keep_tokens(q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out"); ap.add_argument("--model", default="")
    ap.add_argument("--n", type=int, default=3); ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--quant", default=""); ap.add_argument("--fake-responder", default="")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    qs = [json.loads(l) for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8") if l.strip()]
    if os.environ.get("ZC_SMOKE") == "1": qs = qs[:40]
    items = [(q, k) for q in qs for k in range(a.n)]
    convs = [[{"role": "system", "content": SYSTEM + (" " + STYLES[k % len(STYLES)] if STYLES[k % len(STYLES)] else "")},
              {"role": "user", "content": q["question"]}] for q, k in items]
    if a.fake_responder:
        g = {}; exec(compile(open(a.fake_responder, encoding="utf-8").read(), a.fake_responder, "exec"), g)
        texts = [g["responder"](c) for c in convs]
    else:
        sys.argv = [sys.argv[0]]; import agent
        eng = agent.Engine(argparse.Namespace(model=a.model, tp=1, max_len=a.max_len, think=False, eager=False, lora="", quant=a.quant,
                                              gpu_util=0.9, max_num_seqs=0, max_batched_tokens=0, spec_mtp=0, prefix_budget=0))
        texts, _ = eng.gen(convs, [0.9] * len(convs), [1000 + i for i in range(len(convs))], 160, think=False)
    n_ok = 0
    with open(os.path.join(a.out, "paraphrases.jsonl"), "w", encoding="utf-8") as f:
        for (q, k), t in zip(items, texts):
            t = (t or "").strip().strip('"').strip()
            ok = check(q["question"], t); n_ok += ok
            f.write(json.dumps({"id": q["id"], "business_id": q["business_id"], "k": k, "question": q["question"], "paraphrase": t, "ok": ok},
                               ensure_ascii=False) + "\n")
    summ = {"questions": len(qs), "paraphrases": len(items), "kept": n_ok}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("PARAPHRASE", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
