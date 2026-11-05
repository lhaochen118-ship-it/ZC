#!/usr/bin/env python3
"""Stage 1.5 of the submitted flow: the model restates each question the rules left in the plain style of the training
questions (same meaning; every number, date, code and name kept), so that the rule reader can read it. The restatement
becomes the question's cleaned text ('_clean') for agent.py: the rule program then votes on it next to the model's own
programs (--qw), and --confirm takes an answer early only when the two agree; the original question stays authoritative
in every prompt. Style examples are the most similar TRAINING questions (bag of words; no test question is ever an example).

Job: canon_llm.py <split> <out_split> --model <path> [--quant fp8] [--cache DIR] [--n-examples 6] [--fake-responder f.py]
Writes <out_split>/questions.jsonl (the questions with '_clean' where a restatement passed the check) and
<out_split>/canon_log.jsonl (question, restatement, accepted)."""
import os, re, sys, json, math, pickle, shutil, argparse, unicodedata
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qnorm, qparse

SYSTEM = ("Bạn chuẩn hoá câu hỏi của chủ cửa hàng. Viết lại câu hỏi theo đúng văn phong của các CÂU MẪU (ngắn, rõ, tiếng Việt có "
          "dấu, không viết tắt), GIỮ NGUYÊN Ý NGHĨA: cùng con số cần tính, cùng khoảng thời gian, cùng điều kiện lọc. Giữ nguyên "
          "chính xác mọi tên hàng, tên khách, tên người, mã, số, ngày tháng, đơn vị. Không thêm, không bỏ điều kiện, không trả lời. "
          "Nếu câu đã rõ thì giữ gần như nguyên văn. Chỉ in đúng một câu hỏi.")
TOK = re.compile(r"[^\W_]+", re.U)


def keep_tokens(q):
    """tokens a restatement must keep: every token with a digit and every capitalised word inside the sentence."""
    ws = TOK.findall(unicodedata.normalize("NFC", q)); return [w.lower() for i, w in enumerate(ws) if re.search(r"\d", w) or (i > 0 and w[:1].isupper())]


def check(q, p):
    if not p or "\n" in p.strip() or len(p) > 3 * len(q) + 60: return False
    have = {t.lower() for t in TOK.findall(unicodedata.normalize("NFC", p))}
    return all(t in have for t in keep_tokens(q))


class Examples:
    """the training questions as style examples: the k most similar (cosine over accent-free word counts, idf-weighted)."""
    def __init__(self, path):
        self.qs = []
        for l in open(path, encoding="utf-8"):
            b, _, q = l.rstrip("\n").partition("\t")
            if q: self.qs.append(q)
        self.vecs = [Counter(qparse.untelex(qparse.K(q)).split()) for q in self.qs]
        df = Counter(w for v in self.vecs for w in v); n = len(self.vecs)
        self.idf = {w: math.log(n / (1 + c)) + 1 for w, c in df.items()}
        self.norms = [math.sqrt(sum((c * self.idf.get(w, 1)) ** 2 for w, c in v.items())) or 1 for v in self.vecs]

    def top(self, q, k):
        v = Counter(qparse.untelex(qparse.K(q)).split()); nv = math.sqrt(sum((c * self.idf.get(w, 1)) ** 2 for w, c in v.items())) or 1
        sc = [(sum(c * x.get(w, 0) * self.idf.get(w, 1) ** 2 for w, c in v.items()) / (nv * nn), i) for i, (x, nn) in enumerate(zip(self.vecs, self.norms))]
        return [self.qs[i] for s, i in sorted(sc, reverse=True)[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out"); ap.add_argument("--model", default=""); ap.add_argument("--quant", default="")
    ap.add_argument("--cache", default=""); ap.add_argument("--n-examples", type=int, default=6); ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--fake-responder", default="")
    ap.add_argument("--questions", default="", help="questions to restate instead of the split's (a Kaggle dataset file is looked "
                    "up under /kaggle/input by name); for the validation jobs")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    for f in ("calendar.csv", "rulebook.md"):
        if os.path.isfile(os.path.join(a.split, f)): shutil.copy(os.path.join(a.split, f), a.out)
    if not os.path.exists(os.path.join(a.out, "data")): os.symlink(os.path.abspath(os.path.join(a.split, "data")), os.path.join(a.out, "data"))
    qfile = os.path.join(a.split, "questions.jsonl")
    if a.questions:
        qfile = a.questions
        if not os.path.isfile(qfile):
            import glob
            qfile = next(iter(sorted(glob.glob("/kaggle/input/**/" + os.path.basename(a.questions), recursive=True))), "")
        if not qfile:
            if os.environ.get("ZC_SMOKE") == "1": qfile = os.path.join(a.split, "questions.jsonl")
            else: sys.exit(f"questions file {a.questions} not found")
    qs = [json.loads(l) for l in open(qfile, encoding="utf-8") if l.strip()]
    if os.environ.get("ZC_SMOKE") == "1": qs = qs[:30]
    ex = Examples(os.path.join(os.path.dirname(os.path.abspath(__file__)), "qnorm_train_questions.txt"))
    convs = []
    for q in qs:
        shots = "\n".join(f"- {x}" for x in ex.top(q["question"], a.n_examples))
        convs.append([{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": f"CÂU MẪU (văn phong cần theo):\n{shots}\n\nCÂU CẦN CHUẨN HOÁ:\n{q['question']}"}])
    if a.fake_responder:
        g = {}; exec(compile(open(a.fake_responder, encoding="utf-8").read(), a.fake_responder, "exec"), g)
        texts = [g["responder"](c) for c in convs]
    elif qs:
        sys.argv = [sys.argv[0]]; import agent
        eng = agent.Engine(argparse.Namespace(model=a.model, tp=1, max_len=a.max_len, think=False, eager=False, lora="", quant=a.quant,
                                              gpu_util=0.9, max_num_seqs=0, max_batched_tokens=0, spec_mtp=0, prefix_budget=0))
        texts, _ = eng.gen(convs, [0.0] * len(convs), [1] * len(convs), 160, think=False)
    else: texts = []
    norms = {}; n_ok = 0
    with open(os.path.join(a.out, "questions.jsonl"), "w", encoding="utf-8") as f, open(os.path.join(a.out, "canon_log.jsonl"), "w", encoding="utf-8") as lg:
        for q, t in zip(qs, texts):
            t = (t or "").strip().strip('"').strip(); ok = check(q["question"], t)
            rec = {k: v for k, v in q.items() if k != "_clean"}
            if ok:
                b = q["business_id"]
                if b not in norms:
                    p = os.path.join(a.cache, b + ".pkl") if a.cache else ""
                    try:
                        biz = pickle.load(open(p, "rb")) if p and os.path.isfile(p) else None
                        norms[b] = qnorm.Normalizer.for_split(a.split, extra_texts=qnorm.Normalizer.entity_texts(biz) if biz else ())
                    except Exception: norms[b] = None
                try: c = norms[b].clean(t, q["as_of_date"]) if norms[b] else {"clean": t, "period": None, "key": qparse.K(t)}
                except Exception: c = {"clean": t, "period": None, "key": qparse.K(t)}
                rec["_clean"] = {"clean": c["clean"], "period": c.get("period"), "key": c.get("key")}; n_ok += 1
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            lg.write(json.dumps({"id": q["id"], "question": q["question"], "canonical": t, "accepted": ok}, ensure_ascii=False) + "\n")
    print(f"CANON {n_ok} / {len(qs)} restated", flush=True)


if __name__ == "__main__":
    main()
