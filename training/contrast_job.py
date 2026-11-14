#!/usr/bin/env python3
"""v2 E (handoff/04 §3.E): systematic minimal-contrast pairs on TRAINING shops, generated and checked by models + execution.

No contrast axis is chosen by hand or taken from evaluation errors: the axes ("slots") are read from the AST of each checked
training program (the period call, the metric function, each keyword argument, numeric arguments), and one slot is drawn
uniformly among the slots that program has; a "style" slot (same meaning, the way an owner may type it — the model enumerates
the styles itself) gives invariance pairs.
Per item (one checked program of a training question, Process/train_data/p3/d1_all, 6 held-out shops excluded):
  1 the base model rewrites the question changing exactly that slot and writes the matching program (JSON, constrained);
  2 both programs run on the shop's data: kept when the new one runs and (contrast) its answer differs / (style) is equal;
  3 blind back-check: the base model WITH thinking reads the new question and the two programs in random order and says which one
    answers it; kept only when it picks the new program (contrast) — a pair the model itself cannot tell is not a label.
Outputs OUT/contrast.jsonl {id, business_id, as_of_date, slot, question, program, answer, orig_question, orig_program, orig_answer,
backcheck}. Usage (GPU job, all GPUs): contrast_job.py <train_split> <out> --model M --items F [--dp 4] [--style-share 0.25]
[--fake-responder f.py]   (worker mode: --worker i/n)"""
import os, re, sys, ast, json, glob, time, zlib, random, pickle, signal, argparse, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src") if os.path.isdir(os.path.join(ROOT, "src")) else ROOT
sys.path.insert(0, SRC)

SYS = ("Bạn giúp tạo dữ liệu cho một trợ lý viết chương trình Python (toolkit bên dưới) trả lời câu hỏi của chủ cửa hàng nhỏ ở Việt Nam "
       "về dữ liệu bán hàng. Trả lời đúng định dạng được yêu cầu.")
GEN_SCHEMA = {"type": "object", "properties": {"question": {"type": "string", "maxLength": 300}, "program": {"type": "string", "maxLength": 600}},
              "required": ["question", "program"], "additionalProperties": False}
PERIOD_FNS = None


def slots(code):
    """the slots of a checked program: ('period', call text), ('metric', name), ('kwarg', name=value), ('number', value)."""
    global PERIOD_FNS
    if PERIOD_FNS is None:
        import dm
        PERIOD_FNS = set(dm.PERIOD_API)
    out = []
    try: tree = ast.parse(code)
    except SyntaxError: return out
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Name): continue
        name = n.func.id
        if name in PERIOD_FNS: out.append(("period", ast.get_source_segment(code, n) or name)); continue
        out.append(("metric", name))
        for kw in n.keywords:
            if kw.arg: out.append(("kwarg", f"{kw.arg}={ast.get_source_segment(code, kw.value)}"))
        for a in n.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, (int, float)) and not isinstance(a.value, bool): out.append(("number", repr(a.value)))
    seen, uniq = set(), []
    for s in out:
        if s not in seen: seen.add(s); uniq.append(s)
    return uniq


KIND_TEXT = {"period": "KỲ THỜI GIAN (đang là {v}): đổi sang một kỳ khác hợp lý, có thể viết kỳ theo cách khác người Việt hay viết",
             "metric": "ĐẠI LƯỢNG / ĐỐI TƯỢNG ĐƯỢC TÍNH (đang dùng hàm {v}): đổi sang một đại lượng hoặc đối tượng đếm khác gần nghĩa nhưng khác kết quả",
             "kwarg": "ĐIỀU KIỆN / THAM SỐ {v}: đổi giá trị hoặc bỏ / thêm điều kiện này",
             "number": "CON SỐ {v} trong câu (số lượng xếp hạng, ngưỡng...): đổi sang số khác",
             "style": "CÁCH VIẾT: giữ NGUYÊN nghĩa và chương trình, viết lại câu theo một cách chủ cửa hàng hay nhắn mà câu gốc chưa dùng "
                      "(viết tắt, không dấu, từ địa phương, cách viết kỳ khác, câu cụt...) — chương trình mới phải GIỐNG chương trình gốc"}


def gen_prompt(it, api):
    k, v = it["slot"]
    return (f"# TOOLKIT\n{api}\n\n# CÂU GỐC (hôm nay = {it['as_of_date']})\n{it['question']}\n\n# CHƯƠNG TRÌNH GỐC (đúng)\n{it['program']}\n\n"
            f"# NHIỆM VỤ\nViết MỘT câu hỏi mới chỉ khác câu gốc ở {KIND_TEXT[k].format(v=v)}; mọi phần khác giữ nguyên. Viết chương trình "
            "mới trả lời đúng câu mới (chỉ sửa phần tương ứng của chương trình gốc, cùng định dạng ANSWER). JSON: {\"question\": \"...\", \"program\": \"...\"}")


def check_prompt(question, as_of, progs):
    return (f"# CÂU HỎI (hôm nay = {as_of})\n{question}\n\n" + "\n\n".join(f"# CHƯƠNG TRÌNH {i + 1}\n{p}" for i, p in enumerate(progs)) +
            "\n\n# NHIỆM VỤ\nChương trình nào trả lời ĐÚNG câu hỏi (đúng đại lượng, kỳ, điều kiện)? Trả lời đúng một dòng: <choice>1</choice> "
            "hoặc <choice>2</choice>.")


_SHOP = {}


def shop(split, b):
    if b not in _SHOP:
        import ingest, automap
        biz = ingest.load_business(os.path.join(split, "data", b))
        _SHOP[b] = (automap.build(biz)[0], automap.parse_readme(biz.readme).get("data_end"), biz)
    return _SHOP[b]


class _TO(Exception): pass


def run_program(split, b, as_of, code, timeout=20):
    """(ok, answer) of a toolkit program on shop b at as_of (the solver's namespace: toolkit bound to the shop + C, T)."""
    import dm, agent, pandas as pd, numpy as np
    C, de, biz = shop(split, b)
    g = {"C": C, "T": {k: v.copy() for k, v in biz.tables.items()}, "pd": pd, "np": np, "re": re, "TODAY": pd.Timestamp(as_of)}
    try:
        g.update(dm.api(dm.Shop(C, as_of, {"data_end": de}), pd.Timestamp(as_of)))
        def _a(*_): raise _TO()
        old = signal.signal(signal.SIGALRM, _a); signal.alarm(timeout)
        try: exec(compile(code, "<program>", "exec"), g)
        finally: signal.alarm(0); signal.signal(signal.SIGALRM, old)
        if "ANSWER" not in g: return False, None
        return True, agent.fmt_answer(g["ANSWER"])
    except dm.Unanswerable: return True, {"type": "unanswerable", "answer": None}
    except BaseException: return False, None


def same(a, b):
    import judge
    return judge.canon(a) == judge.canon(b)


def first_json(t):
    t = (t or "").split("</think>")[-1]
    m = re.search(r"\{.*\}", t, re.S)
    try: return json.loads(m.group(0)) if m else None
    except Exception: return None


def engine(a, think):
    import agent
    sys.argv = [sys.argv[0]]
    return agent.Engine(argparse.Namespace(model=a.model, tp=1, max_len=8192, think=think, eager=False, lora="", quant="fp8", gpu_util=0.9,
                                           max_num_seqs=128, max_batched_tokens=16384, spec_mtp=0, prefix_budget=0))


def worker(a):
    import dm
    i, n = map(int, a.worker.split("/"))
    items = [json.loads(l) for l in open(a.items_local, encoding="utf-8") if l.strip()][i::n]
    rng = random.Random(1000 + i)
    for it in items:
        sl = slots(it["program"])
        it["slot"] = ("style", "-") if (not sl or rng.random() < a.style_share) else sl[rng.randrange(len(sl))]
    api = dm.API_DOC
    fake = None
    if a.fake_responder:
        g = {}; exec(compile(open(a.fake_responder, encoding="utf-8").read(), a.fake_responder, "exec"), g); fake = g["responder"]
    convs = [[{"role": "system", "content": SYS}, {"role": "user", "content": gen_prompt(it, api)}] for it in items]
    if fake: texts = [fake(c) for c in convs]
    else:
        eng = engine(a, False)
        texts = eng.gen(convs, [0.8] * len(convs), [7000 + j for j in range(len(convs))], 700, think=False, json_schema=GEN_SCHEMA)[0]
    cands = []
    for it, t in zip(items, texts):
        j = first_json(t)
        if not isinstance(j, dict) or not j.get("question") or "ANSWER" not in str(j.get("program")): continue
        q2, p2 = j["question"].strip(), j["program"].strip()
        if q2 == it["question"].strip(): continue
        ok, ans2 = run_program(a.split, it["business_id"], it["as_of_date"], p2)
        if not ok: continue
        ok0, ans0 = run_program(a.split, it["business_id"], it["as_of_date"], it["program"])
        if not ok0: continue
        style = it["slot"][0] == "style"
        if style != same(ans0, ans2): continue                         # contrast must change the answer, style must not
        cands.append(dict(id=f"{it['id']}~c{zlib.crc32(q2.encode()) % 10**6}", business_id=it["business_id"], as_of_date=it["as_of_date"],
                          slot=list(it["slot"]), question=q2, program=p2, answer=ans2, orig_question=it["question"], orig_program=it["program"],
                          orig_answer=ans0))
    # blind back-check of the contrast pairs (thinking)
    todo = [c for c in cands if c["slot"][0] != "style"]
    order = [rng.random() < 0.5 for _ in todo]
    convs = [[{"role": "system", "content": SYS}, {"role": "user", "content": check_prompt(c["question"], c["as_of_date"],
              [c["program"], c["orig_program"]] if o else [c["orig_program"], c["program"]])}] for c, o in zip(todo, order)]
    if fake: texts = [fake(c) for c in convs]
    else:
        del eng; import gc; gc.collect()
        eng = engine(a, True)
        texts = eng.gen(convs, [0.0] * len(convs), [9000 + j for j in range(len(convs))], 3072, think=True)[0]
    for c, o, t in zip(todo, order, texts):
        m = re.findall(r"<choice>\s*([12])\s*</choice>", (t or "").split("</think>")[-1])
        pick = int(m[-1]) if m else None
        c["backcheck"] = pick is not None and (pick == 1) == o
    keep = [c for c in cands if c["slot"][0] == "style" or c.get("backcheck")]
    tag = os.path.splitext(os.path.basename(a.items_local))[0]           # items_<k>: one output per worker
    with open(os.path.join(a.out, f"contrast_{tag}.jsonl"), "w", encoding="utf-8") as w:
        for c in keep: w.write(json.dumps(c, ensure_ascii=False, default=str) + "\n")
    print(f"[contrast {i}/{n}] items {len(items)} generated {len(cands)} back-checked {sum(1 for c in todo if c.get('backcheck'))}/{len(todo)} kept {len(keep)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out"); ap.add_argument("--model", required=True); ap.add_argument("--items", required=True)
    ap.add_argument("--dp", type=int, default=4); ap.add_argument("--style-share", type=float, default=0.25)
    ap.add_argument("--fake-responder", default=""); ap.add_argument("--worker", default=""); ap.add_argument("--items-local", default="")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.worker: return worker(a)
    t0 = time.time()
    p = a.items if os.path.isfile(a.items) else next(iter(glob.glob(f"/kaggle/**/{os.path.basename(a.items)}", recursive=True)), a.items)
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("set", "dev") == "dev"]
    if os.environ.get("ZC_SMOKE") == "1": rows = [r for r in rows if r["business_id"] in sorted({x["business_id"] for x in rows})[:2]][:24]
    rows.sort(key=lambda r: r["business_id"])                            # a worker's shops stay together (data loaded once each)
    bids = sorted({r["business_id"] for r in rows}); part = {b: k % a.dp for k, b in enumerate(bids)}
    loc = os.path.join(a.out, "items.jsonl")
    with open(loc, "w", encoding="utf-8") as w:
        for k in range(a.dp):
            for r in rows:
                if part[r["business_id"]] == k: w.write(json.dumps(r, ensure_ascii=False) + "\n")
    # worker i takes lines i::dp of a file ordered by shop group: re-split so that each worker holds whole shops
    groups = [[r for r in rows if part[r["business_id"]] == k] for k in range(a.dp)]
    procs = []
    for k, g in enumerate(groups):
        f = os.path.join(a.out, f"items_{k}.jsonl")
        with open(f, "w", encoding="utf-8") as w:
            for r in g: w.write(json.dumps(r, ensure_ascii=False) + "\n")
        cmd = [sys.executable, os.path.abspath(__file__), a.split, a.out, "--model", a.model, "--items", a.items, "--items-local", f,
               "--worker", "0/1", "--style-share", str(a.style_share)] + (["--fake-responder", a.fake_responder] if a.fake_responder else [])
        env = dict(os.environ, **({} if a.fake_responder else {"CUDA_VISIBLE_DEVICES": str(k)}))
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(os.path.join(a.out, f"worker{k}.log"), "w"), stderr=subprocess.STDOUT))
        if a.fake_responder: procs[-1].wait()
    rcs = [p.wait() for p in procs]
    # each worker wrote contrast_0of1.jsonl into the same folder: give them distinct names as they finish
    keep = []
    for k in range(a.dp):
        lg = os.path.join(a.out, f"worker{k}.log")
        print(open(lg, encoding="utf-8", errors="replace").read()[-400:], flush=True)
    for f in glob.glob(os.path.join(a.out, "contrast_items_*.jsonl")):
        keep += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    with open(os.path.join(a.out, "contrast.jsonl"), "w", encoding="utf-8") as w:
        for c in keep: w.write(json.dumps(c, ensure_ascii=False, default=str) + "\n")
    from collections import Counter
    print("CONTRAST", json.dumps({"rc": rcs, "kept": len(keep), "by_slot": dict(Counter(c["slot"][0] for c in keep)), "s": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
