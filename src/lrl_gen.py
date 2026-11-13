#!/usr/bin/env python3
"""Rule cards written by the solution's own model (plan 3, docs/16 §2.3; cards: src/lrl.py). GPU job (run_kaggle.py <split> OUT
--script lrl_gen.py --model ... --mode M), also runnable at inference on a new split (no answer is read by any mode but induce,
which reads TRAINING answers only):

  compile   P2: every row of the split's rulebook.md -> a card (triggers = how an owner may say the term, content, toolkit hint)
  aliases   P3: categorical values of each shop's canonical tables -> cards of the ways an owner may write them (per shop)
  translit  loanwords of the toolkit / rulebook vocabulary and how they are typed as they sound in Vietnamese (data + cards)
  induce    P1: cases where the model answered TRAINING questions wrongly (training/data/lrl_cases.py) -> candidate general
            rules (accepted later only by tests/lrl_accept.py: checks on logged samples, A/B runs on other shops)

Output: <out>/cards_<mode>.jsonl (+ translit.jsonl). Every card is validated: JSON, non-empty triggers, toolkit names of
code_hint exist, a check expression compiles. Usage:
  lrl_gen.py <split> <out> --model M --mode compile|aliases|translit|induce [--cases F] [--cache D] [--quant fp8] [--fake-responder f.py]"""
import os, re, sys, ast, json, glob, pickle, argparse, hashlib, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SYS = ("Bạn giúp xây dựng bộ luật cho một trợ lý viết chương trình Python trả lời câu hỏi của chủ cửa hàng nhỏ ở Việt Nam về dữ "
       "liệu bán hàng. Trả lời ĐÚNG một đối tượng JSON, không giải thích.")


def api_doc():
    import dm
    return dm.API_DOC


def api_names():
    import api_spec
    return {e["name"] for e in api_spec.build_api_spec()} | {"TODAY", "ANSWER", "len", "round", "sum", "int", "float", "max", "min", "sorted", "list"}


def first_json(text):
    """the first JSON object / array in a model output (after any think block)."""
    t = (text or "").split("</think>")[-1]
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M)
    order = sorted((("[", "]"), ("{", "}")), key=lambda oc: t.find(oc[0]) if t.find(oc[0]) >= 0 else 1 << 30)
    for op, cl in order:                                                # the outermost value: whichever bracket comes first
        i = t.find(op)
        while i >= 0:
            depth = 0
            for j in range(i, len(t)):
                if t[j] == op: depth += 1
                elif t[j] == cl:
                    depth -= 1
                    if depth == 0:
                        try: return json.loads(t[i:j + 1])
                        except Exception: break
            i = t.find(op, i + 1)
    return None


def valid_hint(h, names):
    """code_hint kept when it parses and calls only toolkit names (parameters alone, 'payment="cod"', are kept too)."""
    if not h or not isinstance(h, str): return None
    h = h.strip().strip("`")
    try: tree = ast.parse(h if "(" in h else f"f({h})")
    except SyntaxError: return None
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)} - {"f"}
    return h if calls <= names else None


def valid_check(c):
    if not c or not isinstance(c, str): return None
    try: compile(c, "<check>", "eval"); return c if len(c) < 400 and "__" not in c and "import" not in c else None
    except SyntaxError: return None


def card(kind, triggers, content, hint, names, source, **extra):
    trig = [str(t).strip() for t in (triggers or []) if str(t).strip() and len(str(t)) <= 60][:16]
    if not trig or not content: return None
    c = {"kind": kind, "trigger": {"words": trig}, "content": str(content).strip()[:400], "code_hint": valid_hint(hint, names), "source": source}
    c["trigger"].update({k: v for k, v in extra.items() if k in ("bid", "industry", "section") and v})
    if extra.get("check"): c["check"] = valid_check(extra["check"])
    c["id"] = "R" + hashlib.sha1(json.dumps([kind, trig, c["content"], c["code_hint"]], ensure_ascii=False).encode()).hexdigest()[:8]
    return c


# ---------------------------------------------------------------- prompts of each mode
def compile_items(split):
    import hints as hintsmod, agent
    rb = open(os.path.join(split, "rulebook.md"), encoding="utf-8").read()
    secs = agent.rulebook_sections(rb); items = []
    seen = set()
    for k, (term, definition) in hintsmod.rulebook_rows(rb).items():
        if (term, definition) in seen: continue
        seen.add((term, definition))
        sec = next((t for t, body in secs.items() if term in body), None)
        u = (f"# TOOLKIT\n{api_doc()}\n\n# RULEBOOK SECTION: {sec or 'chung'}\n# RULE\n| {term} | {definition} |\n\n"
             "Viết thẻ luật cho thuật ngữ này. JSON: {\"triggers\": [5-12 cụm từ ngắn chủ cửa hàng có thể gõ trong câu hỏi để nói thuật "
             "ngữ này: từ đồng nghĩa, cách nói thân mật, viết tắt, viết không dấu], \"content\": \"một câu: thuật ngữ nghĩa là gì và tính "
             "thế nào\", \"code_hint\": \"lời gọi toolkit với đúng tham số luật này cần, ví dụ revenue(p, payment=\\\"chuyển khoản\\\"); "
             "null nếu không có hàm nào hợp\"}")
        items.append(({"term": term, "definition": definition, "section": sec}, u))
    return items


CAT_PARAM = {"payment": "payment", "channel": "channel", "group": "group", "status": "status", "shift": "shift", "unit": "base_unit",
             "base_unit": "base_unit", "customer_group": "customer_group", "brand": "brand"}


def alias_items(split, cache):
    import ingest, automap
    items = []
    bids = sorted(os.path.basename(d) for d in glob.glob(os.path.join(split, "data", "*")) if os.path.isdir(d))
    for b in bids:
        try:
            pc = os.path.join(cache or "", b + ".canon.pkl")
            if cache and os.path.isfile(pc): C = pickle.load(open(pc, "rb"))
            else:
                bp = os.path.join(cache or "", b + ".pkl")
                biz = pickle.load(open(bp, "rb")) if cache and os.path.isfile(bp) else ingest.load_business(os.path.join(split, "data", b))
                C = automap.build(biz)[0]
        except Exception as e:
            print(f"[aliases] {b}: {type(e).__name__}: {e}", flush=True); continue
        for t, df in C.items():
            if not hasattr(df, "columns"): continue
            for col in df.columns:
                if col not in CAT_PARAM: continue
                vals = [str(v) for v in df[col].dropna().astype(str).unique() if str(v).strip()][:40]
                if not 1 <= len(vals) <= 40: continue
                u = (f"Cột '{t}.{col}' trong dữ liệu một cửa hàng có các giá trị: {json.dumps(vals, ensure_ascii=False)}.\n"
                     "Với mỗi giá trị, liệt kê 2-6 cách chủ cửa hàng có thể viết nó trong câu hỏi: viết tắt, không dấu, từ thân mật, từ địa "
                     "phương, từ tiếng Anh viết theo cách đọc tiếng Việt. JSON: {\"<giá trị>\": [\"<cách viết>\", ...], ...}")
                items.append(({"bid": b, "table": t, "col": col, "values": vals}, u))
    return items


def translit_items(split):
    rb = open(os.path.join(split, "rulebook.md"), encoding="utf-8").read()[:6000]
    u = (f"# TOOLKIT\n{api_doc()}\n\n# RULEBOOK (đầu)\n{rb}\n\n"
         "Chủ cửa hàng ở Việt Nam hay dùng từ tiếng Anh (hoặc tiếng nước ngoài) cho các khái niệm bán hàng ở trên, và nhiều khi gõ "
         "theo cách ĐỌC bằng tiếng Việt (ví dụ 'online' -> 'on lai', 'menu' -> 'mê nu'). Liệt kê tới 40 từ như vậy. JSON: "
         "[{\"english\": \"...\", \"vietnamese\": \"từ tiếng Việt tương ứng\", \"phonetic\": [\"2-4 cách gõ theo cách đọc\"]}, ...]")
    return [({"k": k}, u) for k in range(4)]


def induce_items(cases):
    items = []
    for c in cases:
        u = ("Trợ lý viết chương trình (toolkit bên dưới) đã trả lời SAI một câu hỏi của chủ cửa hàng. Hãy rút ra tối đa 2 LUẬT TỔNG "
             "QUÁT (áp dụng cho mọi cửa hàng: KHÔNG nhắc tên hàng, tên khách, tên người, số liệu riêng của cửa hàng này) giúp lần sau "
             "viết đúng chương trình.\n\n"
             f"# TOOLKIT\n{api_doc()}\n\n# CÂU HỎI (hôm nay = {c['as_of_date']})\n{c['question']}\n\n# CHƯƠNG TRÌNH CỦA TRỢ LÝ\n{c.get('program') or '(không có)'}\n"
             f"Kết quả: {json.dumps(c.get('answer'), ensure_ascii=False)}\n\n# ĐÁP ÁN ĐÚNG\n{json.dumps(c['reference'], ensure_ascii=False)}\n"
             + (f"\n# MỘT CHƯƠNG TRÌNH ĐÚNG\n{c['gold_program']}\n" if c.get("gold_program") else "") +
             "\nJSON: [{\"kind\": \"period|lexicon|definition|unit|value|check\", \"triggers\": [\"cụm từ trong câu hỏi khiến luật áp dụng "
             "(viết như chủ cửa hàng gõ, gồm cả bản không dấu)\"], \"content\": \"một câu luật\", \"code_hint\": \"đoạn toolkit đúng hoặc null\", "
             "\"check\": \"biểu thức Python trên q (câu hỏi không dấu, chữ thường) và code (chương trình) = True khi chương trình VI PHẠM "
             "luật, ví dụ 're.search(r\\\"\\\\bmoi thang\\\\b\\\", q) and \\\"month_seq\\\" not in code'; null nếu không viết được\"}]")
        items.append((c, u))
    return items


# ---------------------------------------------------------------- outputs -> cards
def to_cards(mode, items, texts, names):
    out = []
    for (meta, _), t in zip(items, texts):
        j = first_json(t)
        if j is None: continue
        if mode == "compile" and isinstance(j, dict):
            trig = list(j.get("triggers") or []) + [meta["term"]]
            c = card("definition", trig, f"{meta['term']}: {j.get('content') or meta['definition']}", j.get("code_hint"), names,
                     "rulebook:" + meta["term"], section=meta.get("section"))
            if c: c["rule"] = meta["definition"][:400]; out.append(c)
        elif mode == "aliases" and isinstance(j, dict):
            p = CAT_PARAM.get(meta["col"])
            for v, forms in j.items():
                if v not in meta["values"] or not isinstance(forms, list): continue
                c = card("value", forms, f"'{forms[0]}' là giá trị '{v}' của cột {meta['table']}.{meta['col']}", f'{p}="{v}"' if p else None,
                         names | {"f"}, f"data:{meta['table']}.{meta['col']}", bid=meta["bid"])
                if c: out.append(c)
        elif mode == "translit" and isinstance(j, list):
            for e in j:
                if not isinstance(e, dict) or not e.get("phonetic"): continue
                c = card("lexicon", e["phonetic"], f"'{e['phonetic'][0]}' là cách đọc '{e.get('english')}' = {e.get('vietnamese')}", None, names,
                         "translit:" + str(e.get("english")))
                if c: c["translit"] = {"english": e.get("english"), "vietnamese": e.get("vietnamese"), "phonetic": e["phonetic"]}; out.append(c)
        elif mode == "induce":
            for e in (j if isinstance(j, list) else [j]):
                if not isinstance(e, dict): continue
                c = card(e.get("kind") or "definition", e.get("triggers"), e.get("content"), e.get("code_hint"), names,
                         "induced:" + meta["id"], check=e.get("check"))
                if c: c["case"] = {"id": meta["id"], "business_id": meta["business_id"]}; out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out"); ap.add_argument("--model", default=""); ap.add_argument("--mode", required=True)
    ap.add_argument("--cases", default=""); ap.add_argument("--cache", default=""); ap.add_argument("--quant", default="")
    ap.add_argument("--max-len", type=int, default=8192); ap.add_argument("--fake-responder", default="")
    ap.add_argument("--shard", default="")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    names = api_names()
    if a.mode == "compile": items = compile_items(a.split)
    elif a.mode == "aliases": items = alias_items(a.split, a.cache)
    elif a.mode == "translit": items = translit_items(a.split)
    elif a.mode == "induce":
        p = a.cases if os.path.isfile(a.cases) else next(iter(glob.glob(f"/kaggle/**/{os.path.basename(a.cases)}", recursive=True)), a.cases)
        items = induce_items([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()])
    else: sys.exit(f"unknown mode {a.mode}")
    if a.shard:
        i, n = map(int, a.shard.split("/")); items = items[i::n]
    if os.environ.get("ZC_SMOKE") == "1": items = items[:8]
    convs = [[{"role": "system", "content": SYS}, {"role": "user", "content": u}] for _, u in items]
    think = a.mode == "induce"
    if a.fake_responder:
        g = {}; exec(compile(open(a.fake_responder, encoding="utf-8").read(), a.fake_responder, "exec"), g)
        texts = [g["responder"](c) for c in convs]
    elif convs:
        sys.argv = [sys.argv[0]]; import agent
        eng = agent.Engine(argparse.Namespace(model=a.model, tp=1, max_len=a.max_len, think=think, eager=False, lora="", quant=a.quant,
                                              gpu_util=0.9, max_num_seqs=0, max_batched_tokens=0, spec_mtp=0, prefix_budget=0))
        temp = 0.7 if a.mode == "translit" else 0.0
        texts, _ = eng.gen(convs, [temp] * len(convs), [4242 + i for i in range(len(convs))], 2048 if think else 900, think=think)
    else: texts = []
    cards = to_cards(a.mode, items, texts, names)
    tag = a.mode + (("_" + a.shard.replace("/", "of")) if a.shard else "")
    with open(os.path.join(a.out, f"cards_{tag}.jsonl"), "w", encoding="utf-8") as f:
        for c in cards: f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, f"raw_{tag}.jsonl"), "w", encoding="utf-8") as f:
        for (meta, _), t in zip(items, texts): f.write(json.dumps({"meta": meta, "text": (t or "")[-3000:]}, ensure_ascii=False, default=str) + "\n")
    summ = {"mode": a.mode, "items": len(items), "cards": len(cards), "with_hint": sum(bool(c.get("code_hint")) for c in cards),
            "with_check": sum(bool(c.get("check")) for c in cards), "s": round(time.time() - t0)}
    json.dump(summ, open(os.path.join(a.out, f"summary_{tag}.json"), "w"), indent=1)
    print("LRL_GEN", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
