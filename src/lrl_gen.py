#!/usr/bin/env python3
"""Rule cards written by the solution's own model (plan 3, docs/16 §2.3; cards: src/lrl.py). GPU job (run_kaggle.py <split> OUT
--script lrl_gen.py --model ... --mode M), also runnable at inference on a new split (no answer is read by any mode but induce,
which reads TRAINING answers only):

  compile   P2: every row of the split's rulebook.md -> a card (triggers = how an owner may say the term, content, toolkit hint)
  aliases   P3: categorical values of each shop's canonical tables -> cards of the ways an owner may write them (per shop)
  translit  loanwords of the toolkit / rulebook vocabulary and how they are typed as they sound in Vietnamese (data + cards)
  newterm   DR / tr_newterm: a checked (question, program) -> a new rulebook term, its row and the question using it
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


def json_objects(text):
    """every complete flat JSON object of a (possibly cut) output: a list cut by the token cap still gives its first items."""
    out = []
    for m in re.finditer(r"\{[^{}]*\}", (text or "").split("</think>")[-1]):
        try: out.append(json.loads(m.group(0)))
        except Exception: continue
    return out


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
    trig = [str(t).strip() for t in (triggers or []) if re.search(r"[^\W_]{2}", str(t)) and len(str(t)) <= 60][:16]
    content = str(content or "").strip()
    if not trig or len(re.findall(r"[^\W_]+", content)) < 3 or content.strip(". ") == "": return None   # no template echo ('...')
    if kind not in ("period", "lexicon", "definition", "unit", "value", "check"): kind = "definition"
    c = {"kind": kind, "trigger": {"words": trig}, "content": str(content).strip()[:400], "code_hint": valid_hint(hint, names), "source": source}
    c["trigger"].update({k: v for k, v in extra.items() if k in ("bid", "industry", "section", "rulebook") and v})
    if extra.get("check"): c["check"] = valid_check(extra["check"])
    c["id"] = "R" + hashlib.sha1(json.dumps([kind, trig, c["content"], c["code_hint"]], ensure_ascii=False).encode()).hexdigest()[:8]
    return c


# ---------------------------------------------------------------- prompts of each mode
def compile_items(split):
    import hints as hintsmod, agent
    rb = open(os.path.join(split, "rulebook.md"), encoding="utf-8").read()
    import lrl
    rk = lrl.rb_key(rb)
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
        items.append(({"term": term, "definition": definition, "section": sec, "rulebook": rk}, u))
    return items


CAT_PARAM = {"payment": "payment", "payment_method": "payment", "method": "payment", "channel": "channel", "group": "group",
             "item_group": "group", "status": "status", "shift": "shift", "unit": "base_unit", "base_unit": "base_unit",
             "customer_group": "customer_group", "brand": "brand", "size": "size", "item_type": "kind", "area": None, "kind": None}


def alias_values(canon_dir):
    """{bid, table, col, values} of the categorical columns of every shop's canonical tables (a canon dir: <bid>.pkl with "C")."""
    out = []
    for f in sorted(glob.glob(os.path.join(canon_dir, "*.pkl"))):
        d = pickle.load(open(f, "rb")); C = d.get("C") if isinstance(d, dict) else None
        for t, df in (C or {}).items():
            if not hasattr(df, "columns"): continue
            for col in df.columns:
                if col not in CAT_PARAM: continue
                vals = [str(v) for v in df[col].dropna().astype(str).unique() if str(v).strip()][:40]
                if 1 <= len(vals) <= 40: out.append({"bid": os.path.basename(f)[:-4], "table": t, "col": col, "values": vals})
    return out


def alias_prompt(m):
    return (f"Cột '{m['table']}.{m['col']}' trong dữ liệu một cửa hàng có các giá trị: {json.dumps(m['values'], ensure_ascii=False)}.\n"
            "Với mỗi giá trị, liệt kê 2-6 cách chủ cửa hàng có thể viết nó trong câu hỏi: viết tắt, không dấu, từ thân mật, từ địa "
            "phương, từ tiếng Anh viết theo cách đọc tiếng Việt. JSON: {\"<giá trị>\": [\"<cách viết>\", ...], ...}")


def alias_items(split, cache, values_file=""):
    """per shop of the split: the categorical values (from a precomputed values file when given, else the shop's data)."""
    if values_file and os.path.isfile(values_file):
        bids = {os.path.basename(d) for d in glob.glob(os.path.join(split, "data", "*"))}
        return [(m, alias_prompt(m)) for m in map(json.loads, open(values_file, encoding="utf-8")) if not bids or m["bid"] in bids]
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
                m = {"bid": b, "table": t, "col": col, "values": vals}; items.append((m, alias_prompt(m)))
    return items


def translit_items(split):
    rb = open(os.path.join(split, "rulebook.md"), encoding="utf-8").read()[:6000]
    u = (f"# TOOLKIT\n{api_doc()}\n\n# RULEBOOK (đầu)\n{rb}\n\n"
         "Chủ cửa hàng ở Việt Nam hay dùng từ tiếng Anh (hoặc tiếng nước ngoài) cho các khái niệm bán hàng ở trên, và nhiều khi gõ "
         "theo cách ĐỌC bằng tiếng Việt (ví dụ 'online' -> 'on lai', 'menu' -> 'mê nu'). Liệt kê tới 40 từ như vậy. JSON: "
         "[{\"english\": \"...\", \"vietnamese\": \"từ tiếng Việt tương ứng\", \"phonetic\": [\"2-4 cách gõ theo cách đọc\"]}, ...]")
    return [({"k": k}, u) for k in range(4)]


def newterm_items(recs):
    """DR / tr_newterm (docs/16 §3): a checked (question, program) -> a NEW business term for what the program computes, the rulebook
    row defining it (consistent with the program) and the question rewritten with the term. Labels stay the program's answer."""
    items = []
    for r in recs:
        u = ("Dưới đây là một câu hỏi của chủ cửa hàng và chương trình (toolkit bên dưới) trả lời đúng câu đó. Hãy NGHĨ RA một thuật ngữ "
             "nghiệp vụ MỚI (2-5 từ, chưa có trong toolkit, như một cửa hàng tự đặt trong rulebook của mình) cho đại lượng / cách tính mà "
             "chương trình làm, viết một dòng rulebook định nghĩa CHÍNH XÁC thuật ngữ đó theo dữ liệu (khớp đúng chương trình: kỳ, lọc, "
             "cách tính), rồi viết lại câu hỏi dùng thuật ngữ mới thay cho phần mô tả cách tính (giữ nguyên kỳ thời gian, tên hàng, số).\n\n"
             f"# TOOLKIT\n{api_doc()}\n\n# CÂU HỎI (hôm nay = {r['as_of_date']})\n{r['question']}\n\n# CHƯƠNG TRÌNH\n{r['program']}\n\n"
             "JSON: {\"term\": \"...\", \"definition\": \"...\", \"question\": \"...\"}")
        items.append(({"id": r["id"], "business_id": r["business_id"], "as_of_date": r["as_of_date"], "program": r["program"],
                       "orig_question": r["question"], "answer": r.get("answer")}, u))
    return items


UNIT_EXAMPLES = """## Ví dụ 1 — dòng rulebook: | Khách sộp nhất | khách có giá trị trung bình một đơn cao nhất trong kỳ, chỉ xét khách có từ 3 đơn trở lên trong kỳ |
{"name": "khach_sop_nhat", "signature": "khach_sop_nhat(p, k=1)", "doc": "k khách có giá trị TB / đơn cao nhất trong kỳ p, chỉ khách >= 3 đơn", "triggers": ["khách sộp", "khach sop", "khách chịu chi"], "code": "def khach_sop_nhat(p, k=1):\n    so_don = customer_sales(p, by=\"n_orders\")\n    tb = customer_sales(p, by=\"aov\")\n    return rank(tb[so_don >= 3], k)"}
## Ví dụ 2 — dòng rulebook: | Giờ cao điểm | khung một giờ tròn theo giờ tạo đơn có nhiều đơn nhất trong kỳ; đáp án là giờ bắt đầu khung |
{"name": "gio_cao_diem", "signature": "gio_cao_diem(p)", "doc": "giờ bắt đầu của khung một giờ có nhiều đơn nhất trong kỳ p", "triggers": ["giờ cao điểm", "gio cao diem", "giờ đông khách"], "code": "def gio_cao_diem(p):\n    o = orders(p)\n    gio = pd.to_datetime(o[\"order_time\"]).dt.hour\n    dem = gio.value_counts().sort_index()\n    return int(dem.idxmax())"}
## Ví dụ 3 — dòng rulebook: | Số đợt giao trong tháng | số đợt có ngày giao trong kỳ; mỗi đợt của mỗi đơn tính một đợt |
{"name": "so_dot_giao", "signature": "so_dot_giao(p)", "doc": "số đợt giao có ngày giao trong kỳ p (mỗi đợt của mỗi đơn)", "triggers": ["đợt giao", "chuyến giao", "dot giao"], "code": "def so_dot_giao(p):\n    g = C[\"deliveries\"].copy()\n    g[\"delivery_date\"] = pd.to_datetime(g[\"delivery_date\"]).dt.normalize()\n    g = g[p.mask(g[\"delivery_date\"])]\n    return int(g[[\"order_id\", \"trip_no\"]].drop_duplicates().shape[0])"}
## Ví dụ 4 — dòng rulebook: | Mặt hàng được ưa chuộng nhất | mặt hàng có mặt trong nhiều đơn nhất trong kỳ |
{"name": "hang_ua_chuong", "signature": "hang_ua_chuong(p, k=1)", "doc": "k mặt hàng có mặt trong nhiều đơn nhất trong kỳ p", "triggers": ["ưa chuộng", "ua chuong", "được chuộng"], "code": "def hang_ua_chuong(p, k=1):\n    return rank(item_sales(p, by=\"n_orders\"), k)"}
## Ví dụ 5 — dòng rulebook: | Làm tròn | tiền làm tròn đến đồng |
{"name": null}"""


def _rows(df, n=2, width=160):
    """n example rows of a table as 'col=value' text (the writer sees real labels, units, codes)."""
    try:
        cols = [c for c in df.columns if not str(c).startswith("_")][:14]
        sub = df[cols].dropna(how="all").head(200)
        sub = sub.sample(min(n, len(sub)), random_state=0) if len(sub) else sub
        return ["  " + "; ".join(f"{c}={str(v)[:24]}" for c, v in r.items() if str(v) not in ("nan", "None", "NaT"))[:width] for _, r in sub.iterrows()]
    except Exception: return []


def _tables_card(C, T, max_chars=4500):
    import prompting
    out = ["## C (bảng chuẩn)", prompting.compact_card(C, "", max_chars=1500)]
    for t, df in (C or {}).items():
        if hasattr(df, "columns") and t not in ("orders", "lines"): out += [f"C[{t!r}] ví dụ:"] + _rows(df)
    out.append("## T (bảng gốc)")
    for t, df in (T or {}).items():
        cols = [str(c) for c in getattr(df, "columns", []) if not str(c).startswith("_")][:25]
        out.append(f"T[{t!r}] ({len(df)} dòng): {', '.join(cols)}"); out += _rows(df)
    return "\n".join(out)[:max_chars]


_SHOPS = {}                                                             # bid -> (C, data_end, T) of the shops unit_items built
# the writer reasons in plain text past any token cap (G0d: 108 / 125 outputs without JSON at 2560 tokens): constrained JSON with a
# bounded 'plan' field for its reasoning, then the function (vLLM structured outputs)
UNIT_SCHEMA = {"type": "object", "properties": {
    "plan": {"type": "string", "maxLength": 700},
    "name": {"type": ["string", "null"], "maxLength": 48},
    "signature": {"type": "string", "maxLength": 120},
    "doc": {"type": "string", "maxLength": 300},
    "triggers": {"type": "array", "items": {"type": "string", "maxLength": 60}, "maxItems": 8},
    "code": {"type": "array", "items": {"type": "string", "maxLength": 200}, "maxItems": 40}},   # one line per item (G0e: one-line
    # code joined by ';' where a '#' comment swallowed the rest of the function)
    "required": ["plan", "name", "signature", "doc", "triggers", "code"], "additionalProperties": False}


def unit_items(split, cache):
    """per shop of the split: the rows of its rulebook industry section + the rows its rulebook has that neither the training
    nor the public rulebook has (shopctx.rule_changes) -> a prompt asking for ONE function over the shop's tables (or null)."""
    import ingest, automap, hints as hintsmod, shopctx
    rb0 = open(os.path.join(split, "rulebook.md"), encoding="utf-8").read()
    items = []
    bids = sorted(os.path.basename(d) for d in glob.glob(os.path.join(split, "data", "*")) if os.path.isdir(d))
    for b in bids:
        rbp = os.path.join(split, "rulebooks", b + ".md")
        rb = open(rbp, encoding="utf-8").read() if os.path.isfile(rbp) else rb0
        try:
            bp = os.path.join(cache or "", b + ".pkl")
            biz = pickle.load(open(bp, "rb")) if cache and os.path.isfile(bp) else ingest.load_business(os.path.join(split, "data", b))
        except Exception as e:
            print(f"[unit] {b}: {type(e).__name__}: {e}", flush=True); continue
        rows = []
        ind = shopctx.industry_of(rb, biz)
        if ind: rows += [(t, d, ind[0]) for t, d in hintsmod.rulebook_rows(ind[1]).values()]
        try:                                                            # rows neither the training nor the public rulebook has
            known = set()
            for f in (shopctx.TRAIN_RULEBOOK, shopctx.PUBLIC_RULEBOOK):
                try: known |= set(hintsmod.rulebook_rows(open(f, encoding="utf-8").read()))
                except OSError: pass
            rows += [(t, d, "mới") for k, (t, d) in hintsmod.rulebook_rows(rb).items() if k not in known]
        except Exception as e: print(f"[unit] {b}: new rows unavailable ({type(e).__name__}: {e})", flush=True)
        if not rows: continue                                           # no industry section, no new row: nothing to write
        try:
            C = automap.build(biz)[0]; _SHOPS[b] = (C, automap.parse_readme(biz.readme).get("data_end"), biz.tables)
        except Exception as e:
            print(f"[unit] {b}: automap {type(e).__name__}: {e}", flush=True); continue
        seen = set(); card = _tables_card(C, biz.tables)
        for term, definition, sec in rows:
            if term in seen: continue
            seen.add(term)
            u = (f"# TOOLKIT (đã có sẵn trong chương trình)\n{api_doc()}\n\n# DỮ LIỆU CỦA CỬA HÀNG {b}\n{card}\n\n# VÍ DỤ\n{UNIT_EXAMPLES}\n\n"
                 f"# DÒNG RULEBOOK CẦN VIẾT HÀM (mục '{sec}')\n| {term} | {definition} |\n\n"
                 "Viết MỘT hàm Python tính đúng thuật ngữ này trên dữ liệu của cửa hàng: dùng hàm toolkit khi có hàm hợp, nếu không thì pandas "
                 "trên C hoặc T (đúng tên bảng / cột có ở trên). Tham số đầu là kỳ p (Period: p.mask(series), p.start, p.end) khi thuật ngữ "
                 "phụ thuộc kỳ; tham số khác phải có giá trị mặc định. Không đọc file. Nếu dòng này không phải một đại lượng tính được "
                 "(chỉ là cách ghi, cách làm tròn...) thì name = null và code rỗng. Trả lời bằng MỘT đối tượng JSON: plan (vài câu ngắn: "
                 "bảng / cột dùng, cách tính), name, signature, doc, triggers, code = MẢNG các dòng mã (mỗi dòng một phần tử, thụt lề 4 dấu "
                 "cách, không dồn nhiều lệnh vào một dòng). Hàm toolkit, C, T, pd, np, re đã có sẵn: không import chúng.")
            items.append(({"bid": b, "term": term, "definition": definition, "section": sec, "rulebook": __import__("lrl").rb_key(rb)}, u))
    return items


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
        if mode in ("translit", "induce") and (j is None or (isinstance(j, list) and not any(isinstance(x, dict) for x in j))):
            j = json_objects(t) or None                                 # a list cut by the token cap (first_json then finds an inner list)
        if j is None: continue
        if mode == "compile" and isinstance(j, dict):
            trig = list(j.get("triggers") or []) + [meta["term"]]
            c = card("definition", trig, f"{meta['term']}: {j.get('content') or meta['definition']}", j.get("code_hint"), names,
                     "rulebook:" + meta["term"], section=meta.get("section"), rulebook=meta.get("rulebook"))
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
        elif mode == "newterm" and isinstance(j, dict):
            t, d, q = (str(j.get(k) or "").strip() for k in ("term", "definition", "question"))
            if t and d and q and t.lower() in q.lower() and 2 <= len(t.split()) <= 6 and q != meta["orig_question"]:
                out.append(dict(meta, term=t, definition=d, question=q, row=f"| {t} | {d} |"))
        elif mode == "unit" and isinstance(j, dict):
            if not j.get("name") or not j.get("code"): continue
            name = re.sub(r"[^0-9a-zA-Z_]", "_", str(j["name"]))[:48].strip("_") or "unit"
            code = "\n".join(str(x) for x in j["code"]) if isinstance(j["code"], list) else str(j["code"])
            code = "\n".join(l for l in code.split("\n") if not re.match(r"\s*(from|import)\s+(toolkit|datamate)\b", l))   # names already defined
            m = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
            if m and m.group(1) != name: code = code.replace(f"def {m.group(1)}(", f"def {name}(", 1)
            u = {"kind": "unit", "bid": meta["bid"], "term": meta["term"], "definition": meta["definition"][:400], "section": meta.get("section"),
                 "rulebook": meta.get("rulebook"), "name": name, "signature": str(j.get("signature") or name)[:120], "doc": str(j.get("doc") or "")[:300],
                 "triggers": [str(t) for t in (j.get("triggers") or [])][:8], "code": code[:3000]}
            u["id"] = "U" + hashlib.sha1(json.dumps([u["bid"], u["term"], code], ensure_ascii=False).encode()).hexdigest()[:8]
            out.append(u)
        elif mode == "induce":
            for e in (j if isinstance(j, list) else [j]):
                if not isinstance(e, dict): continue
                c = card(e.get("kind") or "definition", e.get("triggers"), e.get("content"), e.get("code_hint"), names,
                         "induced:" + meta["id"], check=e.get("check"))
                if c: c["case"] = {"id": meta["id"], "business_id": meta["business_id"]}; out.append(c)
    return out


def verify_units(cands, split, cache, min_agree=2):
    """per (shop, term): every candidate unit is run on the shop's data (src/units.py); among those that run with a non-degenerate
    result, the one whose results on the test periods most candidates share is kept (self-consistency over the samples; ties ->
    the earliest sample, temperature 0)."""
    import units, ingest, automap
    from collections import defaultdict, Counter
    out, done, fails, shops = [], set(), {}, {}
    good = defaultdict(list)
    for u in cands:
        k = (u["bid"], u["term"])
        if u["bid"] not in shops and u["bid"] in _SHOPS: shops[u["bid"]] = _SHOPS[u["bid"]]
        if u["bid"] not in shops:
            try:
                bp = os.path.join(cache or "", u["bid"] + ".pkl")
                biz = pickle.load(open(bp, "rb")) if cache and os.path.isfile(bp) else ingest.load_business(os.path.join(split, "data", u["bid"]))
                C = automap.build(biz)[0]; de = automap.parse_readme(biz.readme).get("data_end")
                shops[u["bid"]] = (C, de, biz.tables)
            except Exception as e: shops[u["bid"]] = None; print(f"[unit] {u['bid']}: {type(e).__name__}: {e}", flush=True)
        if shops[u["bid"]] is None: continue
        C, de, T = shops[u["bid"]]
        ok, why, sample = units.verify(u, C, de, T)
        if ok:
            u["verified"] = {"sample": str(sample)[:200]}; good[k].append(u)
        else: fails.setdefault(k, why)
    for k, us in good.items():
        votes = Counter(u.get("_sig") for u in us)
        best = max(us, key=lambda u: (votes[u.get("_sig")], -us.index(u)))
        if votes[best.get("_sig")] < min_agree: fails[k] = f"no agreement ({votes[best.get('_sig')]} / {len(us)} runnable samples)"; continue
        best["verified"]["agree"] = f"{votes[best.get('_sig')]}/{len(us)} runnable samples"
        best.pop("_sig", None); out.append(best); done.add(k)
    print(f"[unit] {len(out)} units verified for {len({u['bid'] for u in out})} shops; {len([k for k in fails if k not in done])} rows without a "
          f"working unit", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out"); ap.add_argument("--model", default=""); ap.add_argument("--mode", required=True)
    ap.add_argument("--cases", default=""); ap.add_argument("--cache", default=""); ap.add_argument("--quant", default="")
    ap.add_argument("--max-len", type=int, default=8192); ap.add_argument("--fake-responder", default="")
    ap.add_argument("--shard", default=""); ap.add_argument("--samples", type=int, default=3, help="unit: samples per rulebook row")
    ap.add_argument("--think-units", type=int, default=0, help="unit: the writer reasons first (4096-token cap)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    names = api_names()
    if a.mode == "compile": items = compile_items(a.split)
    elif a.mode == "aliases":
        vf = a.cases if (a.cases and os.path.isfile(a.cases)) else next(iter(glob.glob(f"/kaggle/**/{os.path.basename(a.cases)}", recursive=True)), "") if a.cases else ""
        items = alias_items(a.split, a.cache, vf)
    elif a.mode == "translit": items = translit_items(a.split)
    elif a.mode == "newterm":
        p = a.cases if os.path.isfile(a.cases) else next(iter(glob.glob(f"/kaggle/**/{os.path.basename(a.cases)}", recursive=True)), a.cases)
        items = newterm_items([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()])
    elif a.mode == "induce":
        p = a.cases if os.path.isfile(a.cases) else next(iter(glob.glob(f"/kaggle/**/{os.path.basename(a.cases)}", recursive=True)), a.cases)
        items = induce_items([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()])
    elif a.mode == "unit": items = unit_items(a.split, a.cache)
    else: sys.exit(f"unknown mode {a.mode}")
    if a.shard:
        i, n = map(int, a.shard.split("/")); items = items[i::n]
    if os.environ.get("ZC_SMOKE") == "1": items = items[:8]
    if a.mode == "unit" and a.samples > 1:                              # k samples per row (after sharding: a row's samples stay
        items = [it for it in items for _ in range(a.samples)]          # together for the self-consistency choice)
    convs = [[{"role": "system", "content": SYS}, {"role": "user", "content": u}] for _, u in items]
    think = a.mode == "induce" or (a.mode == "unit" and a.think_units)   # newterm without thinking: with it 930 / 940 answers of
                                                                        # lrl0c reasoned past the cap (13/11); the task is simple
    if a.fake_responder:
        g = {}; exec(compile(open(a.fake_responder, encoding="utf-8").read(), a.fake_responder, "exec"), g)
        texts = [g["responder"](c) for c in convs]
    elif convs:
        sys.argv = [sys.argv[0]]; import agent
        eng = agent.Engine(argparse.Namespace(model=a.model, tp=1, max_len=a.max_len, think=think, eager=False, lora="", quant=a.quant,
                                              gpu_util=0.9, max_num_seqs=0, max_batched_tokens=0, spec_mtp=0, prefix_budget=0))
        temp = 0.7 if a.mode == "translit" else 0.0
        temps = [temp] * len(convs)
        if a.mode == "unit": temps = [0.0 if i % max(1, a.samples) == 0 else 0.7 for i in range(len(convs))]
        mt = (4096 if a.mode == "unit" else 6144) if think else (2560 if a.mode == "unit" else 900)   # unit: the writer often reasons in
        js = UNIT_SCHEMA if a.mode == "unit" and not think else None   # plain text before its JSON (G0c: 33 / 50 cut at 900); 2048 was
        try: texts, _ = eng.gen(convs, temps, [4242 + i for i in range(len(convs))], mt, think=think, json_schema=js)
        except Exception as e:                                          # a schema this vLLM / xgrammar refuses: unconstrained
            if js is None: raise
            print(f"[lrl_gen] constrained JSON unavailable ({type(e).__name__}: {str(e)[:200]}): unconstrained", flush=True)
            texts, _ = eng.gen(convs, temps, [4242 + i for i in range(len(convs))], mt, think=think)
        # too few: 199 / 199 induction answers of lrl0b reasoned past the cap without reaching their JSON (13/11)
    else: texts = []
    cards = to_cards(a.mode, items, texts, names)
    if a.mode == "unit": cards = verify_units(cards, a.split, a.cache, 2 if a.samples >= 3 else 1)
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
