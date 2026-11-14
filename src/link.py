#!/usr/bin/env python3
"""Plan 5 W2 (docs/18): rule linking by the model. For every question of the split, the model reads the numbered rows of the
shop's rulebook (general rows + the rows of the shop's industry section) and names the rows the question uses; those rows (the
rulebook's own wording, + the toolkit hint of the row's compiled card when the model wrote a valid one, src/lrl_gen.py compile)
become the question's # RULES block in the solver's prompt (src/hybrid.py --links). This replaces the keyword triggers of the
compiled cards (src/lrl.py match), which missed the row whenever the owner worded the term differently. Inputs only: the
rulebook, the README and the questions; no answer is read, no hand-written rule reads the question.

    rows_for(rulebook, section_title) -> [(term, definition, section)]
    prompt(rows, question, as_of) -> conversation ; parse(text, n) -> [row numbers]
GPU job (also run by predict_flow.prepare_link, one shard per GPU):
  link.py <split> <out> --model M [--cards compile.jsonl,...] [--k 3] [--shard i/n] [--quant fp8] [--fake-responder f.py]
  writes <out>/links[_iofn].jsonl: {"id", "ids": [row numbers], "cards": [{"id", "kind", "content", "code_hint", "source"}]}"""
import os, re, sys, json, glob, time, pickle, argparse, hashlib
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SYS = ("Bạn nối câu hỏi của chủ cửa hàng nhỏ ở Việt Nam với các dòng luật (rulebook) quyết định cách tính đáp án của câu đó. "
       "Trả lời ĐÚNG một đối tượng JSON, không giải thích.")
TASK = ("# TASK\nChọn các dòng luật ở trên mà câu hỏi này dùng đến: câu hỏi nói đúng thuật ngữ đó, một cách nói đồng nghĩa / thân mật / "
        "viết tắt / không dấu của nó, hoặc hỏi đúng đại lượng mà dòng đó định nghĩa cách tính. Chỉ chọn dòng thật sự quyết định cách "
        "tính đáp án của câu này (không chọn dòng chỉ trùng một từ chung như 'tháng', 'doanh thu'); tối đa 3 dòng; không có dòng nào "
        "thì để rỗng. JSON: {\"ids\": [số thứ tự các dòng]}")
SCHEMA = {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "integer"}, "maxItems": 4}}, "required": ["ids"]}
MAX_LINK = 3
DEF_CHARS = 160


def rows_for(rulebook, section=None):
    """the rulebook rows a shop reads: every row outside the industry sections + the rows of its own industry section."""
    import hints as hintsmod, agent
    secs = agent.rulebook_sections(rulebook)
    in_sec = {}
    for title, body in secs.items():
        for k, (t, d) in hintsmod.rulebook_rows(body).items(): in_sec.setdefault(k, title)
    out, seen = [], set()
    for k, (t, d) in hintsmod.rulebook_rows(rulebook).items():
        s = in_sec.get(k)
        if s is not None and s != section: continue
        if (t, d) in seen: continue
        seen.add((t, d)); out.append((t, d, s))
    return out


def term_block(rows):
    return "# RULEBOOK ROWS (số thứ tự. thuật ngữ: nghĩa / cách tính)\n" + "\n".join(
        f"{i}. {t}: {d[:DEF_CHARS]}" + (f" [mục {s}]" if s else "") for i, (t, d, s) in enumerate(rows, 1))


def prompt(rows, question, as_of):
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": term_block(rows) + f"\n\n# QUESTION (today = {as_of})\n{question.strip()}\n\n" + TASK}]


def parse(text, n):
    t = (text or "").split("</think>")[-1]
    m = re.search(r"\{[^{}]*\}", t)
    ids = None
    if m:
        try: ids = json.loads(m.group(0)).get("ids")
        except Exception: ids = None
    if ids is None:
        m = re.search(r"ids\W{0,4}\[([0-9,\s]*)\]", t)
        ids = [int(x) for x in re.findall(r"\d+", m.group(1))] if m else None
    if not isinstance(ids, list): return None
    out = []
    for x in ids:
        try: x = int(x)
        except Exception: continue
        if 1 <= x <= n and x not in out: out.append(x)
    return out[:MAX_LINK + 1]


def vote(samples, need=2):
    """row numbers named by at least `need` of the parsed samples (one parsed sample: its own), most named first."""
    ok = [s for s in samples if s is not None]
    if not ok: return []
    c = Counter(x for s in ok for x in s)
    need = min(need, len(ok))
    return [x for x, n in c.most_common() if n >= need][:MAX_LINK]


def card_index(paths, rb_key):
    """{term: compiled card} of the cards compiled from this rulebook (their toolkit hints)."""
    out = {}
    here = os.path.dirname(os.path.abspath(__file__))
    for f in [x for x in (paths or "").split(",") if x]:
        if not os.path.isfile(f): f = os.path.join(here, "refs", "lrl", os.path.basename(f))   # a shipped card file by name
        if not os.path.isfile(f): print(f"[link] cards {f} not found", flush=True); continue
        for l in open(f, encoding="utf-8"):
            if not l.strip(): continue
            c = json.loads(l); src = str(c.get("source") or "")
            if not src.startswith("rulebook:"): continue
            rk = (c.get("trigger") or {}).get("rulebook")
            if rk and rk != rb_key: continue
            out.setdefault(src[len("rulebook:"):], c)
    return out


def hint_of(card):
    h = (card or {}).get("code_hint")
    return h if isinstance(h, str) and h.strip() and h.strip().lower() not in ("null", "none") else None


def to_card(term, definition, section, compiled):
    cid = "L" + hashlib.sha1(f"{term}|{definition}".encode("utf-8")).hexdigest()[:8]
    return {"id": cid, "kind": "definition", "content": f"{term}: {definition[:400]}", "code_hint": hint_of(compiled),
            "source": "link:" + term, "section": section}


def shop_section(split, b, rulebook, cache=""):
    """the title of the shop's rulebook industry section (README first, the data only when the README does not name it)."""
    import agent
    readme = ""
    for f in glob.glob(os.path.join(split, "data", b, "README*")):
        try: readme = open(f, encoding="utf-8", errors="replace").read(); break
        except OSError: pass
    ind = agent.industry_of(rulebook, readme) if readme else None
    if ind: return ind[0]
    try:
        import ingest, shopctx
        bp = os.path.join(cache or "", b + ".pkl")
        if cache and os.path.isfile(bp): biz = pickle.load(open(bp, "rb"))
        else:
            biz = ingest.load_business(os.path.join(split, "data", b))
            if cache:                                                   # the agent reads the same file next (ingest's own cache
                os.makedirs(cache, exist_ok=True); tmp = bp + f".tmp{os.getpid()}"   # format): the shop is read once per run
                pickle.dump(biz, open(tmp, "wb"), protocol=4); os.replace(tmp, bp)
        ind = shopctx.industry_of(rulebook, biz)
        return ind[0] if ind else None
    except Exception as e:
        print(f"[link] {b}: industry unknown ({type(e).__name__}: {e})", flush=True); return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out"); ap.add_argument("--model", default=""); ap.add_argument("--cards", default="")
    ap.add_argument("--cache", default=""); ap.add_argument("--quant", default="fp8"); ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--k", type=int, default=3, help="samples per question: 1 greedy + k-1 at 0.7; a row needs 2 votes")
    ap.add_argument("--shard", default=""); ap.add_argument("--fake-responder", default=""); ap.add_argument("--lora", default="")
    ap.add_argument("--max-df", type=float, default=0.0, help="rows linked to more than this share of the split's questions are "
                    "dropped as generic (14/11 zlk: 'Đơn tính vào doanh thu -> revenue(p)' on 27 %% of the questions pulled n_orders "
                    "questions to revenue); 0 = keep all. With --shard the share is per shard (a shard holds whole shops)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    import lrl
    sys.argv = [sys.argv[0]]
    rb0 = open(os.path.join(a.split, "rulebook.md"), encoding="utf-8").read()
    qs = [json.loads(l) for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8") if l.strip()]
    bids = sorted({q["business_id"] for q in qs})
    if a.shard:                                                         # whole shops per shard (one rows prefix per shop group)
        i, n = map(int, a.shard.split("/")); mine = set(bids[i::n]); qs = [q for q in qs if q["business_id"] in mine]
    if os.environ.get("ZC_SMOKE") == "1": qs = qs[:24]
    shop = {}
    for b in sorted({q["business_id"] for q in qs}):
        rbp = os.path.join(a.split, "rulebooks", b + ".md")
        rb = open(rbp, encoding="utf-8").read() if os.path.isfile(rbp) else rb0
        sec = shop_section(a.split, b, rb, a.cache)
        shop[b] = (rows_for(rb, sec), card_index(a.cards, lrl.rb_key(rb)), sec)
        print(f"[link] {b}: {len(shop[b][0])} rows (section {sec}), {sum(1 for c in shop[b][1].values() if hint_of(c))} compiled hints", flush=True)
    qs.sort(key=lambda q: (shop[q["business_id"]][2] or "", q["business_id"]))     # same rows prefix together (prefix cache)
    convs, temps, seeds, meta = [], [], [], []
    for qi, q in enumerate(qs):
        rows = shop[q["business_id"]][0]
        if not rows: continue
        c = prompt(rows, q["question"], q["as_of_date"])
        for j in range(max(1, a.k)):
            convs.append(c); temps.append(0.0 if j == 0 else 0.7); seeds.append(7001 + 31 * qi + j); meta.append((q["id"], j))
    if a.fake_responder:
        g = {}; exec(compile(open(a.fake_responder, encoding="utf-8").read(), a.fake_responder, "exec"), g)
        texts = [g["responder"](c) for c in convs]
    elif convs:
        import agent
        eng = agent.Engine(argparse.Namespace(model=a.model, tp=1, max_len=a.max_len, think=False, eager=False, lora=a.lora, quant=a.quant,
                                              gpu_util=0.92, max_num_seqs=128, max_batched_tokens=16384, spec_mtp=0, prefix_budget=0))
        try: texts = eng.gen(convs, temps, seeds, 64, think=False, json_schema=SCHEMA)[0]
        except Exception as e:                                          # a schema this vLLM / xgrammar refuses: unconstrained
            print(f"[link] constrained JSON unavailable ({type(e).__name__}: {str(e)[:200]}): unconstrained", flush=True)
            texts = eng.gen(convs, temps, seeds, 160, think=False)[0]
    else: texts = []
    by = {}
    for (qid, j), t in zip(meta, texts): by.setdefault(qid, []).append(t)
    tag = "links" + (("_" + a.shard.replace("/", "of")) if a.shard else "")
    voted = {q["id"]: vote([parse(t, len(shop[q["business_id"]][0])) for t in by.get(q["id"], [])]) for q in qs}
    generic = set()
    if a.max_df > 0 and qs:                                             # data-driven, like lrl.prune: a row linked to most questions
        df = Counter(shop[q["business_id"]][0][i - 1][0] for q in qs for i in voted[q["id"]])   # says nothing about any one of them
        generic = {t for t, k in df.items() if k > a.max_df * len(qs)}
        print(f"[link] generic rows dropped (> {a.max_df:.0%} of {len(qs)} questions): {sorted(generic)}", flush=True)
    n_link = n_parsed = n_hint = 0; used = Counter()
    with open(os.path.join(a.out, tag + ".jsonl"), "w", encoding="utf-8") as f:
        for q in qs:
            rows, idx, sec = shop[q["business_id"]]
            samples = [parse(t, len(rows)) for t in by.get(q["id"], [])]
            n_parsed += sum(1 for s in samples if s is not None)
            ids = [i for i in voted[q["id"]] if rows[i - 1][0] not in generic]
            cards = [to_card(*rows[i - 1], idx.get(rows[i - 1][0])) for i in ids]
            n_link += bool(cards); n_hint += sum(1 for c in cards if c["code_hint"]); used.update(rows[i - 1][0] for i in ids)
            f.write(json.dumps({"id": q["id"], "ids": ids, "samples": samples, "cards": cards}, ensure_ascii=False) + "\n")
    summ = {"questions": len(qs), "requests": len(convs), "parsed": n_parsed, "linked_questions": n_link, "cards_with_hint": n_hint,
            "top_rows": used.most_common(12), "s": round(time.time() - t0)}
    json.dump(summ, open(os.path.join(a.out, f"summary_{tag}.json"), "w"), ensure_ascii=False, indent=1)
    print("LINK", json.dumps(summ, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
