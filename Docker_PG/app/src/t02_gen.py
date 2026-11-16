#!/usr/bin/env python3
import os, sys, json, re, argparse, time, signal

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

SCHEMA = {"type": "object", "properties": {"questions": {"type": "array", "minItems": 1, "maxItems": 8, "items": {"type": "object",
    "properties": {"question": {"type": "string", "maxLength": 240}, "program": {"type": "string", "maxLength": 700}},
    "required": ["question", "program"], "additionalProperties": False}}}, "required": ["questions"], "additionalProperties": False}

SYS = "Bạn là chuyên gia phân tích dữ liệu bán hàng. Chỉ trả lời bằng một khối JSON hợp lệ."


def bank_examples(k=6):
    try:
        rows = [json.loads(l) for l in open(os.path.join(HERE, "refs", "bank", "examples.jsonl"), encoding="utf-8") if l.strip()]
    except Exception: return ""
    import lrl
    hid = lrl.hidden_names()
    rows = [r for r in rows if r.get("code") and r.get("question") and len(r["code"]) < 400 and not (hid and lrl.mentions(r["code"], hid))]
    step = max(1, len(rows) // k)
    return "\n".join(f"- {r['question']}\n  {r['code']}" for r in rows[::step][:k])


def prompt(b, tables, rules, api, today, n, ex=""):
    return (f"# API TOOLKIT (chỉ được dùng các hàm này, pandas `pd` và bảng `C`)\n{api}\n\n"
            + (f"# VÍ DỤ CHƯƠNG TRÌNH (cửa hàng khác)\n{ex}\n\n" if ex else "") + f"# DỮ LIỆU CỦA CỬA HÀNG {b}\n{tables}\n\n"
            f"# RULEBOOK (thuật ngữ | định nghĩa)\n{rules}\n\n# NHIỆM VỤ (hôm nay = {today})\nViết {n} câu hỏi KHÁC NHAU mà chủ cửa hàng này "
            "có thể nhắn, trả lời được từ dữ liệu ở trên: đa dạng đại lượng (doanh thu, số đơn, khách, sản phẩm, tỉ lệ, xếp hạng, so sánh "
            "hai kỳ, thuật ngữ trong rulebook), đa dạng kỳ (ngày, tuần, tháng, quý, năm, hôm qua, tháng này...), có câu thân mật / viết tắt / "
            "không dấu. Với mỗi câu viết chương trình Python ngắn dùng toolkit, kết thúc bằng ANSWER = {\"type\": ..., \"answer\": ...} "
            "(type: number, list, string, date hoặc percent). JSON: {\"questions\": [{\"question\": \"...\", \"program\": \"...\"}, ...]}")


ERR = {}


def run_program(sh, today, program, hidden, t=20):
    import dm, agent, pandas as pd, numpy as np
    C, de, T = sh
    g = {"C": C, "T": {k: v.copy() for k, v in (T or {}).items()}, "pd": pd, "np": np, "re": re, "TODAY": pd.Timestamp(today)}
    try:
        g.update(dm.api(dm.Shop(C, today, {"data_end": de}), pd.Timestamp(today)))
        for h in hidden: g.pop(h, None)
        def _a(*_): raise TimeoutError()
        old = signal.signal(signal.SIGALRM, _a); signal.alarm(t)
        try: exec(compile(program, "<t02>", "exec"), g)
        finally: signal.alarm(0); signal.signal(signal.SIGALRM, old)
        a = agent.fmt_answer(g["ANSWER"]) if "ANSWER" in g else None
    except BaseException as e:
        k = type(e).__name__ + ": " + str(e)[:80]; ERR[k] = ERR.get(k, 0) + 1; return None
    v = (a or {}).get("answer")
    if a is None or v is None or (isinstance(v, (list, dict, str)) and not v): ERR["empty answer"] = ERR.get("empty answer", 0) + 1; return None
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out"); ap.add_argument("--model", default=""); ap.add_argument("--shard", default="")
    ap.add_argument("--n-per-shop", type=int, default=12); ap.add_argument("--per-call", type=int, default=6); ap.add_argument("--cache", default="")
    ap.add_argument("--quant", default="fp8"); ap.add_argument("--max-len", type=int, default=12288); ap.add_argument("--seed", type=int, default=5151)
    ap.add_argument("--fake-responder", default="")
    a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out, exist_ok=True)
    import lrl_gen, lrl, hints as hintsmod, html
    hidden = lrl.hidden_names()
    qs = [json.loads(l) for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8") if l.strip()]
    days = {}
    for q in qs: days.setdefault(q["business_id"], []).append(q["as_of_date"])
    shops = sorted(days)
    if a.shard:
        i, n = map(int, a.shard.split("/")); shops = shops[i::n]
    rb = open(os.path.join(a.split, "rulebook.md"), encoding="utf-8").read()
    rules = "\n".join(f"| {t} | {d[:160]} |" for t, d in hintsmod.rulebook_rows(rb).values())[:3500]
    api = lrl_gen.api_doc(); ex = bank_examples()
    items, convs = [], []
    for b in shops:
        sh = lrl_gen._shop_load(a.split, a.cache, b)
        if not sh: continue
        tables = lrl_gen._tables_card(sh[0], sh[2])
        ds = sorted(set(days[b]))
        for k in range(max(1, -(-a.n_per_shop // a.per_call))):
            today = ds[(k * 7919) % len(ds)]
            items.append((b, today)); convs.append([{"role": "system", "content": SYS}, {"role": "user", "content": prompt(b, tables, rules, api, today, a.per_call, ex)}])
    if a.fake_responder:
        g = {}; exec(compile(open(a.fake_responder, encoding="utf-8").read(), a.fake_responder, "exec"), g)
        texts = [g["responder"](c) for c in convs]
    elif convs:
        os.environ["ZC_THINK_KW"] = "explicit"
        sys.argv = [sys.argv[0]]; import agent
        eng = agent.Engine(argparse.Namespace(model=a.model, tp=1, max_len=a.max_len, think=False, eager=False, lora="", quant=a.quant,
                                              gpu_util=0.92, max_num_seqs=128, max_batched_tokens=16384, spec_mtp=0, prefix_budget=0))
        seeds = [a.seed + i for i in range(len(convs))]
        try: texts = eng.gen(convs, [0.7] * len(convs), seeds, 3000, think=False, json_schema=SCHEMA)[0]
        except Exception as e:
            print(f"[t02] constrained JSON unavailable ({type(e).__name__}: {str(e)[:200]}): unconstrained", flush=True)
            texts = eng.gen(convs, [0.7] * len(convs), seeds, 3000, think=False)[0]
    else: texts = []
    n_parsed = sum(1 for x in texts if isinstance(lrl_gen.first_json(x), dict))
    for x in texts[:2]: print(f"[t02] sample output: {str(x)[:600]!r}", flush=True)
    keep, n_gen, seen = [], 0, set()
    for (b, today), t in zip(items, texts):
        j = lrl_gen.first_json(t)
        for e in ((j or {}).get("questions") or []) if isinstance(j, dict) else []:
            if not isinstance(e, dict): continue
            q, p = html.unescape(str(e.get("question") or "")).strip(), html.unescape(str(e.get("program") or "")).strip()
            if not q or "ANSWER" not in p or (b, q.lower()) in seen: continue
            n_gen += 1; seen.add((b, q.lower()))
            if hidden and lrl.mentions(p, hidden): continue
            ans = run_program(lrl_gen._shop_load(a.split, a.cache, b), today, p, hidden)
            if ans is not None: keep.append({"bid": b, "question": q, "as_of_date": today, "program": p, "answer": ans})
    tag = "t02" + (("_" + a.shard.replace("/", "of")) if a.shard else "")
    with open(os.path.join(a.out, f"cards_{tag}.jsonl"), "w", encoding="utf-8") as w:
        for r in keep: w.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"[t02] {len(keep)} / {n_gen} generated pairs run, {len({r['bid'] for r in keep})} / {len(shops)} shops, {n_parsed} / {len(texts)} outputs parsed, "
          f"errors {sorted(ERR.items(), key=lambda kv: -kv[1])[:4]}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
