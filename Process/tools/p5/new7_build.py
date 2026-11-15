#!/usr/bin/env python3
"""New7 offline item builder (CPU): prompts for the GPU pass (src/new7_gen.py), from a run's solver log + the shop's canonical tables.

  n7a  every distinct answer group of every question: Qwen writes a dimensional certificate of the candidate (what the question
       asks for, the units of the quantities the program uses, the expression, the unit of the result).
  n3a  questions whose candidates disagree (>= 2 answer groups): each group's program is run on two row-subsampled copies of the
       shop's data (seeded, CPU) — counter-examples that show how the programs differ — and Qwen chooses the program that computes
       what the question asks (or none).
  n6a  questions whose candidates disagree: Qwen writes ONE DuckDB SELECT over the canonical tables (independent second engine);
       executed later on CPU (src/sqlesc.py) and compared with the pandas candidates.
No business rule anywhere: the checks are algebraic / executional; every semantic decision is the model's.
Usage: new7_build.py <solver_logs.jsonl> <canon_dir> <split_dir> <ingest_cache> <units.jsonl> <out_dir> [--modes n7a,n3a,n6a]"""
import os, sys, json, pickle, random, argparse
from collections import Counter, defaultdict
sys.path.insert(0, "/home/user/ZC/src")

SYS = "Bạn là chuyên gia kiểm tra dữ liệu. Chỉ trả lời bằng một khối JSON hợp lệ, không giải thích thêm."


def schema_text(C, max_chars=2500):
    import sqlesc
    return "\n".join(sqlesc.schema_lines(C))[:max_chars]


def groups_of(r):
    import agent
    g = {}
    for s in (r.get("samples") or {}).get("samples") or []:
        if not s.get("ok") or not isinstance(s.get("answer"), dict): continue
        code = next((t.get("code") for t in reversed(s.get("turns") or []) if t.get("code")), None)
        if not code: continue
        c = agent.canon_of(s["answer"]); e = g.setdefault(c, {"answer": s["answer"], "code": code, "votes": 0}); e["votes"] += 1
    return g


def subsample(src, dst, seed, frac=0.5):
    d = pickle.load(open(src, "rb")); d["v2"] = True; rng = random.Random(seed); C2 = {}
    for k, df in d["C"].items():
        try:
            idx = sorted(rng.sample(range(len(df)), max(1, int(len(df) * frac)))) if len(df) > 1 else list(range(len(df)))
            C2[k] = df.iloc[idx].reset_index(drop=True)
        except Exception: C2[k] = df
    pickle.dump(dict(d, C=C2), open(dst, "wb"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("logs"); ap.add_argument("canon"); ap.add_argument("split"); ap.add_argument("cache")
    ap.add_argument("units"); ap.add_argument("out"); ap.add_argument("--modes", default="n7a,n3a,n6a"); ap.add_argument("--tag", default="")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); modes = a.modes.split(",")
    import agent
    os.environ["ZC_HIDE_API"] = "industry"
    if a.units and a.units != "-": os.environ["ZC_UNITS"] = a.units
    agent._init_worker(a.cache, a.split)
    rows = [json.loads(l) for l in open(a.logs, encoding="utf-8") if l.strip()]
    C = {b: pickle.load(open(os.path.join(a.canon, b + ".pkl"), "rb"))["C"] for b in {r["business_id"] for r in rows}}
    sch = {b: schema_text(c) for b, c in C.items()}
    work = a.canon + "_n3a"; os.makedirs(work, exist_ok=True)
    out = {m: [] for m in modes}
    for r in rows:
        q, b, g = r["question"], r["business_id"], groups_of(r)
        head = f"# CÂU HỎI (ngày hiện tại {r.get('as_of_date')})\n{q}\n\n# DỮ LIỆU CỬA HÀNG (bảng chuẩn)\n{sch[b]}\n"
        if "n7a" in modes:
            for c, e in g.items():
                u = head + (f"\n# CHƯƠNG TRÌNH ỨNG VIÊN\n{e['code'][:2500]}\n\n# KẾT QUẢ\n{json.dumps(e['answer'], ensure_ascii=False)[:600]}\n\n"
                            "# VIỆC CẦN LÀM\nViết chứng chỉ thứ nguyên cho kết quả. Đơn vị là các ký hiệu ngắn do bạn đặt (ví dụ vnd, item, order, "
                            "customer, day, percent); phép chia viết a/b, lũy thừa a^2.\n"
                            'JSON: {"question_unit": "<đơn vị của đại lượng câu hỏi yêu cầu>", "variables": {"<tên>": "<đơn vị>"}, '
                            '"expression": "<biểu thức theo tên biến ra kết quả>", "result_unit": "<đơn vị của kết quả>"}')
                out["n7a"].append({"id": f"{r['id']}|{c}", "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": u}],
                                   "n": 1, "max_tokens": 400})
        if len(g) < 2: continue
        if "n3a" in modes:
            fx = []
            for seed in (1, 2):
                p = os.path.join(work, f"{b}.s{seed}.pkl")
                if not os.path.isfile(p): subsample(os.path.join(a.canon, b + ".pkl"), p, seed)
                fx.append(p)
            lines = []
            for j, (c, e) in enumerate(sorted(g.items(), key=lambda kv: -kv[1]["votes"])[:4]):
                alt = []
                for p in fx:
                    _, rr = agent.run_code((r["id"], b, r.get("as_of_date"), e["code"], 30, p))
                    alt.append(json.dumps(rr.get("answer"), ensure_ascii=False)[:200] if rr.get("ok") else "lỗi")
                lines.append(f"## Chương trình {j + 1} ({e['votes']} phiếu)\n{e['code'][:1800]}\nKết quả trên dữ liệu thật: "
                             f"{json.dumps(e['answer'], ensure_ascii=False)[:300]}\nTrên mẫu con 1: {alt[0]}\nTrên mẫu con 2: {alt[1]}")
            u = head + "\n# CÁC CHƯƠNG TRÌNH CHO KẾT QUẢ KHÁC NHAU\n" + "\n\n".join(lines) + (
                "\n\n# VIỆC CẦN LÀM\nCác mẫu con là phản ví dụ cho thấy các chương trình tính khác nhau. Chọn chương trình tính ĐÚNG đại "
                'lượng câu hỏi yêu cầu (theo câu hỏi và dữ liệu), hoặc 0 nếu không chương trình nào đúng.\nJSON: {"choice": <số>, "reason": "<ngắn>"}')
            out["n3a"].append({"id": f"{r['id']}", "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": u}],
                               "n": 3, "temp": 0.7, "max_tokens": 300, "order": [c for c, _ in sorted(g.items(), key=lambda kv: -kv[1]["votes"])[:4]]})
        if "n6a" in modes:
            u = head + ("\n# VIỆC CẦN LÀM\nViết MỘT câu lệnh DuckDB SELECT trên các bảng chuẩn ở trên (tên bảng như liệt kê) trả lời câu hỏi; "
                        "cột kết quả đặt tên theo kiểu đáp án (AS number / AS percent / AS date / AS string / AS list).\n"
                        'JSON: {"sql": "<câu lệnh>"}')
            out["n6a"].append({"id": f"{r['id']}", "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": u}],
                               "n": 3, "temp": 0.7, "max_tokens": 600})
    for m, items in out.items():
        with open(os.path.join(a.out, f"{m}{a.tag}.jsonl"), "w", encoding="utf-8") as w:
            for it in items: w.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(m, len(items), "items")


if __name__ == "__main__":
    main()
