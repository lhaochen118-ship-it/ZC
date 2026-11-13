#!/usr/bin/env python3
"""CPU tests of plan 3 (docs/16): learned rule cards (src/lrl.py), the # RULES block and <spec> parsing (src/prompting.py),
spec labels from programs (training/data/spec_from_program.py), style augmentation (training/data/dp_style.py), the card writer
with a fake model (src/lrl_gen.py), the per-shop rulebook of mixed evaluation splits (src/hybrid.py rulebook_for)."""
import os, sys, json, random, tempfile, subprocess, datetime as dt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "training", "data"))
import lrl, prompting
from spec_from_program import spec_of, spec_block, add_spec
from dp_style import rewrite_periods, add_noise


def check(cond, msg):
    if not cond: print("FAIL", msg); sys.exit(1)


# 1. cards: matching (accent-free, whole words, per shop), rendering, checks
cards = [{"id": "R1", "kind": "value", "trigger": {"words": ["xi ô đi", "si o di"], "bid": "s1"}, "content": "'xi ô đi' = COD", "code_hint": 'payment="cod"'},
         {"id": "R2", "kind": "definition", "trigger": {"words": ["khách sộp"]}, "content": "khách sộp: giá trị TB / đơn cao nhất",
          "code_hint": 'top_customers(p, 1, by="aov")', "check": "'khach sop' in q and 'aov' not in code"},
         {"id": "R3", "kind": "lexicon", "trigger": {"words": ["dt"]}, "content": "x", "disabled": True}]
lib = lrl.Library(cards)
check(len(lib) == 2, "disabled card dropped")
m = lib.match("Doanh thu các đơn trả XI Ô ĐI tháng này?", bid="s1"); check([c["id"] for c in m] == ["R1"], f"match accent-free {m}")
check(lib.match("đơn trả xi ô đi", bid="s2") == [], "shop card only for its shop")
check(lib.match("xi ô điện") == [], "whole words only")
m = lib.match("Khách sộp nhất tháng 5 là ai?")
check([c["id"] for c in m] == ["R2"], "definition card")
v = lrl.Library.violations(m, "Khách sộp nhất tháng 5 là ai?", 'top_customers(month(2026, 5), 1, by="revenue")')
check(v and v[0].startswith("R2"), f"check flags the program {v}")
check(not lrl.Library.violations(m, "Khách sộp nhất tháng 5 là ai?", 'top_customers(month(2026, 5), 1, by="aov")'), "check passes")
bad = [{"id": "R9", "trigger": {"words": ["a b"]}, "content": "x", "check": "__import__('os').system('echo HACK')"}]
check(lrl.Library.violations(bad, "a b", "code") == [], "no builtins in checks")
blk = lrl.render(m); check(blk.startswith("# RULES") and "by=\"aov\"" in blk, "render")

# 2. prompt: # RULES after the examples, before the question; <spec> parsed and ignored for the code
qb = prompting.question_block([], None, "câu hỏi?", "2026-01-01", "direct", blk)
check(qb.index("# RULES") < qb.index("# QUESTION"), "rules block before the question")
check(prompting.question_block([], None, "câu hỏi?", "2026-01-01") == prompting.question_block([], None, "câu hỏi?", "2026-01-01", "direct", ""),
      "no rules: prompt unchanged")
out = prompting.parse_output('<spec>{"period": "month(2026, 1)"}</spec>\n<plan>revenue | month | -</plan>\n<code>\nANSWER = 1\n</code>')
check(out["code"] == "ANSWER = 1" and out.get("spec", "").startswith('{"period"'), f"spec parsed {out}")
check(prompting.parse_output("<plan>a</plan>\n<code>\nX = 1\n</code>")["code"] == "X = 1", "no spec: unchanged")

# 3. spec labels from programs
sp = spec_of('ANSWER = {"type": "list", "answer": [revenue(month(2026, 3), payment="cod"), revenue(month(2026, 3), payment="ví điện tử")]}')
check(sp == {"period": "month(2026, 3)", "measure": ["revenue"], "params": {"payment": ["cod", "ví điện tử"]}, "answer_type": "list"}, f"spec {sp}")
check(spec_of("not python (") == {}, "unparsable program -> empty spec")
r = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "<plan>x</plan>\n<code>\nA\n</code>"}],
     "program": 'ANSWER = {"type": "number", "answer": n_orders(quarter(2025, 2))}'}
r2, ok = add_spec(r); check(ok and r2["messages"][-1]["content"].startswith('<spec>{"period": "quarter(2025, 2)"'), "add_spec")
check(add_spec(r2)[1] is False, "add_spec idempotent")

# 4. style augmentation: the program's period rewritten in another notation, noise never touches names / numbers
rng = random.Random(3)
q2 = rewrite_periods("doanh thu tháng 4/2025 là bao nhiêu", {"period": "month(2025, 4)"}, dt.date(2026, 1, 5), rng)
check(q2 and q2 != "doanh thu tháng 4/2025 là bao nhiêu" and q2.startswith("doanh thu ") and q2.endswith(" là bao nhiêu"), f"period rewritten {q2}")
check(rewrite_periods("doanh thu hôm qua", {"period": "month(2025, 4)"}, dt.date(2026, 1, 5), rng) is None, "period not in the question -> None")
check(rewrite_periods("so don quy 2/2025", {"period": "quarter(2025, 2)"}, dt.date(2026, 1, 5), rng) is not None, "accent-free question")
tab = {"không": {"ko": 5, "k": 3}, "được": {"đc": 4}}
nq = add_noise("Coca Không 330ml bán được không", tab, random.Random(1), p=1.0)
check("Coca Không 330ml" in nq and nq != "Coca Không 330ml bán được không", f"noise keeps names / numbers {nq}")

# 5. card writer with a fake model (compile mode on a tiny split), validation of hints and checks
with tempfile.TemporaryDirectory() as d:
    sp_dir = os.path.join(d, "split"); os.makedirs(os.path.join(sp_dir, "data"))
    open(os.path.join(sp_dir, "rulebook.md"), "w", encoding="utf-8").write(
        "# Rulebook\n| Luật | Mặc định |\n|---|---|\n| Khách sộp nhất | khách có giá trị trung bình một đơn cao nhất, từ 3 đơn |\n"
        "| Ngày đắt khách nhất | ngày có nhiều đơn nhất |\n")
    fake = os.path.join(d, "fake.py")
    open(fake, "w").write("import json\ndef responder(conv):\n    u = conv[-1]['content']\n"
                          "    if 'sộp' in u: return json.dumps({'triggers': ['khach sop', 'khách xịn'], 'content': 'giá trị TB một đơn cao nhất', 'code_hint': 'top_customers(p, 1, by=\"aov\")'})\n"
                          "    return '<think>x</think>' + json.dumps({'triggers': ['ngày đông khách'], 'content': 'ngày nhiều đơn nhất', 'code_hint': 'no_such_fn(p)'})\n")
    out = os.path.join(d, "out")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "lrl_gen.py"), sp_dir, out, "--mode", "compile", "--fake-responder", fake],
                       capture_output=True, text=True)
    check(r.returncode == 0, "lrl_gen compile: " + r.stdout[-500:] + r.stderr[-1500:])
    cs = [json.loads(l) for l in open(os.path.join(out, "cards_compile.jsonl"), encoding="utf-8")]
    check(len(cs) == 2, f"two cards {cs}")
    a = next(c for c in cs if "sộp" in c["content"]); b = next(c for c in cs if "đắt" in c["content"])
    check(a["code_hint"] == 'top_customers(p, 1, by="aov")' and "Khách sộp nhất" in a["trigger"]["words"], "valid hint kept, term in triggers")
    check(b["code_hint"] is None, "unknown toolkit name dropped")
    lib2 = lrl.Library.load(os.path.join(out, "cards_compile.jsonl"))
    check([c["id"] for c in lib2.match("khach sop nhat thang nay")] == [a["id"]], "compiled card matches an accent-free question")

# 6. evaluation splits: a held-out training shop keeps the training rulebook (docs/15 §4.3)
import argparse, hybrid
class _A: split = ""
with tempfile.TemporaryDirectory() as d:
    os.makedirs(os.path.join(d, "rulebooks"))
    open(os.path.join(d, "rulebooks", "biz_105.md"), "w").write("TRAIN RULEBOOK")
    h = hybrid.HybridPipeline.__new__(hybrid.HybridPipeline); h.a = argparse.Namespace(split=d); h.rulebook = "PUBLIC RULEBOOK"
    check(h.rulebook_for("biz_105") == "TRAIN RULEBOOK" and h.rulebook_for("dev_001") == "PUBLIC RULEBOOK", "rulebook_for")
print("test_lrl PASS")
