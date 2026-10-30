#!/usr/bin/env python3
"""Local smoke test of the agent pipeline without a GPU: a FakeEngine answers with hand-written programs.
It checks the plumbing only (compile -> checks -> signature -> two paths -> repair / self-check -> vote -> score).
Usage: test_agent_local.py <train_split_dir> <scratch_dir>"""
import os, sys, json, shutil, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import agent

split, scratch = sys.argv[1], sys.argv[2]
mini = os.path.join(scratch, "mini"); shutil.rmtree(mini, ignore_errors=True); os.makedirs(os.path.join(mini, "data"))
for f in ("rulebook.md", "calendar.csv"): shutil.copy(os.path.join(split, f), mini)
os.symlink(os.path.join(split, "data", "biz_101"), os.path.join(mini, "data", "biz_101"))
want = ["q_0cb7c9b93048", "q_1cc70c12aa27", "q_20aa992c0d3f"]
for f in ("questions.jsonl", "answers.jsonl"):
    rows = [l for l in open(os.path.join(split, f), encoding="utf-8") if json.loads(l)["id"] in want]
    open(os.path.join(mini, f), "w", encoding="utf-8").writelines(rows)

COMPILE = '''```python
df = T['ban_hang'].copy()
smap = {'Xong': 'hoàn thành', 'Đã hủy': 'huỷ', 'Đổi trả': 'trả hàng', 'Đang giao': 'đang giao'}
df['status'] = df['Kết quả'].map(smap)
st = df.groupby('Mã hóa đơn')['status'].agg(lambda s: 'trả hàng' if (s == 'trả hàng').any() else ('huỷ' if (s == 'huỷ').any() else s.iloc[0]))
lines = pd.DataFrame({'order_id': df['Mã hóa đơn'], 'order_date': df['Ngày tháng'], 'item_name': df['Nội dung'], 'qty': df['so_luong'],
                      'qty_base': df['Quy ra ĐV lẻ'], 'line_amount': df['Thanh tien']})
lines['status'] = lines['order_id'].map(st); lines['valid'] = ~lines['status'].isin(['huỷ', 'trả hàng'])
orders = lines.groupby('order_id').agg(order_date=('order_date', 'first'), revenue=('line_amount', 'sum')).reset_index()
orders['status'] = orders['order_id'].map(st); orders['valid'] = ~orders['status'].isin(['huỷ', 'trả hàng'])
C = {'orders': orders, 'lines': lines}
NOTES = 'one csv of order lines; no cost price'
print(len(orders))
```'''
BAD_COMPILE = COMPILE.replace("'Xong': 'hoàn thành', ", "")                       # leaves a non-canonical (NaN) status -> repaired

PROG_A = {  # canonical path
 "q_0cb7c9b93048": "o = C['orders']; d = TODAY - pd.Timedelta(days=1)\nANSWER = {'type': 'number', 'answer': int(((o.order_date == d) & o.valid).sum())}",
 "q_1cc70c12aa27": "o = C['orders']\nANSWER = {'type': 'number', 'answer': o.loc[(o.order_date == TODAY) & o.valid, 'revenue'].sum()}",
 "q_20aa992c0d3f": "o = C['orders']; s, e = month_range(2026, 5); x = o[(o.order_date >= s) & (o.order_date <= e) & o.valid]\n"
                   "ANSWER = {'type': 'date', 'answer': x.groupby('order_date').revenue.sum().idxmax()}"}
PROG_B = {  # raw path (first answer of q_1cc7 is a wrong 0 -> self-check round fixes it)
 "q_0cb7c9b93048": "df = T['ban_hang']; bad = df.groupby('Mã hóa đơn')['Kết quả'].agg(lambda s: s.isin(['Đã hủy','Đổi trả']).any())\n"
                   "ok = df[(df['Ngày tháng'] == TODAY - pd.Timedelta(days=1)) & ~df['Mã hóa đơn'].map(bad)]\nANSWER = {'type': 'number', 'answer': ok['Mã hóa đơn'].nunique()}",
 "q_1cc70c12aa27": "df = T['ban_hang']; bad = df.groupby('Mã hóa đơn')['Kết quả'].agg(lambda s: s.isin(['Đã hủy','Đổi trả']).any())\n"
                   "ok = df[(df['Ngày tháng'] == TODAY) & ~df['Mã hóa đơn'].map(bad)]\nANSWER = {'type': 'number', 'answer': ok['Thanh tien'].sum()}",
 "q_20aa992c0d3f": "raise KeyError('typo_column')"}                     # fails -> repair round -> fixed program


QID = {json.loads(l)["question"]: json.loads(l)["id"] for l in open(os.path.join(mini, "questions.jsonl"), encoding="utf-8")}
FIXED_20AA = ("df = T['ban_hang']; bad = df.groupby('Mã hóa đơn')['Kết quả'].agg(lambda s: s.isin(['Đã hủy','Đổi trả']).any())\n"
              "s, e = month_range(2026, 5); x = df[(df['Ngày tháng'] >= s) & (df['Ngày tháng'] <= e) & ~df['Mã hóa đơn'].map(bad)]\n"
              "ANSWER = {'type': 'date', 'answer': x.groupby('Ngày tháng')['Thanh tien'].sum().idxmax()}")


def responder(conv):
    sysmsg, user, turns = conv[0]["content"], conv[1]["content"], len(conv)
    if sysmsg.startswith("You are DataMate's data engineer"):
        return BAD_COMPILE if turns == 2 else COMPILE                  # first try leaves 'Xong' unmapped -> check fails -> repaired
    qid = next(i for q, i in QID.items() if q in user)
    if "CANONICAL TABLES" in user: return "```python\n" + PROG_A[qid] + "\n```"
    if qid == "q_1cc70c12aa27" and turns == 2: return "```python\nANSWER = {'type': 'number', 'answer': 0}\n```"   # -> self-check round
    if qid == "q_20aa992c0d3f" and turns > 2: return "```python\n" + FIXED_20AA + "\n```"
    return "```python\n" + PROG_B[qid] + "\n```"


args = argparse.Namespace(split=mini, out=os.path.join(scratch, "out"), model="fake", k=4, paths="", compile=True, kc=2, repairs=2, subset=0,
                          tau=0.5, cache=os.path.join(scratch, "cache"), tp=0, max_len=0, think=False, workers=2, max_tokens=0,
                          compile_tokens=0, timeout=60, temp=0.7, eager=False, no_reflect=False, lora="", dp=1, shard="", chunk=0, budget=0, log_dir="")
if "--budget-test" in sys.argv:                                      # out of time before the first chunk -> every answer abstains
    args.budget = 1e-6; args.out = os.path.join(scratch, "out_budget")
    agent.run(args, engine=agent.FakeEngine(responder))
    rows = [json.loads(l) for l in open(os.path.join(args.out, "submission.jsonl"), encoding="utf-8")]
    ok = len(rows) == len(want) and all(r["type"] == "abstain" for r in rows)
    print("TEST", "PASS" if ok else "FAIL", "budget fallback:", [r["type"] for r in rows]); sys.exit(0)
agent.run(args, engine=agent.FakeEngine(responder))
res = json.load(open(os.path.join(args.out, "score.json")))
clog = [json.loads(l) for l in open(os.path.join(args.out, "compile_logs.jsonl"), encoding="utf-8")]
first = clog[0]["attempts"]["0"] if "0" in clog[0]["attempts"] else clog[0]["attempts"][0]
caught = (not first[0]["ok"]) and "not mapped" in first[0]["error"]
print("TEST", "PASS" if res["PA"] == 1.0 and caught else "FAIL", res["PA"], "| bad compile caught:", caught)
