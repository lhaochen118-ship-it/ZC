import pickle, sys, json
sys.path.insert(0, "src"); import dm, ingest, industry, pandas as pd
S = sys.argv[1]
biz = pickle.load(open(f"{S}/cache_pub/dev_014.pkl", "rb"))
gt = biz.tables["SoBanHang"]; d = biz.tables["SoBanHang_1"].copy()
for c in ("Ngày", "Khách", "Ghi chú", "Bớt", "Khách trả"): d[c] = d.groupby(d["STT"].ffill())[c].transform("first")
d["STT"] = d["STT"].ffill()
full = {dm.key(a): (b, c) for a, b, c in zip(gt["Ghi tắt"], gt["Tên hàng"], gt["Loại"])}
for b, c in zip(gt["Tên hàng"], gt["Loại"]): full.setdefault(dm.key(b), (b, c))
d["kind"] = d["Hàng"].map(lambda s: industry.device_kind(full.get(dm.key(s), (None, None))[1]))
d["name"] = d["Hàng"].map(lambda s: full.get(dm.key(s), (s, None))[0])
num = lambda v: ingest.parse_num(v) if isinstance(v, str) else v
for c, n in (("SL", "sl"), ("Thành tiền", "tt"), ("Khách trả", "pay"), ("Bớt", "bot")): d[n] = pd.to_numeric(d[c].map(num), errors="coerce")
d["date"] = pd.to_datetime(d["Ngày"], errors="coerce")
d = d.drop_duplicates(subset=[c for c in d.columns if not str(c).startswith("_")])
note = d.groupby("STT")["Ghi chú"].apply(lambda s: " ".join(s.dropna().astype(str))); bad = set(note[note.str.contains("trả", case=False)].index)
v = d[~d.STT.isin(bad)]
def M(y, m, upto=None):
    x = v[(v.date.dt.year == y) & (v.date.dt.month == m)]
    return x[x.date <= upto] if upto is not None else x
def Q(y, q): return v[(v.date.dt.year == y) & ((v.date.dt.month - 1) // 3 == q - 1)]
dev = lambda x: x[x.kind.isin(["máy mới", "máy cũ"])]
ml = lambda x: x["name"].map(industry.model_line)
out = {}
out["devices Aug25"] = dev(M(2025, 8)).sl.sum()
out["used Aug25"] = M(2025, 8)[M(2025, 8).kind == "máy cũ"].sl.sum()
x = dev(Q(2026, 1)); out["top3 lines Q1-26"] = x.groupby(ml(x)).sl.sum().sort_values(ascending=False).head(3).to_dict()
x = dev(M(2026, 1)); out["top3 lines Jan26"] = x.groupby(ml(x)).sl.sum().sort_values(ascending=False).head(3).to_dict()
for (y, m) in ((2026, 9), (2026, 4)):
    x = dev(M(y, m)); out[f"new share rev {y}-{m}"] = round(100 * x[x.kind == "máy mới"].tt.sum() / x.tt.sum(), 2)
x = dev(M(2025, 10)); out["Samsung share qty Oct25"] = round(100 * x[x["name"].map(industry.brand_of) == "Samsung"].sl.sum() / x.sl.sum(), 2)
for (y, m) in ((2025, 12), (2026, 9)):
    x = dev(M(y, m)); out[f"avg device price {y}-{m}"] = round(x.tt.sum() / x.sl.sum())
o = v.groupby("STT").agg(date=("date", "first"), pay=("pay", "first"))
for (y, m) in ((2026, 2), (2025, 12)):
    oo = o[(o.date.dt.year == y) & (o.date.dt.month == m)]; out[f"payable {y}-{m}"] = oo.pay.sum()
for (y, m) in ((2026, 8), (2026, 6), (2025, 8)):
    oo = o[(o.date.dt.year == y) & (o.date.dt.month == m)]; out[f"avg payable {y}-{m}"] = round(oo.pay.sum() / len(oo))
x = dev(M(2026, 2)); x = x[x.kind == "máy mới"]; out["top new item Feb26"] = x.groupby("name").sl.sum().sort_values(ascending=False).head(2).to_dict()
x = dev(M(2026, 1)); out["Redmi Note 14 Jan26"] = x[ml(x) == "Xiaomi Redmi Note 14"].sl.sum()
print(json.dumps({k: (float(v_) if isinstance(v_, (int, float)) else v_) for k, v_ in out.items()}, ensure_ascii=False, indent=0, default=str))
