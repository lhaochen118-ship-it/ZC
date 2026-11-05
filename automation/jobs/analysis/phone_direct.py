import pandas as pd, sys, re
sys.path.insert(0, "src"); import ingest, dm, industry
D = sys.argv[1]
x = pd.read_excel(f"{D}/SoBanHang.xlsx", sheet_name=None, header=None)
gt_name = next(k for k in x if "tắt" in k.lower())
gt = pd.read_excel(f"{D}/SoBanHang.xlsx", sheet_name=gt_name, header=0)
full = {dm.key(a): (b, c) for a, b, c in zip(gt.iloc[:, 0], gt.iloc[:, 1], gt.iloc[:, 2])}
for b, c in zip(gt.iloc[:, 1], gt.iloc[:, 2]): full.setdefault(dm.key(b), (b, c))
frames = []
for k, v in x.items():
    if k == gt_name: continue
    h = next(i for i in range(6) if "STT" in v.iloc[i].astype(str).tolist())
    v = v.iloc[h + 1:].copy(); v.columns = x[k].iloc[h].tolist(); frames.append(v)
d = pd.concat(frames, ignore_index=True)
num = lambda v: ingest.parse_num(v) if isinstance(v, str) else v
def dt(v):
    if isinstance(v, (int, float)) and not pd.isna(v): return pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(v))
    return pd.to_datetime(v, dayfirst=True, errors="coerce")
d["date"] = d["Ngày"].map(dt); d["sl"] = pd.to_numeric(d["SL"].map(num), errors="coerce"); d["tt"] = pd.to_numeric(d["Thành tiền"].map(num), errors="coerce")
d["name"] = d["Hàng"].map(lambda s: full.get(dm.key(s), (s, None))[0]); d["kind"] = d["Hàng"].map(lambda s: full.get(dm.key(s), (None, None))[1])
d = d.drop_duplicates(subset=["STT", "Hàng", "SL", "Giá", "Thành tiền", "IMEI"])
note = d.groupby("STT")["Ghi chú"].apply(lambda s: " ".join(s.dropna().astype(str))); bad = set(note[note.str.contains("trả", case=False)].index)
od = d.groupby("STT")["date"].min(); d["od"] = d["STT"].map(od); v = d[~d.STT.isin(bad)]
def month(y, m): return v[(v.od.dt.year == y) & (v.od.dt.month == m)]
print("unmapped names:", d[d.kind.isna()]["Hàng"].unique()[:10], "kinds:", d.kind.value_counts().to_dict())
exec(sys.argv[2]) if len(sys.argv) > 2 else None
