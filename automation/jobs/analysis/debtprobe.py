import sys, glob, pickle, os
sys.path.insert(0, "src"); import dm, pandas as pd
b, d = sys.argv[1], sys.argv[2]
for p in sorted(glob.glob(f"{sys.argv[3]}/{b}.*.pkl")):
    try:
        C = pickle.load(open(p, "rb"))
        if isinstance(C, dict) and "C" in C: C = C["C"]
        s = dm.Shop(C, today=d)
        tot = s.debt_total(d)
        dc = s.debts(d); n = (dc > 0.5).sum()
        rev = pd.to_numeric(s.O.get("total_payable"), errors="coerce").sum()
        print(os.path.basename(p), "debt", tot, "n_orders", int(n), "rev_all", round(rev), "tables", sorted(k for k in C if isinstance(C[k], pd.DataFrame) and len(C[k])))
    except Exception as e: print(os.path.basename(p), "ERR", type(e).__name__, str(e)[:120])
