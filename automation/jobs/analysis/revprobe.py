import sys, glob, pickle, os
sys.path.insert(0, "src"); import dm, pandas as pd
b = sys.argv[1]
for p in sorted(glob.glob(f"{sys.argv[2]}/{b}.*.pkl")):
    C = pickle.load(open(p, "rb")); C = C.get("C", C)
    try:
        s = dm.Shop(C, today="2026-09-30")
        print(os.path.basename(p), [s.revenue(dm.month(2026, m) if m < 13 else None) for m in (1, 7, 8, 9)], "n_orders jan", s.n_orders(dm.month(2026, 1)), "status", C["orders"]["status"].value_counts().to_dict())
    except Exception as e: print(os.path.basename(p), "ERR", e)
