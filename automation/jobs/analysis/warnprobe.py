import sys, glob, pickle, os, time
sys.path.insert(0, "src"); import canon, dm, agent
cache, b = sys.argv[1], sys.argv[2]
biz = pickle.load(open(os.path.join(cache, b + ".pkl"), "rb"))
for p in sorted(glob.glob(f"{sys.argv[3]}/{b}.*.pkl")):
    C = pickle.load(open(p, "rb"))
    if isinstance(C, dict) and "C" in C: C = C["C"]
    t = time.time()
    try: w = canon.secondary_checks(C, biz)
    except Exception as e: print(os.path.basename(p), "ERR", e); continue
    pen = sum(canon.warning_penalty(x) for x in w)
    print(os.path.basename(p), f"pen {pen:.0f} ({time.time()-t:.1f}s)", [x[:70] for x in w if canon.warning_penalty(x) >= 10])
