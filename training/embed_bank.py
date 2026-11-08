#!/usr/bin/env python3
"""Embeddings of the example bank's skeletons (docs/08 §5.10 S1, G23): src/refs/bank/examples.jsonl -> embeddings.npy (row i =
bank line i, L2-normalised float16). CPU or GPU. Usage: embed_bank.py <embedding_model_dir> [--bank F] [--out F]"""
import os, sys, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src") if os.path.isdir(os.path.join(ROOT, "src")) else ROOT)


def main():
    import numpy as np, embedder, retrieval
    m = sys.argv[1]
    bank = sys.argv[sys.argv.index("--bank") + 1] if "--bank" in sys.argv else retrieval.EX_PATH
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(os.path.dirname(bank), "embeddings.npy")
    rows = [json.loads(l) for l in open(bank, encoding="utf-8") if l.strip()]
    t0 = time.time(); E = embedder.Embedder(m)
    V = E.encode([r.get("skeleton") or retrieval.skeleton(r["question"]) for r in rows], query=False)
    np.save(out, V.astype(np.float16))
    print(f"embedded {len(rows)} bank rows -> {out} {V.shape} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
