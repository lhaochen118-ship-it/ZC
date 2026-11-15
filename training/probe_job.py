#!/usr/bin/env python3
"""P5 plan H01 / H02: hidden-state probe of the base Qwen3.5-9B as a candidate verifier (Orgad et al., ICLR 2025).

Items (prepared offline by Process/tools/p5/probe_items.py from the solver logs of the zero-shot runs): one per distinct
(question, program, answer) candidate, text = question + program + executed answer; training items carry a label (the answer is
correct on the TRAINING answers) and the shop id (group); public items carry no label.
1. features: the base model (bf16, transformers) reads each text once; the hidden state of the last token at layers
   {L/2, 3L/4, L} is kept (one worker process per GPU, items sharded);
2. probes fitted on TRAINING items only, GroupKFold by shop (5 folds): H01 logistic regression (C chosen by fold AUC),
   H02 small MLP (1 hidden layer, early stopping); out-of-fold probabilities for the training items (to set the selection policy
   on training data) and probabilities of the full-fit probe for the public items.
Output: <out>/probe_probs.jsonl {"uid", "split", "p_lr", "p_mlp"} + <out>/probe_metrics.json (fold AUC, chosen C).

Usage (Kaggle 4xL4: run_kaggle.py ... --train --script training/probe_job.py):
    probe_job.py <split> <out> --items <file under training/data/out> [--model-dir auto] [--layers auto] [--max-tokens 1024]"""
import os, sys, json, glob, time, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)


def resolve(p):
    for c in (p, os.path.join(ROOT, p), os.path.join(ROOT, "training", "data", "out", p), os.path.join(HERE, "data", "out", p)):
        if os.path.isfile(c): return c
    hits = glob.glob(f"/kaggle/input/**/{os.path.basename(p)}", recursive=True)
    if hits: return hits[0]
    sys.exit(f"[probe] {p} not found")


def find_model(want):
    if os.path.isdir(want): return want
    for pat in (f"/tmp/models/{want}/config.json", f"/kaggle/input/**/models/{want}/config.json"):
        c = glob.glob(pat, recursive=True)
        if c: return os.path.dirname(c[0])
    hj = json.load(open(os.path.join(ROOT, "model_hashes.json"))) if os.path.isfile(os.path.join(ROOT, "model_hashes.json")) else {"files": {}}
    for c in glob.glob("/kaggle/input/**/config.json", recursive=True):   # the Kaggle Model holding the official files
        if hj["files"] and all(os.path.isfile(os.path.join(os.path.dirname(c), f)) for f in hj["files"]): return os.path.dirname(c)
    sys.exit(f"[probe] model {want} not found")


def worker(a, gid, n):
    import torch, numpy as np, transformers
    from transformers import AutoTokenizer
    items = [json.loads(l) for l in open(resolve(a.items), encoding="utf-8") if l.strip()][gid::n]
    d = find_model(a.model_dir); tok = AutoTokenizer.from_pretrained(d)
    model = None
    for cls in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        try: model = getattr(transformers, cls).from_pretrained(d, dtype=torch.bfloat16, device_map={"": 0}); break
        except Exception as e: print(f"[probe {gid}] {cls}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    model.eval()
    feats, t0 = [], time.time()
    order = sorted(range(len(items)), key=lambda i: len(items[i]["text"]))  # length-sorted batches: little padding
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    out = [None] * len(items)
    bs = a.batch
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        texts = [tok.apply_chat_template([{"role": "user", "content": items[i]["text"]}], tokenize=False, add_generation_prompt=False) for i in idx]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=a.max_tokens).to("cuda")
        with torch.no_grad(): o = model(**enc, output_hidden_states=True, use_cache=False)
        hs = o.hidden_states; L = len(hs) - 1
        lay = [int(x) for x in a.layers.split(",")] if a.layers != "auto" else [L // 2, (3 * L) // 4, L]
        v = torch.cat([hs[l][:, -1, :].float() for l in lay], dim=-1).cpu().numpy().astype(np.float16)   # left padding: last = real
        for j, i in enumerate(idx): out[i] = v[j]
        if (s // bs) % 50 == 0: print(f"[probe {gid}] {s + len(idx)}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    np.save(os.path.join(a.out, f"feat_{gid}.npy"), np.stack(out))
    json.dump([it["uid"] for it in items], open(os.path.join(a.out, f"uid_{gid}.json"), "w"))
    print(f"[probe {gid}] {len(items)} items in {time.time() - t0:.0f}s, layers {lay}", flush=True)


def fit(a):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    items = {it["uid"]: it for it in (json.loads(l) for l in open(resolve(a.items), encoding="utf-8") if l.strip())}
    X, U = [], []
    for g in range(a.n):
        X.append(np.load(os.path.join(a.out, f"feat_{g}.npy")).astype(np.float32)); U += json.load(open(os.path.join(a.out, f"uid_{g}.json")))
    X = np.concatenate(X)
    tr = [i for i, u in enumerate(U) if items[u]["split"] == "train"]; pu = [i for i, u in enumerate(U) if items[u]["split"] != "train"]
    y = np.array([int(items[U[i]]["label"]) for i in tr]); grp = np.array([items[U[i]]["group"] for i in tr])
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
    pca = PCA(n_components=min(256, len(tr) - 1), random_state=0).fit(Xs[tr]); Z = pca.transform(Xs)
    gkf = GroupKFold(n_splits=5); folds = list(gkf.split(Z[tr], y, grp))
    res = {"n_train": len(tr), "n_public": len(pu), "pos_rate": float(y.mean()), "dims": int(X.shape[1])}
    best = None
    for C in (0.01, 0.1, 1.0):
        oof = np.zeros(len(tr))
        for f_tr, f_va in folds:
            m = LogisticRegression(C=C, max_iter=2000).fit(Z[tr][f_tr], y[f_tr]); oof[f_va] = m.predict_proba(Z[tr][f_va])[:, 1]
        auc = roc_auc_score(y, oof); res[f"lr_auc_C{C}"] = round(float(auc), 4)
        if best is None or auc > best[0]: best = (auc, C, oof)
    res["lr_C"] = best[1]; res["lr_auc"] = round(float(best[0]), 4); oof_lr = best[2]
    lr = LogisticRegression(C=best[1], max_iter=2000).fit(Z[tr], y); p_lr = lr.predict_proba(Z[pu])[:, 1] if pu else []
    oof_mlp = np.zeros(len(tr))
    for f_tr, f_va in folds:
        m = MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3, early_stopping=True, max_iter=300, random_state=0).fit(Z[tr][f_tr], y[f_tr])
        oof_mlp[f_va] = m.predict_proba(Z[tr][f_va])[:, 1]
    res["mlp_auc"] = round(float(roc_auc_score(y, oof_mlp)), 4)
    mlp = MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3, early_stopping=True, max_iter=300, random_state=0).fit(Z[tr], y)
    p_mlp = mlp.predict_proba(Z[pu])[:, 1] if pu else []
    with open(os.path.join(a.out, "probe_probs.jsonl"), "w") as w:
        for k, i in enumerate(tr): w.write(json.dumps({"uid": U[i], "split": "train", "p_lr": float(oof_lr[k]), "p_mlp": float(oof_mlp[k])}) + "\n")
        for k, i in enumerate(pu): w.write(json.dumps({"uid": U[i], "split": items[U[i]]["split"], "p_lr": float(p_lr[k]), "p_mlp": float(p_mlp[k])}) + "\n")
    json.dump(res, open(os.path.join(a.out, "probe_metrics.json"), "w"), indent=1); print("[probe] fit", json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out"); ap.add_argument("--items", required=True)
    ap.add_argument("--model-dir", "--model", dest="model_dir", default="Qwen3.5-9B"); ap.add_argument("--layers", default="auto")
    ap.add_argument("--max-tokens", type=int, default=768); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--worker", type=int, default=-1); ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.worker >= 0: return worker(a, a.worker, a.n)
    import torch
    a.n = a.n or max(1, torch.cuda.device_count()); t0 = time.time()
    vis = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x] or [str(i) for i in range(a.n)]
    ps = [subprocess.Popen([sys.executable, os.path.abspath(__file__)] + sys.argv[1:] + ["--worker", str(g), "--n", str(a.n)],
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=vis[g % len(vis)])) for g in range(a.n)]
    rcs = [p.wait() for p in ps]; print(f"[probe] features rc {rcs} in {time.time() - t0:.0f}s", flush=True)
    if any(rcs): sys.exit(4)
    fit(a); print(f"[probe] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
