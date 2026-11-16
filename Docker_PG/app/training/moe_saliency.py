#!/usr/bin/env python3
import os, sys, json, glob, time, argparse, hashlib
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)


def find_model(name):
    c = sorted(p for p in glob.glob("/kaggle/input/**/config.json", recursive=True) if name.lower() in p.lower()
               and glob.glob(os.path.join(os.path.dirname(p), "*.safetensors")))
    if not c: sys.exit(f"[sal] no mounted model matching {name}")
    return os.path.dirname(c[0])


def check_hashes(d, hashes):
    def one(f):
        h = hashlib.sha256()
        with open(os.path.join(d, f), "rb") as fh:
            for ch in iter(lambda: fh.read(64 << 20), b""): h.update(ch)
        return f, h.hexdigest() == hashes[f]["sha256"]
    t = time.time()
    with ThreadPoolExecutor(8) as ex: res = dict(ex.map(one, [f for f in hashes if f.endswith(".safetensors")]))
    bad = [f for f, ok in res.items() if not ok]
    print(f"[sal] sha256 of {len(res)} weight files checked in {time.time() - t:.0f}s, mismatches: {bad}", flush=True)
    if bad: sys.exit(3)


def resolve(p):
    for c in (p, os.path.join(ROOT, p), os.path.join(ROOT, "training", "data", "out", p), os.path.join(HERE, "data", "out", p)):
        if os.path.isfile(c): return c
    hits = glob.glob(f"/kaggle/input/**/{os.path.basename(p)}", recursive=True)
    if hits: return hits[0]
    sys.exit(f"[sal] {p} not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out")
    ap.add_argument("--src", default="qwen3-6-35b-a3b"); ap.add_argument("--prompts", required=True)
    ap.add_argument("--model-dir", default="", help="tests: a local model directory instead of the Kaggle mount")
    ap.add_argument("--hashes", default=""); ap.add_argument("--max-prompts", type=int, default=400)
    ap.add_argument("--max-tokens", type=int, default=6144); ap.add_argument("--budgets", default="64,77,88,90,92")
    ap.add_argument("--time-limit", type=float, default=4200, help="seconds of forward passes at most (partial statistics are kept)")
    a = ap.parse_args()
    t0 = time.time(); os.makedirs(a.out, exist_ok=True)
    import torch
    import transformers
    from transformers import AutoTokenizer
    d = a.model_dir or find_model(a.src); print(f"[sal] model {d} transformers {transformers.__version__} torch {torch.__version__}", flush=True)
    if a.hashes: check_hashes(d, json.load(open(resolve(a.hashes)))["files"])
    cfg = json.load(open(os.path.join(d, "config.json"))); tc = cfg.get("text_config", cfg)
    E, K = int(tc["num_experts"]), int(tc["num_experts_per_tok"]); norm = bool(tc.get("norm_topk_prob", True))
    tok = AutoTokenizer.from_pretrained(d)
    ng = torch.cuda.device_count()
    mm = {i: ("19GiB" if i == 0 else "21GiB") for i in range(ng)} or None
    model = None
    for cls in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        try:
            model = getattr(transformers, cls).from_pretrained(d, dtype=torch.bfloat16, device_map="auto", max_memory=mm)
            print(f"[sal] loaded with {cls} in {time.time() - t0:.0f}s", flush=True); break
        except Exception as e: print(f"[sal] {cls}: {type(e).__name__}: {str(e)[:300]}", flush=True)
    if model is None: sys.exit("[sal] model not loadable")
    model.eval()
    import re
    routers = []
    for name, m in model.named_modules():
        mt = re.search(r"layers\.(\d+)\.mlp\.gate$", name)
        if mt and "mtp" not in name and "visual" not in name and getattr(m, "weight", None) is not None and m.weight.shape[0] == E:
            routers.append((int(mt.group(1)), name, m))
    print(f"[sal] {len(routers)} routers found (E={E}, top-{K}, norm_topk {norm})", flush=True)
    if not routers: sys.exit("[sal] no router found")
    mods = dict(model.named_modules())
    stats = {L: {"count": torch.zeros(E, dtype=torch.float64), "gate": torch.zeros(E, dtype=torch.float64),
                 "reap": torch.zeros(E, dtype=torch.float64)} for L, _, _ in routers}
    reap_ok = {"yes": 0, "no": 0}

    def expert_fn(L, name):
        ex = mods.get(name[:-len("gate")] + "experts")
        if ex is None: return None
        if isinstance(ex, torch.nn.ModuleList):
            return lambda j, x: ex[j](x)
        gu = getattr(ex, "gate_up_proj", None); dn = getattr(ex, "down_proj", None)
        if isinstance(gu, torch.Tensor) and isinstance(dn, torch.Tensor) and gu.dim() == 3 and dn.dim() == 3:
            H = x_dim = None
            def f(j, x):
                g, u = torch.nn.functional.linear(x, gu[j]).chunk(2, dim=-1)
                return torch.nn.functional.linear(torch.nn.functional.silu(g) * u, dn[j])
            return f
        return None

    def make_hook(L, name):
        fexp = [None, False]
        def hook(mod, inp, out):
            x = inp[0].detach(); x = x.reshape(-1, x.shape[-1])
            logits = (x.float() @ mod.weight.detach().float().t())
            if getattr(mod, "bias", None) is not None: logits = logits + mod.bias.detach().float()
            p = torch.softmax(logits, dim=-1); w, idx = torch.topk(p, K, dim=-1)
            if norm: w = w / w.sum(-1, keepdim=True)
            s = stats[L]; fi = idx.reshape(-1).cpu(); fw = w.reshape(-1).double().cpu()
            s["count"].index_add_(0, fi, torch.ones_like(fw)); s["gate"].index_add_(0, fi, fw)
            if not fexp[1]: fexp[0] = expert_fn(L, name); fexp[1] = True
            if fexp[0] is None: reap_ok["no"] += 1; return
            try:
                with torch.no_grad():
                    for j in torch.unique(idx).tolist():
                        rows, slot = (idx == j).nonzero(as_tuple=True)
                        y = fexp[0](j, x[rows].to(x.dtype))
                        nr = y.float().norm(dim=-1) * w[rows, slot]
                        s["reap"][j] += float(nr.sum())
                reap_ok["yes"] += 1
            except Exception as e:
                if reap_ok["no"] < 3: print(f"[sal] layer {L}: expert recompute failed ({type(e).__name__}: {e}); gate fallback", flush=True)
                reap_ok["no"] += 1; fexp[0] = None
        return hook

    for L, name, m in routers: m.register_forward_hook(make_hook(L, name))
    rows = [json.loads(l) for l in open(resolve(a.prompts), encoding="utf-8") if l.strip()][:a.max_prompts]
    ntok, t1 = 0, time.time()
    dev = next(model.parameters()).device
    for i, r in enumerate(rows):
        if time.time() - t1 > a.time_limit: print(f"[sal] time limit after {i} prompts", flush=True); break
        try: text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", truncation=True, max_length=a.max_tokens)["input_ids"].to(dev)
        with torch.no_grad(): model(input_ids=ids, use_cache=False)
        ntok += ids.shape[1]
        if i % 25 == 0: print(f"[sal] {i + 1}/{len(rows)} prompts, {ntok} tokens, {ntok / max(1, time.time() - t1):.0f} tok/s", flush=True)
    print(f"[sal] {ntok} tokens in {time.time() - t1:.0f}s; expert recompute ok {reap_ok['yes']} / fallback {reap_ok['no']}", flush=True)
    out = {"model": d, "E": E, "K": K, "tokens": ntok, "prompts": len(rows), "reap_ok": reap_ok, "layers": {}}
    keeps = {int(b): {} for b in a.budgets.split(",")}
    for L in sorted(stats):
        s = stats[L]; c = s["count"].clamp(min=1)
        sal = (s["reap"] / c) if reap_ok["yes"] else (s["gate"] / c) * (s["count"] / s["count"].sum())
        sal = torch.where(s["count"] > 0, sal, torch.full_like(sal, -1.0))
        out["layers"][str(L)] = {"count": s["count"].tolist(), "gate": s["gate"].tolist(), "reap": s["reap"].tolist(), "saliency": sal.tolist()}
        order = torch.argsort(sal, descending=True).tolist()
        for b in keeps: keeps[b][str(L)] = sorted(order[:b])
    json.dump(out, open(os.path.join(a.out, "saliency.json"), "w"))
    ref = None
    try: ref = json.load(open(resolve("configs/moe/keep_reap64.json")))["keep"]
    except SystemExit: pass
    for b, kp in keeps.items():
        json.dump({"source": "moe_saliency (task prompts, training shops)", "budget": b, "keep": kp}, open(os.path.join(a.out, f"keep_e{b}.json"), "w"))
        if ref: print(f"[sal] E={b}: overlap with REAP-64 keep-list {sum(len(set(kp[L]) & set(ref[L])) for L in kp) / sum(len(ref[L]) for L in ref):.3f}", flush=True)
    print(f"[sal] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
