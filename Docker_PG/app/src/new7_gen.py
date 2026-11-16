#!/usr/bin/env python3
import os, sys, json, glob, argparse, time

HERE = os.path.dirname(os.path.abspath(__file__))


def resolve(p):
    for c in (p, os.path.join(HERE, "training", "data", "out", p), os.path.join(os.path.dirname(HERE), "training", "data", "out", p)):
        if os.path.isfile(c): return c
    hits = glob.glob(f"/kaggle/input/**/{os.path.basename(p)}", recursive=True)
    if hits: return hits[0]
    sys.exit(f"[new7_gen] {p} not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out"); ap.add_argument("--model", required=True); ap.add_argument("--items", required=True)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--quant", default="fp8"); ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--fake-responder", default="")
    a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out, exist_ok=True)
    i, n = map(int, a.shard.split("/"))
    items = [json.loads(l) for l in open(resolve(a.items), encoding="utf-8") if l.strip()][i::n]
    convs, temps, seeds, owner, think = [], [], [], [], None
    for it in items:
        for k in range(int(it.get("n", 1))):
            convs.append(it["messages"]); temps.append(0.0 if k == 0 else float(it.get("temp", 0.7))); seeds.append(1000 + k); owner.append(it["id"])
    mt = max([int(it.get("max_tokens", 600)) for it in items] or [600]); think = any(it.get("think") for it in items)
    if a.fake_responder:
        g = {}; exec(compile(open(a.fake_responder, encoding="utf-8").read(), a.fake_responder, "exec"), g)
        texts = [g["responder"](c) for c in convs]
    elif convs:
        sys.argv = [sys.argv[0]]; import agent
        eng = agent.Engine(argparse.Namespace(model=a.model, tp=1, max_len=a.max_len, think=think, eager=False, lora="", quant=a.quant,
                                              gpu_util=0.92, max_num_seqs=128, max_batched_tokens=16384, spec_mtp=0, prefix_budget=0))
        texts = eng.gen(convs, temps, seeds, mt, think=think)[0]
    else: texts = []
    by = {}
    for o, t in zip(owner, texts): by.setdefault(o, []).append(t)
    with open(os.path.join(a.out, f"gen_{i}of{n}.jsonl"), "w", encoding="utf-8") as w:
        for k, v in by.items(): w.write(json.dumps({"id": k, "texts": v}, ensure_ascii=False) + "\n")
    print(f"NEW7_GEN {json.dumps({'items': len(items), 'requests': len(convs), 's': round(time.time() - t0)})}", flush=True)


if __name__ == "__main__":
    main()
