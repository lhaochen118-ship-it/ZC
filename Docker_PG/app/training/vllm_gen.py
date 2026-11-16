#!/usr/bin/env python3
import os, sys, json, time, argparse
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def sampling(SP, temp, max_tokens, seed, n, grammar=None, think=False):
    kw = dict(temperature=temp, top_p=1.0 if temp == 0 else 0.95, max_tokens=max_tokens, seed=seed, n=n)
    if think: kw.update(temperature=temp if temp > 0 else 0.6, top_p=0.95, top_k=20)
    if grammar:
        try:
            from vllm.sampling_params import StructuredOutputsParams
            kw["structured_outputs"] = StructuredOutputsParams(grammar=grammar)
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams
            kw["guided_decoding"] = GuidedDecodingParams(grammar=grammar)
    return SP(**kw)


def render(tok, msgs, think):
    try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=bool(think))
    except TypeError: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--quant", default=""); ap.add_argument("--tp", type=int, default=1); ap.add_argument("--lora", default="")
    ap.add_argument("--grammar-dir", default=""); ap.add_argument("--max-tokens", type=int, default=256); ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=1); ap.add_argument("--max-model-len", type=int, default=4096); ap.add_argument("--gpu-util", type=float, default=0.92)
    ap.add_argument("--max-num-seqs", type=int, default=128); ap.add_argument("--think", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kv-dtype", default="auto"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--max-lora-rank", type=int, default=64)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    recs = [json.loads(l) for l in open(a.prompts, encoding="utf-8") if l.strip()]
    if a.limit: recs = recs[:a.limit]
    t0 = time.time()
    kw = dict(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util,
              enable_prefix_caching=True, seed=a.seed, max_num_seqs=a.max_num_seqs)
    cfgp = os.path.join(a.model, "config.json")
    if os.path.isfile(cfgp) and "vision_config" in json.load(open(cfgp)): kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    if a.quant: kw["quantization"] = a.quant
    if a.kv_dtype != "auto": kw["kv_cache_dtype"] = a.kv_dtype
    if a.lora: kw.update(enable_lora=True, max_lora_rank=a.max_lora_rank, max_loras=max(1, len(a.lora.split(","))))
    llm = LLM(**kw); t_load = time.time() - t0
    kv = None
    try: c = llm.llm_engine.vllm_config.cache_config; kv = int(c.num_gpu_blocks) * int(c.block_size)
    except Exception: pass
    tok = llm.get_tokenizer()
    lora = None
    if a.lora:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest("a0", 1, a.lora.split(",")[0])
    G = {}
    def gram(r):
        g = r.get("grammar")
        if not g or not a.grammar_dir: return None
        if g not in G: G[g] = open(os.path.join(a.grammar_dir, g), encoding="utf-8").read()
        return G[g]
    prompts = [render(tok, r["messages"], a.think) for r in recs]
    sps = [sampling(SamplingParams, a.temp, a.max_tokens, a.seed + i, a.n, gram(r), a.think) for i, r in enumerate(recs)]
    t1 = time.time()
    outs = llm.generate(prompts, sps, use_tqdm=False, lora_request=lora)
    t_gen = time.time() - t1
    ntok = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for r, o in zip(recs, outs):
            for j, c in enumerate(o.outputs):
                ntok += len(c.token_ids)
                f.write(json.dumps({"id": r.get("id"), "j": j, "text": c.text, "n_tokens": len(c.token_ids), "finish": c.finish_reason}, ensure_ascii=False) + "\n")
    s = {"model": os.path.basename(a.model.rstrip("/")), "n": len(recs), "load_s": round(t_load, 1), "gen_s": round(t_gen, 1), "out_tokens": ntok,
         "tok_s": round(ntok / max(t_gen, 1e-6), 1), "req_s": round(len(recs) / max(t_gen, 1e-6), 2), "kv_tokens": kv, "tp": a.tp,
         "grammar": bool(a.grammar_dir), "lora": bool(a.lora), "quant": a.quant}
    print("SUMMARY " + json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
