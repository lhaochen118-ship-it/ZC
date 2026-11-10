#!/usr/bin/env python3
"""M4 (docs/12_PLAN2.md §6; ExCoT-style DPO with execution feedback): LoRA DPO of the main model on pairs (a correct answer,
a wrong answer of the model itself) built by training/data/dpo_from_logs.py from training-split logs only. TRL DPOTrainer +
PEFT, bf16, the reference = the same model with the LoRA disabled (no second copy), prompt rendered with the model's chat
template exactly as at inference (thinking off), optional FSDP through `accelerate launch --config_file training/fsdp4.yaml`
(one 9B LoRA does not fit one L4 with the DPO pair: docs/08 WP1, 22 GB for SFT alone).

Records (JSONL): {"messages": [system, user], "chosen": text, "rejected": text, ...}.
Usage: [accelerate launch --config_file training/fsdp4.yaml] training/dpo.py --model DIR --data pairs.jsonl[,b.jsonl] --out DIR
       [--beta 0.1 --lr 5e-6 --epochs 1 --r 32 --alpha 64 --max-len 4608 --mb 1 --eff-batch 32 --max-steps N --sft-weight 0]
       [--metrics out.json] [--no-save]"""
import os, sys, json, time, math, argparse, random
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sft import LM_TARGET, TorchLogProb


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--r", type=int, default=32); ap.add_argument("--alpha", type=int, default=64); ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=5e-6); ap.add_argument("--beta", type=float, default=0.1); ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=4608); ap.add_argument("--mb", type=int, default=1); ap.add_argument("--eff-batch", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=-1); ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--sft-weight", type=float, default=0.0, help="> 0: loss = DPO (sigmoid) + w x NLL of the chosen answer (RPO)")
    ap.add_argument("--metrics", default=""); ap.add_argument("--no-save", action="store_true"); ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--arch", default="causal", choices=["causal", "vl"]); ap.add_argument("--no-gc", action="store_true")
    return ap.parse_args()


def records(paths, tok, max_len, seed=0):
    rows = [json.loads(l) for p in paths.split(",") for l in open(p, encoding="utf-8") if l.strip()]
    out, skipped = [], 0
    for r in rows:
        prompt = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ch, rj = r["chosen"] + "<|im_end|>", r["rejected"] + "<|im_end|>"
        n = len(tok(prompt + (ch if len(ch) >= len(rj) else rj), add_special_tokens=False)["input_ids"])
        if n > max_len: skipped += 1; continue
        out.append({"prompt": prompt, "chosen": ch, "rejected": rj, "n": n})
    random.Random(seed).shuffle(out)
    return out, skipped


def main():
    a = parse()
    import torch
    if os.environ.get("ZC_TORCH_LOGPROB") == "1":
        import trl.trainer.utils as _tu; _tu._ChunkedLogProbFunction = TorchLogProb
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
    from peft import LoraConfig
    try:
        import peft.import_utils as _piu, peft.tuners.lora.torchao as _pto
        _piu.is_torchao_available = lambda: False; _pto.is_torchao_available = lambda: False
    except Exception: pass
    from trl import DPOTrainer, DPOConfig
    rank = int(os.environ.get("RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1")); local = int(os.environ.get("LOCAL_RANK", "0"))
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token = "<|endoftext|>"
    recs, skipped = records(a.data, tok, a.max_len, a.seed)
    ntok = sum(r["n"] for r in recs)
    if rank == 0: print(f"[dpo] {len(recs)} pairs ({skipped} longer than {a.max_len} skipped), {ntok} tokens (longer side), world {world}", flush=True)
    ds = Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in recs])
    ga = max(1, a.eff_batch // (a.mb * world))
    fsdp = bool(os.environ.get("ACCELERATE_USE_FSDP") == "true" or os.environ.get("FSDP_SHARDING_STRATEGY") or os.environ.get("ACCELERATE_FSDP_VERSION"))
    cpu = not torch.cuda.is_available()
    kw = dict(dtype=torch.float32 if cpu else torch.bfloat16, attn_implementation="sdpa")
    if not fsdp and world == 1 and not cpu: kw["device_map"] = {"": local}
    if a.arch == "vl":
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(a.model, **kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    if rank == 0: print(f"[dpo] {type(model).__name__} loaded in {time.time() - t0:.0f}s", flush=True)
    lcfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, target_modules=LM_TARGET, bias="none", task_type="CAUSAL_LM")
    total = a.max_steps if a.max_steps > 0 else math.ceil(math.ceil(len(recs) / (a.mb * world * ga)) * a.epochs)
    want = dict(output_dir=a.out, per_device_train_batch_size=a.mb, gradient_accumulation_steps=ga, learning_rate=a.lr,
                num_train_epochs=a.epochs, max_steps=a.max_steps, lr_scheduler_type="cosine", warmup_steps=max(1, int(a.warmup * total)),
                logging_steps=1, save_strategy="no", report_to="none", bf16=not cpu, seed=a.seed, max_length=a.max_len, beta=a.beta,
                truncation_mode="keep_start", gradient_checkpointing=not a.no_gc, gradient_checkpointing_kwargs={"use_reentrant": False},
                dataset_num_proc=1, dataloader_num_workers=1, remove_unused_columns=False, precompute_ref_log_probs=False)
    if a.sft_weight > 0: want.update(loss_type=["sigmoid", "sft"], loss_weights=[1.0, a.sft_weight])
    import dataclasses
    fields = {f.name for f in dataclasses.fields(DPOConfig)}
    if rank == 0 and set(want) - fields: print(f"[dpo] DPOConfig has no {sorted(set(want) - fields)}: left out", flush=True)
    cfg = DPOConfig(**{k: v for k, v in want.items() if k in fields})
    stats = {"loss": [], "margin": [], "acc": []}

    class Meter(TrainerCallback):
        def on_step_begin(self, args, state, control, **k):
            if state.global_step == 2:
                stats["t2"] = time.time()
                if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        def on_log(self, args, state, control, logs=None, **k):
            if not logs: return
            if "loss" in logs: stats["loss"].append(round(float(logs["loss"]), 4))
            for k_, key in (("rewards/margins", "margin"), ("rewards/accuracies", "acc")):
                if k_ in logs: stats[key].append(round(float(logs[k_]), 4))
            if rank == 0: print(f"[dpo] step {state.global_step} {logs}", flush=True)

    tr = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lcfg, callbacks=[Meter()])
    t1 = time.time(); res = tr.train(); dt = time.time() - t1
    steps = res.global_step
    steady = (time.time() - stats["t2"]) if "t2" in stats else dt
    m = {"ok": True, "world": world, "mb": a.mb, "ga": ga, "pairs": len(recs), "tokens": ntok, "steps": steps, "train_s": round(dt, 1),
         "pairs_per_s": round(a.mb * ga * world * max(1, steps - 2) / max(steady, 1e-6), 3) if steps > 2 else None,
         "loss_first": stats["loss"][0] if stats["loss"] else None, "loss_last": stats["loss"][-1] if stats["loss"] else None,
         "margin_last": stats["margin"][-1] if stats["margin"] else None, "acc_last": stats["acc"][-1] if stats["acc"] else None,
         "losses": stats["loss"][:200], "margins": stats["margin"][:200], "beta": a.beta, "lr": a.lr,
         "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0}
    if world > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            t = torch.tensor([m["peak_mem_gb"]], device=f"cuda:{local}"); dist.all_reduce(t, op=dist.ReduceOp.MAX); m["peak_mem_gb"] = round(float(t.item()), 2)
    if not a.no_save:
        if getattr(tr, "is_fsdp_enabled", False):
            try: tr.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
            except Exception as e: print("[dpo] FULL_STATE_DICT not set:", e, flush=True)
        tr.save_model(a.out)
        if rank == 0: tok.save_pretrained(a.out)
    if rank == 0:
        print("[dpo] METRICS " + json.dumps(m), flush=True)
        if a.metrics: json.dump(m, open(a.metrics, "w"), indent=1)


if __name__ == "__main__":
    main()
