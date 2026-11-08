#!/usr/bin/env python3
"""The same LoRA SFT as training/sft.py with Unsloth's kernels on ONE GPU (WP1 QĐ1: 4 branches per session when it fits).
Run with PYTHONPATH pointing at an isolated --target install of the Unsloth wheels (training/smoke.py does it).
Usage: sft_unsloth.py --model DIR --data a.jsonl --mb 4 --max-steps 8 [--eff-batch 16 --max-len 3072 --r 32 --alpha 64 --out DIR]"""
import os, sys, json, time, math, argparse
import unsloth                                                           # before transformers / trl (it patches them)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sft import records, ref_records, ref_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", default="/tmp/uns_out")
    ap.add_argument("--mb", type=int, default=4); ap.add_argument("--max-steps", type=int, default=-1); ap.add_argument("--eff-batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=3072); ap.add_argument("--r", type=int, default=32); ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--no-save", action="store_true"); ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=float, default=0, help="token budget: records subsampled (seeded) to about this many tokens")
    ap.add_argument("--warmup", type=float, default=0.03)
    a = ap.parse_args()
    import torch
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback
    t0 = time.time()
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "out_proj"]
    try:
        from unsloth import FastLanguageModel as FM
        model, tok = FM.from_pretrained(model_name=a.model, max_seq_length=a.max_len, dtype=torch.bfloat16, load_in_4bit=False, load_in_8bit=False)
    except Exception as e:
        print(f"[sft] FastLanguageModel failed ({type(e).__name__}: {str(e)[:200]}); FastModel", flush=True)
        from unsloth import FastModel as FM
        model, tok = FM.from_pretrained(model_name=a.model, max_seq_length=a.max_len, dtype=torch.bfloat16, load_in_4bit=False, load_in_8bit=False)
    tok = getattr(tok, "tokenizer", tok)
    model = FM.get_peft_model(model, r=a.r, lora_alpha=a.alpha, lora_dropout=0, target_modules=targets, use_gradient_checkpointing="unsloth", random_state=a.seed)
    print(f"[sft] unsloth model ready in {time.time() - t0:.0f}s", flush=True)
    recs, skipped = records(a.data, tok, a.max_len, a.max_tokens, a.seed); ntok = sum(r["n"] for r in recs)
    print(f"[sft] {len(recs)} records ({skipped} too long), {ntok} tokens", flush=True)
    ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in recs])
    ga = max(1, a.eff_batch // a.mb); stats = {"loss": []}
    class Meter(TrainerCallback):
        def on_step_begin(self, args, state, control, **k):
            if state.global_step == 2: stats["t2"] = time.time(); torch.cuda.reset_peak_memory_stats()
        def on_log(self, args, state, control, logs=None, **k):
            if logs and "loss" in logs: stats["loss"].append(round(float(logs["loss"]), 4))
    import dataclasses
    total = a.max_steps if a.max_steps > 0 else math.ceil(math.ceil(len(recs) / (a.mb * ga)) * a.epochs)
    want = dict(output_dir=a.out, per_device_train_batch_size=a.mb, gradient_accumulation_steps=ga, learning_rate=a.lr, max_steps=a.max_steps,
                num_train_epochs=a.epochs, logging_steps=5, save_strategy="no", report_to="none", bf16=True, seed=a.seed, max_length=a.max_len,
                completion_only_loss=True, warmup_steps=max(1, int(a.warmup * total)), lr_scheduler_type="cosine", dataset_num_proc=1)
    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    tr = SFTTrainer(model=model, args=SFTConfig(**{k: v for k, v in want.items() if k in fields}), train_dataset=ds, processing_class=tok, callbacks=[Meter()])
    ref = ref_records(a.data, tok, a.max_len, 4); lr0 = ref_loss(model, ref)
    t1 = time.time(); res = tr.train(); dt = time.time() - t1
    lr1 = ref_loss(model, ref)
    if a.out and not getattr(a, "no_save", False): model.save_pretrained(a.out); tok.save_pretrained(a.out)
    steady = time.time() - stats.get("t2", t1); steps = res.global_step
    tps = a.mb * ga * (ntok / max(1, len(recs)))
    m = {"ok": True, "unsloth": True, "mb": a.mb, "ga": ga, "steps": steps, "train_s": round(dt, 1),
         "tok_per_s": round(tps * max(1, steps - 2) / max(steady, 1e-6), 1) if steps > 2 else None,
         "loss_first": stats["loss"][0] if stats["loss"] else None, "loss_last": stats["loss"][-1] if stats["loss"] else None, "records": len(recs), "tokens": ntok,
         "ref_loss_before": lr0, "ref_loss_after": lr1, "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    print("[sft] METRICS " + json.dumps(m), flush=True)


if __name__ == "__main__":
    main()
