#!/usr/bin/env python3
"""LoRA SFT of the main model on the factory's records (docs/08_MASTER_PLAN.md §5.9): TRL SFTTrainer + PEFT, bf16, loss on the
assistant output only (prompt-completion records rendered with the model's chat template exactly as at inference, thinking
off), gradient checkpointing, optional FSDP through `accelerate launch --config_file training/fsdp{2,4}.yaml`.

Records (JSONL): {"messages": [system, user, assistant], ...} (training/data/*). Prompt = chat template of [system, user] with
add_generation_prompt=True, enable_thinking=False (agent.Engine.render does the same at inference); completion = assistant
content + '<|im_end|>'.
LoRA targets: every linear layer of the language model (attention q/k/v/o, linear-attention in_proj_* / out_proj, MLP), never the
vision tower, the MTP head or lm_head (regex over module names).

Usage: [accelerate launch --config_file training/fsdp2.yaml] training/sft.py --model DIR --data a.jsonl[,b.jsonl] --out DIR
       [--r 32 --alpha 64 --lr 2e-4 --epochs 1 --max-len 3072 --mb 4 --eff-batch 64 --max-steps N --max-tokens N --seed 20261108]
       [--arch causal|vl] [--metrics out.json] [--no-save]
Writes the adapter to --out (PEFT format) and metrics (peak memory per GPU, tokens/s, losses) to --metrics."""
import os, sys, json, time, math, argparse, random

LM_TARGET = r"^(?!.*(?:visual|mtp|lm_head)).*\blayers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_qkvz|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
# (linear-attention in_proj_a / in_proj_b project to the 32 heads: smaller than the LoRA rank, left frozen)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--r", type=int, default=32); ap.add_argument("--alpha", type=int, default=64); ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=3072); ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--eff-batch", type=int, default=64); ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-tokens", type=float, default=0, help="token budget: records are subsampled (seeded) to about this many tokens")
    ap.add_argument("--seed", type=int, default=20261108); ap.add_argument("--arch", default="causal", choices=["causal", "vl"])
    ap.add_argument("--metrics", default=""); ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--no-gc", action="store_true", help="no gradient checkpointing"); ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--group-by-length", type=int, default=0, help="v2: batches of similar length (less padding per micro-batch)")
    ap.add_argument("--ckpt-dir", default="", help="where resumable checkpoints go (a kept output folder); default --out")
    ap.add_argument("--resume", default="", help="a checkpoint (or a folder holding checkpoint-N) to continue from (training/deadline.py)")
    ap.add_argument("--save-every", type=int, default=0, help="also checkpoint every N steps (0: only before the session deadline)")
    ap.add_argument("--deadline-reserve", type=float, default=900, help="seconds before ZC_DEADLINE at which training checkpoints + stops")
    return ap.parse_args()


def records(paths, tok, max_len, max_tokens=0, seed=0):
    rows = []
    for p in paths.split(","):
        for l in open(p, encoding="utf-8"):
            if l.strip(): rows.append(json.loads(l))
    out, skipped = [], 0
    for r in rows:
        m = r["messages"]                                               # [system, user, (assistant, user)*, assistant]: loss on the last turn
        prompt = tok.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        comp = m[-1]["content"] + "<|im_end|>"
        n = len(tok(prompt + comp, add_special_tokens=False)["input_ids"])
        if n > max_len: skipped += 1; continue
        out.append({"prompt": prompt, "completion": comp, "n": n})
    random.Random(seed).shuffle(out)
    if max_tokens:
        keep, tot = [], 0
        for r in out:
            if tot >= max_tokens: break
            keep.append(r); tot += r["n"]
        out = keep
    return out, skipped


def ref_records(paths, tok, max_len, n):
    """n records (the last ones of the first file) as (input_ids, labels) with the prompt masked: a loss that means the same
    thing for every trainer (mean over the assistant tokens), measured before and after training."""
    import torch
    rows = [json.loads(l) for l in open(paths.split(",")[0], encoding="utf-8") if l.strip()][-n:]
    out = []
    for r in rows:
        m = r["messages"]
        p = tok.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pi = tok(p, add_special_tokens=False)["input_ids"]; ci = tok(m[-1]["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        ids = (pi + ci)[:max_len]; lab = ([-100] * len(pi) + ci)[:max_len]
        out.append((torch.tensor([ids]), torch.tensor([lab])))
    return out


def ref_loss(model, recs):
    """mean assistant-token loss of the reference records (no gradient), or None when the forward fails."""
    import torch
    try:
        dev = next(p for p in model.parameters()).device
        was = model.training; model.eval(); tot = 0.0
        with torch.no_grad():
            for ids, lab in recs:
                out = model(input_ids=ids.to(dev), labels=lab.to(dev))
                tot += float(out.loss)
        if was: model.train()
        return round(tot / max(1, len(recs)), 4)
    except Exception as e:
        print(f"[sft] reference loss unavailable: {type(e).__name__}: {str(e)[:200]}", flush=True); return None


class TorchLogProb:
    """TRL's fused LM-head loss (a Triton kernel, GPU only) computed with plain torch: CPU tests only (ZC_TORCH_LOGPROB=1)."""
    @staticmethod
    def apply(hidden, weight, bias, targets, temperature, chunk, softcap, scale, outputs):
        import torch
        logits = (hidden.float() @ weight.float().t()) * scale / temperature
        if bias is not None: logits = logits + bias.float()
        lsm = torch.log_softmax(logits, -1)
        lp = lsm.gather(-1, targets[:, None]).squeeze(-1)
        ent = -(lsm.exp() * lsm).sum(-1) if "entropy" in outputs else None
        lss = torch.logsumexp(2 * lsm, -1) if "log_sum_sq_probs" in outputs else None
        ml = logits.mean(-1) if "mean_logits" in outputs else None      # DPO logs the mean logit of each side
        top1 = (logits.argmax(-1) == targets) if "is_top1" in outputs else None
        return lp, ent, lss, ml, top1


def main():
    a = parse()
    import torch
    if os.environ.get("ZC_TORCH_LOGPROB") == "1":
        import trl.trainer.utils as _tu; _tu._ChunkedLogProbFunction = TorchLogProb
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
    from peft import LoraConfig
    try:                                                                # an old torchao (the Kaggle image: 0.10) makes PEFT raise while
        import peft.import_utils as _piu, peft.tuners.lora.torchao as _pto   # building LoRA layers; we never use torchao
        _piu.is_torchao_available = lambda: False; _pto.is_torchao_available = lambda: False
    except Exception: pass
    from trl import SFTTrainer, SFTConfig
    rank = int(os.environ.get("RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1")); local = int(os.environ.get("LOCAL_RANK", "0"))
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token = "<|endoftext|>"
    recs, skipped = records(a.data, tok, a.max_len, a.max_tokens, a.seed)
    ntok = sum(r["n"] for r in recs)
    if rank == 0: print(f"[sft] {len(recs)} records ({skipped} longer than {a.max_len} skipped), {ntok} tokens, world {world}", flush=True)
    ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in recs])
    ga = max(1, a.eff_batch // (a.mb * world))
    fsdp = bool(os.environ.get("ACCELERATE_USE_FSDP") == "true" or os.environ.get("FSDP_SHARDING_STRATEGY") or os.environ.get("ACCELERATE_FSDP_VERSION"))
    cpu = not torch.cuda.is_available()                                   # CPU test of the pipeline (tiny model): fp32
    kw = dict(dtype=torch.float32 if cpu else torch.bfloat16, attn_implementation="sdpa")
    if not fsdp and world == 1 and not cpu: kw["device_map"] = {"": local}
    if a.arch == "vl":
        from transformers import AutoModelForImageTextToText
        model, info = AutoModelForImageTextToText.from_pretrained(a.model, output_loading_info=True, **kw)
    else:
        model, info = AutoModelForCausalLM.from_pretrained(a.model, output_loading_info=True, **kw)
    if rank == 0:
        print(f"[sft] {type(model).__name__} loaded in {time.time() - t0:.0f}s; missing {len(info.get('missing_keys', []))} "
              f"{list(info.get('missing_keys', []))[:5]}; unexpected {len(info.get('unexpected_keys', []))}", flush=True)
    lcfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, target_modules=LM_TARGET, bias="none", task_type="CAUSAL_LM")
    total = a.max_steps if a.max_steps > 0 else math.ceil(math.ceil(len(recs) / (a.mb * world * ga)) * a.epochs)
    want = dict(output_dir=a.ckpt_dir or a.out, per_device_train_batch_size=a.mb, gradient_accumulation_steps=ga, learning_rate=a.lr,
                num_train_epochs=a.epochs, max_steps=a.max_steps, lr_scheduler_type="cosine", warmup_steps=max(1, int(a.warmup * total)),
                logging_steps=a.log_every, save_strategy="steps" if a.save_every else "no", save_steps=a.save_every or 500, save_total_limit=1, report_to="none", bf16=not cpu, seed=a.seed, max_length=a.max_len,
                gradient_checkpointing=not a.no_gc, gradient_checkpointing_kwargs={"use_reentrant": False},
                completion_only_loss=True, dataset_num_proc=1, dataloader_num_workers=2, remove_unused_columns=True,
                packing=False, padding_free=False, group_by_length=bool(a.group_by_length),
                ddp_timeout=600)                                             # a stuck rank fails in 10 min, not 30 (13/11 WR30 part 1:
                                                                             # NCCL all-gather hung at step 0 for the default 1800 s)
    import dataclasses
    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    if rank == 0 and set(want) - fields: print(f"[sft] SFTConfig has no {sorted(set(want) - fields)}: left out", flush=True)
    cfg = SFTConfig(**{k: v for k, v in want.items() if k in fields})
    stats = {"steps": [], "loss": []}

    class Meter(TrainerCallback):
        def on_step_begin(self, args, state, control, **k):
            if state.global_step == 2:
                stats["t2"] = time.time()
                if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        def on_log(self, args, state, control, logs=None, **k):
            if logs and "loss" in logs: stats["loss"].append(round(float(logs["loss"]), 4)); stats["steps"].append(state.global_step)
            if rank == 0 and logs: print(f"[sft] step {state.global_step} {logs}", flush=True)

    import deadline
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lcfg,
                    callbacks=[Meter(), deadline.callback(TrainerCallback, a.deadline_reserve, stats)])
    pm = tr.model
    try:
        names = [n for n, _ in pm.named_modules() if n.endswith("lora_A")]
        if rank == 0:
            kinds = sorted({n.split(".")[-2] for n in names})
            ntrain = sum(p.numel() for p in pm.parameters() if p.requires_grad)
            print(f"[sft] LoRA on {len(names)} modules, kinds {kinds}, trainable {ntrain / 1e6:.1f}M", flush=True)
            stats["lora_modules"] = len(names); stats["lora_kinds"] = kinds; stats["trainable_m"] = round(ntrain / 1e6, 1)
            stats["vision_or_mtp_targeted"] = any(("visual" in n or "mtp" in n or "lm_head" in n) for n in names)
    except Exception as e: print("[sft] module listing failed:", e, flush=True)
    ref = ref_records(a.data, tok, a.max_len, 4)
    lr0 = ref_loss(tr.model, ref) if world == 1 else None             # before FSDP wraps it the model is on CPU / meta: skipped
    t1 = time.time()
    ck = deadline.latest_checkpoint(a.resume)
    if a.resume and rank == 0: print(f"[sft] resume from {ck}", flush=True)
    res = tr.train(resume_from_checkpoint=ck)
    # not after a deadline stop (13/11 v3a part 3: the FSDP reference pass ran out of memory on one rank right after the
    # deadline checkpoint, the other ranks waited 10 min on NCCL, the job exited 1 without train_metrics.json)
    # one GPU only (13/11 v3b: 87 / 87 steps done, then the FSDP reference pass ran out of memory on SOME ranks: rank 2 went
    # on to the all_reduce below while the others still all-gathered parameters for the forward -> NCCL timeout, no adapter
    # saved, 3 h lost; v3a part 4 survived only because every rank ran out of memory). The value is a diagnostic.
    lr1 = ref_loss(tr.model, ref) if world == 1 and not stats.get("stopped_early") else None
    dt = time.time() - t1
    steps = res.global_step
    tok_per_step = a.mb * ga * world * (ntok / max(1, len(recs)))
    steady = (time.time() - stats["t2"]) if "t2" in stats else dt
    m = {"ok": True, "world": world, "mb": a.mb, "ga": ga, "max_len": a.max_len, "records": len(recs), "tokens": ntok, "steps": steps,
         "train_s": round(dt, 1), "tok_per_s": round(tok_per_step * max(1, steps - 2) / max(steady, 1e-6), 1) if steps > 2 else None,
         "loss_first": stats["loss"][0] if stats["loss"] else None, "loss_last": stats["loss"][-1] if stats["loss"] else None,
         "losses": stats["loss"][:200], "stopped_early": stats.get("stopped_early"), "resumed_from": ck, "ref_loss_before": lr0, "ref_loss_after": lr1, "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0, **{k: v for k, v in stats.items() if k.startswith(("lora", "trainable", "vision"))}}
    if world > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            t = torch.tensor([m["peak_mem_gb"]], device=f"cuda:{local}"); dist.all_reduce(t, op=dist.ReduceOp.MAX); m["peak_mem_gb"] = round(float(t.item()), 2)
    if not a.no_save:
        if getattr(tr, "is_fsdp_enabled", False):
            try: tr.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
            except Exception as e: print("[sft] FULL_STATE_DICT not set:", e, flush=True)
        tr.save_model(a.out)
        if rank == 0: tok.save_pretrained(a.out)
    if rank == 0:
        print("[sft] METRICS " + json.dumps(m), flush=True)
        if a.metrics: json.dump(m, open(a.metrics, "w"), indent=1)


if __name__ == "__main__":
    main()
