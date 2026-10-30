#!/usr/bin/env python3
"""QLoRA supervised fine-tuning of the agent's LLM on harvested correct programs (src/harvest.py), all GPUs of the
session via DDP. Launch (one process per GPU):

  torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) train_sft.py --model <bf16 weights dir> --data sft.jsonl --out OUT/lora

Memory / speed choices (24 GB L4 or 15 GB T4 per process, prompts up to ~20k tokens):
- base weights in 4-bit NF4 (bitsandbytes, double quantisation) -> ~9 GB for 14B; LoRA r=16 on every linear layer;
- bf16 compute on Ampere+ (L4), fp16 + loss scaling on T4; gradient checkpointing (non-reentrant) for long sequences;
- micro-batch 1 sequence per GPU (no padding waste) + gradient accumulation for the effective batch;
- loss only on the assistant turn (the program), the long shared prompt is context only;
- examples sorted into length buckets and spread round-robin over ranks so GPUs finish steps together.
The adapter (a few tens of MB, ~0.06 B parameters) is served by vLLM on top of the quantised base (--lora in agent.py);
total parameters stay under 15 B."""
import os, json, math, time, random, argparse
import torch
import torch.distributed as dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=20480); ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--holdout", default="", help="comma-separated business ids kept out of training (eval loss)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ddp = "LOCAL_RANK" in os.environ
    if ddp: dist.init_process_group("nccl"); lr_ = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr_)
    else: lr_ = 0
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    bf16 = torch.cuda.get_device_capability(lr_)[0] >= 8; dtype = torch.bfloat16 if bf16 else torch.float16
    random.seed(a.seed); torch.manual_seed(a.seed)

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    tok = AutoTokenizer.from_pretrained(a.model)

    # ---- data: chat template, labels only on the assistant turn
    hold = set(filter(None, a.holdout.split(",")))
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    def encode(msgs):
        kw = {"enable_thinking": False} if "qwen3" in a.model.lower() else {}
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, **kw)
        full = prompt + msgs[-1]["content"] + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]; ids = tok(full, add_special_tokens=False)["input_ids"]
        if len(ids) > a.max_len: return None
        return ids, [-100] * len(p_ids) + ids[len(p_ids):]
    train, evals = [], []
    for r in rows:
        e = encode(r["messages"])
        if e: (evals if r["business_id"] in hold else train).append(e)
    if rank == 0: print(f"train {len(train)} eval {len(evals)} (dropped {len(rows) - len(train) - len(evals)} longer than {a.max_len})", flush=True)
    train.sort(key=lambda e: len(e[0]))                              # length buckets, then shuffle buckets
    buckets = [train[i:i + world * a.accum] for i in range(0, len(train), world * a.accum)]
    random.shuffle(buckets); order = [e for b in buckets for e in b]
    mine = order[rank::world]                                          # round-robin: ranks get similar lengths per step

    # ---- model
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=q, torch_dtype=dtype, device_map={"": lr_},
                                                 attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    if rank == 0: model.print_trainable_parameters()
    if ddp: model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lr_], find_unused_parameters=False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    steps = max(1, math.ceil(len(mine) * a.epochs / a.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 20)) * max(0.0, 1 - s / steps))
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)

    def run_example(ids, labels):
        x = torch.tensor([ids], device=lr_); y = torch.tensor([labels], device=lr_)
        with torch.autocast("cuda", dtype=dtype):
            return model(input_ids=x, labels=y).loss

    model.train(); t0 = time.time(); seen_tok = 0; n_ex = int(len(mine) * a.epochs)
    for step in range(steps):
        batch = [mine[(step * a.accum + i) % len(mine)] for i in range(a.accum) if step * a.accum + i < n_ex]
        if not batch: break
        for i, (ids, labels) in enumerate(batch):
            sync = (i == len(batch) - 1) or not ddp                    # all-reduce gradients once per optimiser step
            ctx = model.no_sync() if (ddp and not sync) else torch.enable_grad()
            with ctx:
                loss = run_example(ids, labels) / len(batch)
                scaler.scale(loss).backward()
            seen_tok += len(ids)
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        if rank == 0 and (step % 10 == 0 or step == steps - 1):
            el = time.time() - t0
            print(f"step {step + 1}/{steps} loss {loss.item() * len(batch):.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{seen_tok * world / el:.0f} tok/s mem {torch.cuda.max_memory_allocated(lr_) / 2**30:.1f} GiB", flush=True)

    if evals and rank == 0:
        model.eval(); tot = 0.0
        with torch.no_grad():
            for ids, labels in evals[:200]: tot += run_example(ids, labels).item()
        print(f"eval loss (held-out businesses) {tot / min(len(evals), 200):.4f}", flush=True)
    if rank == 0:
        m = model.module if ddp else model
        m.save_pretrained(a.out); tok.save_pretrained(a.out); print("saved adapter to", a.out, flush=True)
    if ddp: dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
