#!/usr/bin/env python3
"""CPU check of the training stack before any GPU run (WP1 review gate): training/sft.py end to end on a tiny random model of
the Qwen3.5 architecture (same checkpoint naming, 4 layers), then training/merge.py checked numerically against PEFT's own
merge, and the vLLM adapter copy. TRL's fused LM-head loss is a Triton kernel (GPU only); here it is replaced by the same
quantities computed with plain torch (test only).
Usage: test_sft_cpu.py <tiny_model_dir> <records.jsonl> <scratch>"""
import os, sys, json, shutil, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))


class _TorchLogProb:
    @staticmethod
    def apply(hidden, weight, bias, targets, temperature, chunk, softcap, scale, outputs):
        import torch
        logits = (hidden.float() @ weight.float().t()) * scale / temperature
        if bias is not None: logits = logits + bias.float()
        lsm = torch.log_softmax(logits, -1)
        lp = lsm.gather(-1, targets[:, None]).squeeze(-1)
        ent = -(lsm.exp() * lsm).sum(-1) if "entropy" in outputs else None
        top1 = (logits.argmax(-1) == targets) if "is_top1" in outputs else None
        return lp, ent, None, None, top1


def main():
    model, recs, S = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(S, exist_ok=True); ad = os.path.join(S, "adapter"); shutil.rmtree(ad, ignore_errors=True)
    import trl.trainer.utils as U
    U._ChunkedLogProbFunction = _TorchLogProb
    import sft
    sys.argv = ["sft.py", "--model", model, "--data", recs, "--out", ad, "--mb", "1", "--max-steps", "3", "--eff-batch", "2",
                "--metrics", os.path.join(S, "m.json"), "--lr", "1e-3"]
    sft.main()
    m = json.load(open(os.path.join(S, "m.json")))
    assert m["ok"] and m["steps"] == 3 and m["lora_modules"] > 0 and not m["vision_or_mtp_targeted"], m
    assert os.path.isfile(os.path.join(ad, "adapter_config.json")), "adapter not saved"
    # merge.py vs PEFT
    import torch, merge
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    md = os.path.join(S, "merged"); shutil.rmtree(md, ignore_errors=True)
    n = merge.merge(model, md, [ad])
    assert n == m["lora_modules"], (n, m["lora_modules"])
    base = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
    ref = PeftModel.from_pretrained(base, ad).merge_and_unload()
    mine = AutoModelForCausalLM.from_pretrained(md, dtype=torch.float32)
    sd_r, sd_m = ref.state_dict(), mine.state_dict()
    diff = max((sd_r[k].float() - sd_m[k].float()).abs().max().item() for k in sd_r if k in sd_m)
    changed = sum(1 for k in sd_r if "proj" in k and k in sd_m and (sd_m[k].float() - base.state_dict()[k].float()).abs().max().item() > 0)
    print(f"merge: {n} tensors, max |PEFT - merge.py| = {diff:.2e}, changed {changed}")
    assert diff < 2e-2, diff                                            # bf16 rounding of the stored weights
    va = os.path.join(S, "adapter_vllm"); merge.vllm_adapter(model, ad, va)
    cfg = json.load(open(os.path.join(va, "adapter_config.json")))
    assert isinstance(cfg["target_modules"], list) and "in_proj_a" not in cfg["target_modules"], cfg["target_modules"]
    from safetensors.torch import load_file
    ks = list(load_file(os.path.join(va, "adapter_model.safetensors")))
    assert all(".language_model." in k for k in ks), ks[:3]
    print(f"loss {m['loss_first']} -> {m['loss_last']}; LoRA {m['lora_modules']} modules {m['lora_kinds']}")
    print("test_sft_cpu PASS")


if __name__ == "__main__":
    main()
