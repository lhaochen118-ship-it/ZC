"""16/11: ZC_INIT_ADAPTER (training/sft.py) — a saved LoRA (incl. LoRA on fused MoE experts) loaded into a fresh PEFT model with the
same config restores every LoRA tensor exactly, with no unexpected key (the check sft.py uses to refuse a layout mismatch)."""
import os, re, tempfile, torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LM = eval(re.search(r'^LM_TARGET = (r".*")$', open(os.path.join(ROOT, "training", "sft.py")).read(), re.M).group(1))
cfg = Qwen3_5MoeTextConfig(vocab_size=128, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                           moe_intermediate_size=32, shared_expert_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
                           layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2, linear_num_value_heads=2,
                           linear_key_head_dim=16, linear_value_head_dim=16)
for experts in (False, True):
    torch.manual_seed(0)
    mk = lambda: LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=LM, bias="none", task_type="CAUSAL_LM",
                            **({"target_parameters": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]} if experts else {}))
    a = get_peft_model(Qwen3_5MoeForCausalLM(cfg), mk())
    for n, p in a.named_parameters():
        if "lora" in n:
            with torch.no_grad(): p.normal_(0, 0.1)
    d = tempfile.mkdtemp(); a.save_pretrained(d)
    torch.manual_seed(1); b = get_peft_model(Qwen3_5MoeForCausalLM(cfg), mk())
    sd = load_file(os.path.join(d, "adapter_model.safetensors")); r = set_peft_model_state_dict(b, sd)
    bad = [k for k in getattr(r, "unexpected_keys", []) or [] if "lora" in k]
    A = {n: p for n, p in a.named_parameters() if "lora" in n}; B = {n: p for n, p in b.named_parameters() if "lora" in n}
    diff = max(float((A[n] - B[n]).abs().max()) for n in A)
    print("experts" if experts else "plain", len(sd), "tensors, unexpected", len(bad), "max diff", diff); assert not bad and diff == 0.0
print("INIT ADAPTER OK")
