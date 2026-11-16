"""P5 16/11: check the per-expert delta of a PEFT LoRA on fused MoE expert parameters (target_parameters) against PEFT's own merge:
delta[e] = B[:, e::E] @ A[e*r:(e+1)*r] * alpha / r   (A: (E*r, in) expert blocks; B: (out, r*E) expert-strided)."""
import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
from peft import LoraConfig, get_peft_model
torch.manual_seed(0)
cfg = Qwen3_5MoeTextConfig(vocab_size=128, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                           moe_intermediate_size=32, shared_expert_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
                           layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2, linear_num_value_heads=2,
                           linear_key_head_dim=16, linear_value_head_dim=16)
m = Qwen3_5MoeForCausalLM(cfg); W0 = {n: p.detach().clone() for n, p in m.named_parameters()}
r = 4; pm = get_peft_model(m, LoraConfig(r=r, lora_alpha=8, lora_dropout=0.0, target_modules=[], target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]))
sd = {}
for n, p in pm.named_parameters():
    if "lora" in n:
        with torch.no_grad(): p.normal_(0, 0.1)
        sd[n] = p.detach().clone()
W1 = {n: p.detach() for n, p in pm.merge_and_unload().named_parameters()}
worst = 0.0
for L in range(2):
    for tgt, key in (("gate_up_proj", f"base_model.model.model.layers.{L}.mlp.experts.base_layer"), ("down_proj", f"base_model.model.model.layers.{L}.mlp.experts")):
        A, B = sd[key + ".lora_A.default.weight"], sd[key + ".lora_B.default.weight"]; n = f"model.layers.{L}.mlp.experts.{tgt}"
        d = W1[n] - W0[n]; E = d.shape[0]
        c = torch.stack([B[:, e::E] @ A[e * r:(e + 1) * r] for e in range(E)]) * (8 / r)
        worst = max(worst, float((c - d).abs().max()))
print("max abs diff", worst); assert worst < 1e-5, worst; print("EXPERT LORA LAYOUT OK")
