"""P5 16/11: training/merge_fp8.py on a tiny synthetic block-FP8 checkpoint (official per-expert layout) with a PEFT adapter that has
LoRA on attention AND on the fused routed experts (target_parameters): the merged FP8 weights must equal W + PEFT's own delta."""
import os, sys, json, tempfile, subprocess, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "training")); sys.path.insert(0, os.path.join(ROOT, "src"))
from safetensors.torch import save_file, load_file
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
from peft import LoraConfig, get_peft_model
import merge_fp8 as mf
torch.manual_seed(0)
H, I, E, NL = 128, 128, 4, 2
cfg = Qwen3_5MoeTextConfig(vocab_size=128, hidden_size=H, num_hidden_layers=NL, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                           moe_intermediate_size=I, shared_expert_intermediate_size=I, num_experts=E, num_experts_per_tok=2,
                           layer_types=["full_attention"] * NL)
m = Qwen3_5MoeForCausalLM(cfg); W0 = {n: p.detach().clone().float() for n, p in m.named_parameters()}
pm = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]))
for n, p in pm.named_parameters():
    if "lora" in n:
        with torch.no_grad(): p.normal_(0, 0.05)
d = tempfile.mkdtemp(); ad = os.path.join(d, "adapter"); pm.save_pretrained(ad)
W1 = {n: p.detach().float() for n, p in pm.merge_and_unload().named_parameters()}
fp8 = os.path.join(d, "fp8"); os.makedirs(fp8); T = {}
def put(name, W):
    q, s = mf.block_quant(W.float()); T[name] = q; T[name + "_scale_inv"] = s.float()
for L in range(NL):
    put(f"model.language_model.layers.{L}.self_attn.q_proj.weight", W0[f"model.layers.{L}.self_attn.q_proj.weight"])
    for e in range(E):
        gu = W0[f"model.layers.{L}.mlp.experts.gate_up_proj"][e]
        put(f"model.language_model.layers.{L}.mlp.experts.{e}.gate_proj.weight", gu[:I]); put(f"model.language_model.layers.{L}.mlp.experts.{e}.up_proj.weight", gu[I:])
        put(f"model.language_model.layers.{L}.mlp.experts.{e}.down_proj.weight", W0[f"model.layers.{L}.mlp.experts.down_proj"][e])
save_file(T, os.path.join(fp8, "model-00001.safetensors"))
json.dump({"weight_map": {k: "model-00001.safetensors" for k in T}}, open(os.path.join(fp8, "model.safetensors.index.json"), "w"))
json.dump({"text_config": {"moe_intermediate_size": I, "num_experts": E}}, open(os.path.join(fp8, "config.json"), "w"))
out = os.path.join(d, "out")
r = subprocess.run([sys.executable, os.path.join(ROOT, "training", "merge_fp8.py"), "x", out, "--name", "M", "--adapter", ad, "--fp8-dir", fp8], capture_output=True, text=True)
print(r.stdout[-600:], r.stderr[-1500:]); assert r.returncode == 0
M = load_file(os.path.join(out, "models", "M", "model-00001.safetensors")); worst = 1.0
B0 = load_file(os.path.join(fp8, "model-00001.safetensors"))
def chk(name, ref):                     # the merged delta (dequant merged - dequant base) must match PEFT's delta (ref - W0): cosine
    global worst
    got = mf.dequant(M[name], M[name + "_scale_inv"]) - mf.dequant(B0[name], B0[name + "_scale_inv"])
    want = ref.float() - mf.dequant(B0[name], B0[name + "_scale_inv"])
    cos = float(torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0)); worst = min(worst, cos)
for L in range(NL):
    chk(f"model.language_model.layers.{L}.self_attn.q_proj.weight", W1[f"model.layers.{L}.self_attn.q_proj.weight"])
    for e in range(E):
        gu = W1[f"model.layers.{L}.mlp.experts.gate_up_proj"][e]
        chk(f"model.language_model.layers.{L}.mlp.experts.{e}.gate_proj.weight", gu[:I]); chk(f"model.language_model.layers.{L}.mlp.experts.{e}.up_proj.weight", gu[I:])
        chk(f"model.language_model.layers.{L}.mlp.experts.{e}.down_proj.weight", W1[f"model.layers.{L}.mlp.experts.down_proj"][e])
print("min cosine(merged delta, PEFT delta)", worst); assert worst > 0.95, worst; print("MERGE_FP8 EXPERTS OK")
