"""offline check of lrl_gen.force_json with a fake vLLM engine: phase 1 stops at </think> or the cap, phase 2 gets prompt+thought+</think>"""
import sys, types, json
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
vllm = types.ModuleType("vllm"); sp_mod = types.ModuleType("vllm.sampling_params")
class SP:
    def __init__(self, **kw): self.kw = kw
class SO:
    def __init__(self, **kw): self.kw = kw
vllm.SamplingParams = SP; sp_mod.StructuredOutputsParams = SO; sys.modules["vllm"] = vllm; sys.modules["vllm.sampling_params"] = sp_mod
import lrl_gen
class O:
    def __init__(self, t, n): self.outputs = [types.SimpleNamespace(text=t, token_ids=list(range(n)))]
class LLM:
    def __init__(self): self.calls = []
    def generate(self, prompts, sps, use_tqdm=False, lora_request=None):
        self.calls.append((prompts, sps))
        if len(self.calls) == 1:
            assert all(s.kw.get("stop") == ["</think>"] for s in sps)
            return [O("short thought", 3), O("long thought " * 100, sps[1].kw["max_tokens"])]
        assert all("structured_outputs" in s.kw for s in sps)
        assert prompts[0].endswith("short thought\n</think>\n\n"), prompts[0][-40:]
        return [O('{"name": "f"}', 5), O('{"name": "g"}', 5)]
class Eng:
    lora = None
    def __init__(self): self.llm = LLM()
    def fit(self, c, n, think): return ("PROMPT<think>\n", n)
e = Eng()
t = lrl_gen.force_json(e, [[1], [2]], [0.0, 0.7], [1, 2], 7168, {"type": "object"})
assert lrl_gen.first_json(t[0]) == {"name": "f"} and lrl_gen.first_json(t[1]) == {"name": "g"}, t
assert e.llm.calls[0][1][0].kw["max_tokens"] == 7168, e.llm.calls[0][1][0].kw
print("force_json PASS")
