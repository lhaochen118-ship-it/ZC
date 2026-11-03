#!/usr/bin/env python3
"""Engine.gen batching without a GPU: a mock vLLM records every generate() call.
Checks: requests of one business share a prefix group; groups are packed into waves under --prefix-budget; each wave
is one warm-up call (1 token per distinct prefix) + one generation call; outputs return in input order."""
import os, sys, types, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
calls = []


class Tok:
    def encode(self, s, add_special_tokens=False): return list(range(len(s) // 4))
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw): return "\n".join(m["content"] for m in msgs)


class Out:
    def __init__(self, t): self.outputs = [types.SimpleNamespace(text=t, token_ids=[0] * 3)]


class LLM:
    def __init__(self, **kw): pass
    def get_tokenizer(self): return Tok()
    def generate(self, prompts, sps, use_tqdm=False, lora_request=None):
        calls.append((len(prompts), sps[0].max_tokens)); return [Out("OUT:" + p[-12:]) for p in prompts]


class SP:
    def __init__(self, **kw): self.__dict__.update(kw)


sys.modules["vllm"] = types.SimpleNamespace(LLM=LLM, SamplingParams=SP)
import agent
eng = agent.Engine.__new__(agent.Engine)
eng.llm, eng.tok, eng.lora, eng.prefix_budget = LLM(), Tok(), None, 3100
eng.a = argparse.Namespace(max_len=100000, think=False, model="mock", prefix_budget=3100, quant="", think_map=False)
pref = {b: f"# RULEBOOK\n{'r' * 4000}\n\n# BUSINESS README\n{b * 2000}\n\n# CANONICAL TABLES\ncard {b}" for b in "abcd"}
convs = [[{"role": "user", "content": pref[b] + f"\n\n# EXAMPLES\nex {q}\n\n# QUESTION\nq{b}{q}-{s}"}] for b in "abcd" for q in range(5) for s in range(4)]
texts, ntok = eng.gen(convs, [0.0] * len(convs), [1] * len(convs), 256)
assert all(t.endswith(c[0]["content"][-8:]) for t, c in zip(texts, convs)), "outputs out of order"
# prefix ~1500 tokens each, budget 3100 -> 2 businesses per wave -> 2 waves x (warm-up + generation)
assert calls == [(2, 1), (40, 256), (2, 1), (40, 256)], calls
print("calls", calls, "| PASS")

# ---- StepLLM (pipeline adapter) against a mock vLLM LLMEngine: add_request / step, priority refused -> plain FCFS
import pipeline
sys.modules["vllm.sampling_params"] = types.SimpleNamespace(RequestOutputKind=types.SimpleNamespace(FINAL_ONLY="final"))


class Core:
    def __init__(self): self.q, self.calls = [], []
    def add_request(self, rid, prompt, sp, **kw):
        if "priority" in kw: raise ValueError("Got priority 3 but Priority scheduling is not enabled.")
        self.q.append((rid, prompt, sp)); self.calls.append((rid, sp))
    def step(self):
        out, self.q = self.q[:3], self.q[3:]
        return [types.SimpleNamespace(request_id=r, finished=True, outputs=[types.SimpleNamespace(text="T" + p[-3:], token_ids=[1, 2])]) for r, p, _ in out]


eng.llm.llm_engine = Core()
st = pipeline.StepLLM(eng)
rids = [st.add([{"role": "user", "content": f"q{i:03d}"}], 0.0, 1, 64, priority=3) for i in range(7)]
got = []
while st.busy(): got += st.step()
assert [g[0] for g in got] == rids and got[0][1] == "T000" and not st.prio_ok, (got, st.prio_ok)
assert all(sp.output_kind == "final" and sp.max_tokens == 64 and sp.temperature == 0.0 for _, sp in eng.llm.llm_engine.calls)
print("StepLLM | PASS")

# ---- budget forcing: a thinking request is cut at think_cap; still reasoning -> the end of the reasoning is appended and
# the request continues with the rest of its budget; the caller gets one answer (both parts, both token counts)
class Core2(Core):
    def add_request(self, rid, prompt, sp, **kw): self.q.append((rid, prompt, sp)); self.calls.append((rid, prompt, sp))
    def step(self):
        out, self.q = self.q[:3], self.q[3:]
        res = []
        for r, p, sp in out:
            if r.endswith("c"): text, fr = "```python\nC = 1\n```", "stop"
            elif "loop" in p: text, fr = "Wait, " * 10, "length"                     # reasons to the cap
            elif "late" in p: text, fr = "ok\n</think>\n\n```python\nC =", "length"   # cut inside the program
            else: text, fr = "short\n</think>\n\n```python\nC = 2\n```", "stop"
            res.append(types.SimpleNamespace(request_id=r, finished=True, outputs=[types.SimpleNamespace(text=text, token_ids=[1] * (sp.max_tokens if fr == "length" else 5), finish_reason=fr)]))
        return res


eng.llm.llm_engine = Core2(); eng.a.think = False
st = pipeline.StepLLM(eng, think_cap=2000)
r1 = st.add([{"role": "user", "content": "loop"}], 0.0, 7, 4000, think=True)
r2 = st.add([{"role": "user", "content": "late"}], 0.7, 8, 4000, think=True)
r3 = st.add([{"role": "user", "content": "plain"}], 0.7, 9, 4000, think=True)
r4 = st.add([{"role": "user", "content": "loop"}], 0.0, 1, 4000)               # not thinking: never capped
got = {}
while st.busy():
    for rid, text, n in st.step(): got[rid] = (text, n)
calls = {rid: (p, sp) for rid, p, sp in eng.llm.llm_engine.calls}
assert calls[r1][1].max_tokens == 2000 and calls[r1][1].temperature == 0.6 and calls[r1][1].top_k == 20, vars(calls[r1][1])
assert got[r1][0].startswith("Wait, ") and st.THINK_END in got[r1][0] and got[r1][0].endswith("```python\nC = 1\n```") and got[r1][1] == 2000 + 5, got[r1]
assert calls[r1 + "c"][0] == "loop" + "Wait, " * 10 + st.FORCE and calls[r1 + "c"][1].max_tokens == 2000, calls[r1 + "c"]
assert calls[r2 + "c"][0].endswith("```python\nC =") and got[r2][0].count(st.THINK_END) == 1, (calls[r2 + "c"], got[r2])   # no second end
assert r3 + "c" not in calls and got[r3][1] == 5 and r4 + "c" not in calls and calls[r4][1].max_tokens == 4000 and calls[r4][1].temperature == 0.0
assert not st.req and not st.cont, (st.req, st.cont)
print("StepLLM budget forcing | PASS")
