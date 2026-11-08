#!/usr/bin/env python3
"""Hybrid flow scheduler (docs/08_MASTER_PLAN.md §4.1, §5.7): EVERY question is answered by a program the model writes; rules
only prepare context (schema card, hints, retrieved checked examples), check programs (precheck / postcheck) and run tools.

Per shop (inherited from pipeline.Pipeline): mapping to canonical tables C (automap first: --prefer-map fmt:automap; LLM mapping
samples only when it fails), claims / work stealing between GPU processes, deadlines.
Per question:
  hints (src/hints.py) + examples (src/retrieval.py) -> prompt (src/shopctx.py / prompting.py, same as the training records)
  round 1: --k-fast samples (temperature 0, then --temp), grammar-constrained (--grammar 1) -> parse -> precheck (errors: a repair
           turn with the issues and tool hints, up to --max-turns) -> run on the CPU pool (the toolkit + tools) -> postcheck flags
  early stop (--early-stop 1): the round-1 samples all ran, agree and carry no flag -> answer
  round 2: up to --k-max samples in all (variants --variants: direct / rephrase), same repairs
  vote: valid samples grouped by answer (flagged ones weigh 0.5); conf = top weight / samples tried; answer when conf >= --theta-abstain
Everything a question went through is logged (hints, examples, every turn: text, code, precheck, execution, postcheck, logprob)."""
import os, re, json, time, pickle
from collections import Counter
import pipeline, dm, prompting, api_spec, retrieval, shopctx
import precheck as pre
import tools as toolsmod

agent = pipeline.agent                                                  # rebound by pipeline.bind()


class HybridPipeline(pipeline.Pipeline):
    def __init__(self, a, llm, pool, rulebook, bizs, qs, sub, logf, clog, claim_dir=""):
        global agent
        agent = pipeline.agent
        a.k = max(1, int(a.k_max))                                      # Pipeline.run: fast mode = k_now < a.k
        super().__init__(a, llm, pool, rulebook, bizs, qs, sub, logf, clog, claim_dir)
        self.hide = tuple(x for x in (getattr(a, "hide_api", "") or "").split(",") if x)
        self.escapes = tuple(x for x in (getattr(a, "escape", "python") or "").split(",") if x and x != "none")
        self.spec = api_spec.build_api_spec(hide=self.hide)
        self.bank = retrieval.Bank.load(include_holdout=bool(getattr(a, "bank_holdout", 0)))
        self.excl = {x for x in (getattr(a, "exclude_bids", "") or "").split(",") if x}
        self.variants = [v for v in (getattr(a, "variants", "direct") or "direct").split(",") if v in prompting.VARIANT] or ["direct"]
        self.ctxs = {}
        self.stats = Counter()

    # -- per shop context (prompt prefix, hints, grammar)
    def ctx_of(self, b, can):
        if b in self.ctxs: return self.ctxs[b]
        c = None
        if can and can.get("path"):
            try:
                d = pickle.load(open(can["path"], "rb"))
                c = shopctx.ShopCtx(self.a.split, b, self.bizs[b], d["C"], d.get("notes", ""), d.get("data_end"), self.rulebook,
                                    hide=self.hide, mode=getattr(self.a, "prompt", "compact"), escapes=self.escapes)
                if getattr(self.a, "grammar", 1): c.grammar(int(getattr(self.a, "max_enum", 5000)))
            except Exception as e:
                print(f"[hybrid] {b}: no context ({type(e).__name__}: {e})", flush=True); c = None
        self.ctxs[b] = c
        return c

    def gen2(self, conv, temp, seed, max_tokens, cb, think=None, prio=0, grammar=None):
        rid = self.llm.add(conv, temp, seed, max_tokens, think, prio, grammar)
        self.cb[rid] = lambda text, ntok, rid=rid: cb(text, ntok, getattr(self.llm, "lp", {}).pop(rid, None))

    # -- questions
    def answer_chunk(self, b, ci, can):
        qs = self.chunks[b][ci]; left = {"n": len(qs)}; self.inflight_q += len(qs); prio = self.order.index(b)
        if not qs: return self.chunk_done(b, ci)
        ctx = self.ctx_of(b, can)
        def done():
            self.inflight_q -= 1; left["n"] -= 1
            if left["n"] == 0: self.chunk_done(b, ci)
        for q in qs:
            try: self.ask(q, b, can, ctx, prio, done)
            except Exception as e:                                      # never lose a question: abstain it with the reason
                print(f"[hybrid] {q['id']}: {type(e).__name__}: {e}", flush=True)
                self.finish({"q": q, "b": b, "t": time.time(), "samples": [], "hints": {}, "examples": [], "done": done, "error": str(e)[:300]})

    def ask(self, q, b, can, ctx, prio, done):
        a = self.a
        st = {"q": q, "b": b, "t": time.time(), "samples": [], "pending": 0, "round": 1, "done": done, "ctx": ctx, "can": can, "prio": prio}
        if ctx is None:
            st.update(hints={}, examples=[], error="no canonical tables"); return self.finish(st)
        h = ctx.hints(q["question"], q["as_of_date"])
        st["hints"] = h
        excl = set(self.excl) | {b}                                     # never the shop's own checked examples (honest on train shops)
        exs = self.bank.retrieve(q["question"], h, k=int(a.shots), exclude_bids=excl, exclude_ids={q["id"]})
        st["examples"] = [e.get("id") for e in exs]
        shown = h if getattr(a, "hints", 1) else None
        st["msgs"] = {v: ctx.messages(q["question"], q["as_of_date"], exs, shown, v) for v in set(self.variants) | {"direct"}}
        st["qkey"] = dm.key(h.get("clean") or q["question"])
        k1 = 1 if self.k_now < a.k else max(1, int(a.k_fast))
        for j in range(k1): self.sample(st, j, "direct", 0.0 if j == 0 else a.temp)

    def sample(self, st, j, variant, temp):
        a, q, ctx = self.a, st["q"], st["ctx"]
        conv = list(st["msgs"][variant]); smp = {"j": j, "variant": variant, "temp": temp, "turns": [], "ok": False}
        st["samples"].append(smp); st["pending"] += 1
        think = bool(getattr(a, "think", False))
        grammar = ctx.grammar(int(getattr(a, "max_enum", 5000))) if getattr(a, "grammar", 1) and not think else None
        mt = int(a.max_tokens)
        def launch():
            seed = 1000 * len(smp["turns"]) + 17 * j + 1 + 7919 * int(getattr(a, "seed_base", 0) or 0)
            self.gen2(conv, temp, seed, mt, on_text, think, st["prio"], grammar)
        def on_text(text, ntok, lp):
            self.ntok += ntok; self.stats["gen"] += 1
            out = prompting.parse_output(text)
            turn = {"text": text[-1200:], "ntok": ntok, "lp": lp, "code": out["code"], "escape": out["escape"], "plan": out["plan"]}
            smp["turns"].append(turn); conv.append({"role": "assistant", "content": text})
            iss = pre.precheck(out["code"], self.spec, st["hints"], ctx.entity_values, out["escape"], q["as_of_date"])
            turn["precheck"] = [f"{i['level']}:{i['rule']}:{i['msg'][:120]}" for i in iss]
            if pre.errors(iss):
                self.stats["precheck_error"] += 1
                return retry("The program has problems:\n" + pre.message(iss))
            code = out["code"]
            if out["escape"] == "sql": code = sql_program(code)
            key = (q["id"], f"{j}.{len(smp['turns'])}")
            self.cpu.submit(agent.run_code, (key, st["b"], q["as_of_date"], code, a.timeout, st["can"]["path"]), a.timeout, on_exec)
        def on_exec(r):
            turn = smp["turns"][-1]
            turn["exec"] = {k: (str(v)[:400] if k in ("error", "stdout") else v) for k, v in r.items() if k in ("ok", "error", "answer", "reason")}
            if not r.get("ok"):
                self.stats["exec_error"] += 1
                return retry(f"The program failed:\n{str(r.get('error'))[:700]}" + repair_hint(r.get("error"), self.spec))
            k_asked = top_k_asked(st["qkey"])
            flags = toolsmod.postcheck(r["answer"], st["hints"].get("answer_type"), None, st["qkey"], k_asked)
            turn["post"] = flags; smp.update(ok=True, answer=r["answer"], flags=flags, lp=turn.get("lp"), escape=turn.get("escape"))
            settle()
        def retry(msg):
            if len(smp["turns"]) <= int(getattr(a, "max_turns", 2)):
                conv.append({"role": "user", "content": msg + "\n\nWrite the corrected output in the same format (<plan> ... </plan> then <code> ... </code>)."})
                return launch()
            settle()
        def settle():
            st["pending"] -= 1
            if st["pending"] == 0: self.after_round(st)
        launch()

    def after_round(self, st):
        a = self.a
        valid = [s for s in st["samples"] if s["ok"]]
        if st["round"] == 1:
            groups = Counter(agent.canon_of(s["answer"]) for s in valid)
            agree = len(valid) == len(st["samples"]) and len(groups) == 1 and not any(s["flags"] for s in valid)
            fast = self.k_now < a.k
            if (getattr(a, "early_stop", 1) and len(valid) >= 2 and agree) or fast or len(st["samples"]) >= a.k:
                st["stop"] = "early" if agree and len(valid) >= 2 else ("fast" if fast else "kmax")
                return self.finish(st)
            st["round"] = 2
            n0 = len(st["samples"])
            for i, j in enumerate(range(n0, int(a.k)) ):
                v = self.variants[i % len(self.variants)]
                self.sample(st, j, v, a.temp)
            return
        self.finish(st)

    def finish(self, st):
        a, q = self.a, st["q"]
        if q["id"] in self.written: return st["done"]()
        votes, rep = Counter(), {}
        for s in st["samples"]:
            if not s.get("ok"): continue
            c = agent.canon_of(s["answer"]); votes[c] += 0.5 if s.get("flags") else 1.0; rep.setdefault(c, s["answer"])
        n = max(1, len(st["samples"]))
        if votes:
            top, w = votes.most_common(1)[0]; conf = w / n
            final = rep[top] if conf >= float(getattr(a, "theta_abstain", 0.5)) else {"type": "abstain"}
        else: conf, final = 0.0, {"type": "abstain"}
        self.stats["answered" if final.get("type") != "abstain" else "abstained"] += 1
        self.stats[f"stop_{st.get('stop', 'round2')}"] += 1
        log = {"hints": st.get("hints"), "examples": st.get("examples"), "stop": st.get("stop", "round2"), "error": st.get("error"),
               "best": rep[top] if votes else None,
               "samples": [{k: v for k, v in s.items()} for s in st["samples"]]}
        agent.write_answer(self.sub, self.logf, a, q, final, conf, votes, log, {}, time.time() - st["t"])
        self.sub.flush(); self.logf.flush(); self.n_answered += 1; self.written.add(q["id"])
        st["done"]()

    def chunk_done(self, b, ci):
        super().chunk_done(b, ci)
        if self.done_chunks % 10 == 0: print(f"[hybrid] stats {dict(self.stats)}", flush=True)


_TOPK = re.compile(r"\btop\s*(\d{1,2})\b|(?<![/\d.,])\b(\d{1,2})\s+(?:mat hang|mon|san pham|sp|khach(?: hang)?|nha cung cap|ncc|nhan vien|tho|dong (?:may|xe)|xe|hang)\b(?![/\d])")


def top_k_asked(qkey):
    m = _TOPK.search(qkey or "")
    if m and re.search(r"\b(?:nao|nhat|nhut|top|chay)\b", qkey): return int(m.group(1) or m.group(2))
    return None


def repair_hint(err, spec):
    """extra context for a failed program: the help of the function a TypeError names, the closest names of a LookupError."""
    e = str(err or "")
    m = re.search(r"([a-z_][a-z0-9_]*)\(\) (?:got an unexpected keyword|missing \d+ required|takes \d+)", e)
    if m: return "\n" + api_spec.help_text(spec, m.group(1))[:600]
    if "LookupError" in e or "KeyError" in e:
        return "\nUse the exact data value (find(kind, text) lists matching values) or the right column (schema())."
    if "Unanswerable" in e: return ""
    return ""


def sql_program(sql):
    """the <sql> escape (WP12): a DuckDB query over the canonical tables as a program."""
    return "ANSWER = sql_answer(" + json.dumps(sql) + ")"
