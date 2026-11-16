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
import pipeline, dm, prompting, api_spec, retrieval, shopctx, lrl
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
        if self.hide: os.environ["ZC_HIDE_API"] = ",".join(self.hide)      # mode H: lrl / link / judge / units drop hidden names
        if int(getattr(a, "api_extra", 0) or 0): os.environ["ZC_API_EXTRA"] = "1"   # v2 F (prompting.api_extra)
        self.escapes = tuple(x for x in (getattr(a, "escape", "python") or "").split(",") if x and x != "none")
        self.spec = api_spec.build_api_spec(hide=self.hide)
        self.bank = retrieval.Bank.load(include_holdout=bool(getattr(a, "bank_holdout", 0)))
        self.shop_ex, self.own_k = {}, int(os.environ.get("ZC_SHOP_EXAMPLES_K", "0") or 0)   # v2 I3: model-written examples of the
        if self.own_k and os.path.isfile(os.environ.get("ZC_SHOP_EXAMPLES", "")):           # shop's own functions (predict_flow)
            byb = {}
            for l in open(os.environ["ZC_SHOP_EXAMPLES"], encoding="utf-8"):
                if l.strip(): r = json.loads(l); byb.setdefault(r["business_id"], []).append(r)
            self.shop_ex = {b: retrieval.Bank(rs) for b, rs in byb.items()}
            print(f"[hybrid] shop examples: {sum(len(v) for v in byb.values())} for {len(byb)} shops, {self.own_k} per question", flush=True)
        self.excl = {x for x in (getattr(a, "exclude_bids", "") or "").split(",") if x}
        self.variants = [v for v in (getattr(a, "variants", "direct") or "direct").split(",") if v in prompting.VARIANT] or ["direct"]
        self.ctxs = {}
        self.stats = Counter()
        lp = [x for x in (getattr(a, "lrl", "") or "").split(",") if x]  # plan 3: learned rule cards (src/lrl.py): files, or names
        here = os.path.dirname(os.path.abspath(__file__))               # under src/refs/lrl/ (shipped with the code) or <split>/lrl/
        lp = [next((c for c in (x, os.path.join(here, "refs", "lrl", x), os.path.join(a.split, "lrl", x)) if os.path.isfile(c)), x) for x in lp]
        missing = [x for x in lp if not os.path.isfile(x)]
        if missing: print(f"[hybrid] WARNING learned rule cards not found: {missing}", flush=True)
        self.lib = lrl.Library.load(",".join(lp)) if lp else None
        if self.lib is not None:                                        # generic triggers out, by the run's own questions (inputs)
            self.lib.prune([x["question"] for x in qs], float(getattr(a, "lrl_max_df", 0.02) or 0.02))
        if self.lib is not None: print(f"[hybrid] learned rule cards: {len(self.lib)} ({lp})", flush=True)
        self.links = None                                               # plan 5 W2: rulebook rows the model linked to each question
        if getattr(a, "links", ""):                                     # (src/link.py); a named file that is missing stops the run
            if not os.path.isfile(a.links):                             # a shipped file named relative to the code (repo: src/refs/...,
                here = os.path.dirname(os.path.abspath(__file__))      # Kaggle stage: <stage>/refs/...)
                for cand in (os.path.join(os.path.dirname(here), a.links), os.path.join(here, a.links), os.path.join(here, a.links[4:] if a.links.startswith("src/") else a.links)):
                    if os.path.isfile(cand): a.links = cand; break
            if not os.path.isfile(a.links): raise SystemExit(f"[hybrid] --links {a.links} not found")   # (no silent fallback, 14/11)
            self.links = {}
            for l in open(a.links, encoding="utf-8"):
                if l.strip(): r = json.loads(l); self.links[r["id"]] = [lrl.strip_hidden(c) for c in (r.get("cards") or [])]
            print(f"[hybrid] linked rules: {len(self.links)} questions, {sum(1 for v in self.links.values() if v)} with rows ({a.links})", flush=True)
        self.units_all = []                                             # plan 4 W: executable rule units (src/units.py)
        if getattr(a, "units", ""):
            import units as unitsmod
            self.units_all = unitsmod.load(a.units)
            print(f"[hybrid] rule units: {len(self.units_all)} ({a.units})", flush=True)

    # -- per shop context (prompt prefix, hints, grammar)
    def rulebook_for(self, b):
        """the rulebook of shop b: <split>/rulebooks/<b>.md when an evaluation split mixes shops of two original splits
        (src/eval_flow.py: a held-out TRAINING shop keeps the training rulebook, docs/15 §4.3), else the split's rulebook."""
        p = os.path.join(self.a.split, "rulebooks", b + ".md")
        return open(p, encoding="utf-8").read() if os.path.isfile(p) else self.rulebook

    def ctx_of(self, b, can):
        if b in self.ctxs: return self.ctxs[b]
        c = None
        if can and can.get("path"):
            try:
                d = pickle.load(open(can["path"], "rb"))
                c = shopctx.ShopCtx(self.a.split, b, self.bizs[b], d["C"], d.get("notes", ""), d.get("data_end"), self.rulebook_for(b),
                                    hide=self.hide, mode=getattr(self.a, "prompt", "compact"), escapes=self.escapes)
                c.hint_opts = {"counts_hints": bool(int(getattr(self.a, "hint_counts", 0) or 0))}
                if self.units_all:                                     # the units of this shop that run on its data
                    import units as unitsmod
                    ok = [u for u in unitsmod.for_shop(self.units_all, b) if unitsmod.verify(u, d["C"], d.get("data_end"), self.bizs[b].tables, months=1)[0]]
                    c.units = ok[:unitsmod.MAX_UNITS]
                    if c.units:
                        c.business = c.business + "\n\n" + unitsmod.block(c.units)
                        c.spec_u = list(self.spec) + unitsmod.spec_entries(c.units)
                    print(f"[hybrid] {b}: {len(c.units)} rule units verified", flush=True)
                if getattr(self.a, "grammar", 1): c.grammar(int(getattr(self.a, "max_enum", 5000)))
            except Exception as e:
                print(f"[hybrid] {b}: no context ({type(e).__name__}: {e})", flush=True); c = None
        self.ctxs[b] = c
        return c

    def gen2(self, conv, temp, seed, max_tokens, cb, think=None, prio=0, grammar=None, prefill=""):
        rid = self.llm.add(conv, temp, seed, max_tokens, think, prio, grammar, prefill) if prefill else \
            self.llm.add(conv, temp, seed, max_tokens, think, prio, grammar)
        self.cb[rid] = lambda text, ntok, rid=rid: cb(prefill + text, ntok, getattr(self.llm, "lp", {}).pop(rid, None))

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
        pure = bool(int(getattr(a, "pure", 0) or 0))                     # pure model (user 12/11): no rule-read hints anywhere
        h = {} if pure else ctx.hints(q["question"], q["as_of_date"])   # (prompt, retrieval, prechecks, flags, review)
        st["hints"] = h
        excl = set(self.excl) | {b}                                     # never the shop's own checked examples (honest on train shops)
        ws = float(getattr(a, "shot_schema", 0) or 0)                      # P5 plan T01: schema-matched examples
        if ws and getattr(ctx, "schema_sig", None) is None: ctx.schema_sig = retrieval.schema_sig(ctx.C)
        exs = self.bank.retrieve(q["question"], h, k=int(a.shots), exclude_bids=excl, exclude_ids={q["id"]},
                                 schema=getattr(ctx, "schema_sig", None) if ws else None, w_schema=ws)
        if self.shop_ex.get(b):                                             # v2 I3: the shop's own function examples replace the last
            ow = self.shop_ex[b].retrieve(q["question"], h, k=self.own_k, per_family=1)   # (least similar) training examples
            if ow: exs = exs[:max(0, len(exs) - len(ow))] + ow
        st["examples"] = [e.get("id") for e in exs]
        shown = h if getattr(a, "hints", 1) and not pure else None
        cards = self.lib.match(q["question"], bid=b, industry=ctx.ind[2] if ctx.ind else None, k=int(getattr(a, "lrl_k", 4) or 4),
                               rulebook=lrl.rb_key(ctx.rulebook)) if self.lib is not None else []
        if self.links is not None and q["id"] in self.links:            # W2: the rows the model linked, then the shop's value /
            other = [c for c in cards if c.get("kind") != "definition"][:2]   # lexicon cards (keyword-matched definitions dropped)
            cards = list(self.links[q["id"]]) + other; st["linked"] = len(self.links[q["id"]])
            self.stats["linked_q" if st["linked"] else "linked_none"] += 1
        elif self.links is not None: self.stats["link_missing"] += 1    # (a shard of the linker timed out: keyword cards)
        st["cards"] = cards; st["rule_ids"] = [c.get("id") for c in cards]
        rules = lrl.render(cards)
        st["msgs"] = {v: ctx.messages(q["question"], q["as_of_date"], exs, shown, v, rules) for v in set(self.variants) | {"direct"}}
        st["qkey"] = dm.key(h.get("clean") or q["question"])
        k1 = 1 if self.k_now < a.k else max(1, int(a.k_fast))
        for j in range(k1): self.sample(st, j, "direct", 0.0 if j == 0 else a.temp)

    def sample(self, st, j, variant, temp):
        a, q, ctx = self.a, st["q"], st["ctx"]
        conv = list(st["msgs"][variant]); smp = {"j": j, "variant": variant, "temp": temp, "turns": [], "ok": False}
        st["samples"].append(smp); st["pending"] += 1
        think = bool(getattr(a, "think", False)) or variant == "think"     # 'think' variant: round-2 samples reason first (the
        grammar = ctx.grammar(int(getattr(a, "max_enum", 5000))) if getattr(a, "grammar", 1) and not think else None   # hard ~15 %)
        mt = int(getattr(a, "think_tokens", 0) or a.max_tokens) if variant == "think" else int(a.max_tokens)
        def launch():
            seed = 1000 * len(smp["turns"]) + 17 * j + 1 + 7919 * int(getattr(a, "seed_base", 0) or 0)
            if smp.get("rescue"):                                       # think rescue: the same prompt, reasoning allowed
                self.gen2(conv, temp, seed, int(getattr(a, "think_tokens", 0) or 4096), on_text, True, st["prio"], None)
            else:                                                      # P2-02 alt: a plain sample starts inside '<plan>' (no
                pf = "<plan>" if int(getattr(a, "prefill_plan", 0) or 0) and not think else ""   # free-text reasoning first)
                self.gen2(conv, temp, seed, mt, on_text, think, st["prio"], grammar, pf)
        def on_text(text, ntok, lp):
            self.ntok += ntok; self.stats["gen"] += 1
            out = prompting.parse_output(text)
            if out["code"] is None and not think and not smp.get("rescue") and int(getattr(a, "think_rescue", 1) or 0) and ntok >= mt - 8:
                # the model reasoned in plain text until the token cap without a program (fine-tuned soups: 6-10 % of the
                # questions on the evaluation set): re-run the sample once in thinking mode (reasoning goes to the think block)
                smp["rescue"] = True; self.stats["think_rescue"] += 1
                smp["turns"].append({"text": text[-600:], "ntok": ntok, "code": None, "rescued": True})
                return launch()
            turn = {"text": text[-1200:], "ntok": ntok, "lp": lp, "code": out["code"], "escape": out["escape"], "plan": out["plan"]}
            smp["turns"].append(turn); conv.append({"role": "assistant", "content": text})
            code = out["code"]
            if out["escape"] == "sql" and code is not None:          # the query becomes a program first (a SQL text is no Python)
                code = sql_program(code, (st["hints"] or {}).get("answer_type"))
            uc = bool(int(getattr(a, "unit_check", 0) or 0)); cc = bool(int(getattr(a, "count_check", 0) or 0))
            su = bool(int(getattr(a, "size_use_check", 0) or 0))
            iss = pre.precheck(code, getattr(ctx, "spec_u", None) or self.spec, st["hints"], ctx.entity_values, out["escape"], q["as_of_date"],
                               q["question"] if (int(getattr(a, "size_check", 0) or 0) or uc or cc or su) else None, uc,
                               bool(int(getattr(a, "mask_check", 0) or 0)), cc, su,
                               getattr(ctx, "industry_doc", None) if int(getattr(a, "ticket_avg_check", 0) or 0) else None)
            turn["precheck"] = [f"{i['level']}:{i['rule']}:{i['msg'][:120]}" for i in iss]
            viol = lrl.Library.violations(st.get("cards") or [], q["question"], code) if int(getattr(a, "lrl_check", 1) or 0) else []
            if viol and len(smp["turns"]) <= int(a.max_turns):          # a learned check card flags the program: one repair turn
                turn["lrl_check"] = viol; self.stats["lrl_check"] += 1
                return retry("The program breaks these rules:\n" + "\n".join("- " + v for v in viol))
            if pre.errors(iss):
                self.stats["precheck_error"] += 1
                return retry("The program has problems:\n" + pre.message(iss))
            key = (q["id"], f"{j}.{len(smp['turns'])}")
            self.cpu.submit(agent.run_code, (key, st["b"], q["as_of_date"], code, a.timeout, st["can"]["path"]), a.timeout, on_exec)
        def on_exec(r):
            turn = smp["turns"][-1]
            turn["exec"] = {k: (str(v)[:400] if k in ("error", "stdout") else v) for k, v in r.items() if k in ("ok", "error", "answer", "reason")}
            if not r.get("ok"):
                self.stats["exec_error"] += 1
                return retry(f"The program failed:\n{str(r.get('error'))[:700]}" + repair_hint(r.get("error"), self.spec))
            k_asked = None if int(getattr(a, "pure", 0) or 0) else top_k_asked(st["qkey"])   # pure: no question-reading rule
            if int(getattr(a, "fmt_fix", 0) or 0):                     # P2: the wrapping of the hinted answer type ([5] -> 5)
                fixed = fmt_fix(r["answer"], st["hints"].get("answer_type"))
                if fixed is not r["answer"]: turn["fmt_fix"] = True; r = dict(r, answer=fixed)
            flags = toolsmod.postcheck(r["answer"], st["hints"].get("answer_type"), None, st["qkey"], k_asked)
            if int(getattr(a, "period_flag", 0) or 0) and any(str(p).startswith("warn:period") for p in turn.get("precheck") or []):
                flags = list(flags) + ["the program's period differs from the question's"]   # P2: such samples were right 2 / 23
            if int(getattr(a, "threshold_flag", 0) or 0) and any(str(p).startswith("warn:threshold") for p in turn.get("precheck") or []):
                flags = list(flags) + ["the threshold the question sets is not used"]       # P2: such samples were right 13 / 19
            if int(getattr(a, "named_periods_flag", 0) or 0) and any(str(p).startswith("warn:named_periods") for p in turn.get("precheck") or []):
                flags = list(flags) + ["a period the question names is not used"]           # P2: such samples were right 85 / 106
            if int(getattr(a, "review", 0) or 0) == 3 and smp.get("reviewed") and (st["hints"] or {}).get("missing_data") \
                    and (r["answer"] or {}).get("type") != "unanswerable":       # review 3: still an answer after the missing-data
                flags = list(flags) + ["answered although the hints say the data does not record it"]   # review (hint right 75 / 75)
            turn["post"] = flags; smp.update(ok=True, answer=r["answer"], flags=flags, lp=turn.get("lp"), escape=turn.get("escape"))
            rv = review_msg(st["hints"], r["answer"], flags, int(getattr(a, "review", 0) or 0)) if int(getattr(a, "review", 0) or 0) and not smp.get("reviewed") else None
            if rv:                                                     # M11: the model re-reads a suspicious result once (it decides;
                smp["reviewed"] = True; turn["review"] = rv[:300]      # the answer above stands if the review turn fails)
                self.stats["review"] += 1
                return retry(rv)
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

    def calibrator(self):
        """the logistic calibrator of --calibrator (tests/calibrate.py --out), loaded once; None = vote share."""
        if not hasattr(self, "_cal"):
            path = getattr(self.a, "calibrator", "") or ""
            if path and not os.path.isabs(path) and not os.path.isfile(path):
                path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
            import confidence
            self._cal = confidence.Calibrator.load(path) if path and os.path.isfile(path) else None
            if path and self._cal is None: print(f"[hybrid] calibrator {path} not found: vote share", flush=True)
        return self._cal

    def selector(self):
        """the learned selector of --selector (tests/train_selector.py --out), loaded once; None = vote / calibrator."""
        if not hasattr(self, "_sel"):
            path = getattr(self.a, "selector", "") or ""
            if path and not os.path.isabs(path) and not os.path.isfile(path):
                here = os.path.dirname(os.path.abspath(__file__))       # repo / image: <root>/src/refs/...; a Kaggle kernel's code
                rel = path[4:] if path.startswith("src/") else path    # stage has no src/ level: <stage>/refs/...
                for cand in (os.path.join(os.path.dirname(here), path), os.path.join(here, path), os.path.join(here, rel)):
                    if os.path.isfile(cand): path = cand; break
            import selector as _selmod
            if path and not os.path.isfile(path):                       # a missing selector silently became the plain vote in every
                raise SystemExit(f"[hybrid] --selector {path} not found")   # plan 4 run up to 14/11 00:50 (Kaggle stage layout)
            self._sel = _selmod.Selector.load(path) if path else None
            if self._sel is not None: print(f"[hybrid] learned selector {path} (theta {self._sel.theta})", flush=True)
        return self._sel

    def finish(self, st):
        a, q = self.a, st["q"]
        if q["id"] in self.written: return st["done"]()
        votes, rep = Counter(), {}
        for s in st["samples"]:
            if not s.get("ok"): continue
            c = agent.canon_of(s["answer"]); votes[c] += 0.5 if s.get("flags") else 1.0; rep.setdefault(c, s["answer"])
        n = max(1, len(st["samples"]))
        top = votes.most_common(1)[0][0] if votes else None
        log = {"hints": st.get("hints"), "examples": st.get("examples"), "rules": st.get("rule_ids") or None, "linked": st.get("linked"),
               "rule_text": [str(c.get("content"))[:160] + (" -> " + str(c["code_hint"]) if c.get("code_hint") else "") for c in (st.get("cards") or [])] or None, "stop": st.get("stop", "round2"), "error": st.get("error"),
               "best": rep[top] if votes else None,
               "samples": [{k: v for k, v in s.items()} for s in st["samples"]]}
        sel = self.selector()
        if votes and sel is not None:                                   # plan 4 V: learned selector chooses the group + abstains
            conf_vote = votes[top] / n; log["conf_vote"] = conf_vote
            final, conf = sel.decide({"samples": log}); conf = float(conf); log["conf_sel"] = conf
        elif votes:
            conf = votes[top] / n; log["conf_vote"] = conf
            cal = self.calibrator()
            if cal is not None:                                         # QĐ14: calibrated probability of the top answer (WP8)
                import confidence
                conf = float(cal.predict(confidence.matrix([{"samples": log}]))[0]); log["conf_cal"] = conf
            final = rep[top] if conf >= float(getattr(a, "theta_abstain", 0.5)) else {"type": "abstain"}
        else: conf, final = 0.0, {"type": "abstain"}
        self.stats["answered" if final.get("type") != "abstain" else "abstained"] += 1
        self.stats[f"stop_{st.get('stop', 'round2')}"] += 1
        agent.write_answer(self.sub, self.logf, a, q, final, conf, votes, log, {}, time.time() - st["t"])
        self.sub.flush(); self.logf.flush(); self.n_answered += 1; self.written.add(q["id"])
        st["done"]()

    def chunk_done(self, b, ci):
        super().chunk_done(b, ci)
        if self.done_chunks % 10 == 0: print(f"[hybrid] stats {dict(self.stats)}", flush=True)


_TOPK = re.compile(r"\btop\s*(\d{1,2})\b|(?<![/\d.,])\b(\d{1,2})\s+(?:mat hang|mon|san pham|sp|khach(?: hang)?|nha cung cap|ncc|nhan vien|tho|dong (?:may|xe)|xe|hang)\b(?![/\d])")


_ISO = re.compile(r"^\d{4}-\d{2}(?:-\d{2})?$")


def fmt_fix(ans, hint_type):
    """the answer in the format of the hinted answer type, when only the wrapping differs (rulebook 'format'): a one-number list
    for a 'number' question -> the number; an ISO date / month string for a 'date' question -> type 'date'. The value is
    never changed; any other answer is returned as it is."""
    if not isinstance(ans, dict) or not hint_type: return ans
    t, v = ans.get("type"), ans.get("answer")
    if hint_type == "number" and t == "list" and isinstance(v, list) and len(v) == 1 and isinstance(v[0], (int, float)) and not isinstance(v[0], bool):
        return {"type": "number", "answer": v[0]}
    if hint_type == "date" and t == "string" and isinstance(v, str) and _ISO.match(v.strip()):
        return {"type": "date", "answer": v.strip()}
    return ans


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


def review_msg(hints, answer, flags, mode=1):
    """the review turn of a suspicious result, or None. mode 1 (M11, --review 1): an answer although the hints say the data does
    not record what the question needs (dev: 32 / 32 such hints were right), or postcheck flags; mode 2 (--review 2): the
    missing-data case only (postcheck flags made the model change right answers: M11 A/B -2.5 on quick400)."""
    md = (hints or {}).get("missing_data") or []
    if md and (answer or {}).get("type") != "unanswerable":
        return ("Check before answering: the HINTS say the shop's data does not record " + "; ".join(map(str, md)) + ". If the question "
                "needs that, write raise Unanswerable(\"<reason>\") instead of ANSWER; if it does not, write the same program again.")
    if flags and mode == 1:                                             # (mode 3 = mode 2 + a sample still answering is flagged)
        return ("The result " + json.dumps(answer, ensure_ascii=False, default=str)[:200] + " looks suspicious: " + "; ".join(flags) +
                ". Check the program against the question (period, filters, function, answer type); if it is right, write the same program again.")
    return None


def sql_program(sql, expected_type=None):
    """the <sql> escape (WP12): a DuckDB query over the canonical tables as a program (src/sqlesc.py); the hints' answer type
    is the default type of the result (the query's answer column named after a type overrides it)."""
    return "ANSWER = sql_answer(" + json.dumps(sql, ensure_ascii=False) + (", " + json.dumps(expected_type) if expected_type else "") + ")"
