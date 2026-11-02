#!/usr/bin/env python3
"""Pipelined v3 scheduler: GPU generation, CPU program execution and repair rounds overlap, businesses advance
independently, every GPU process pulls the next business from a shared queue.

Why: the round-based path (agent.generate_and_run) generates a whole round, then waits for the CPU to execute every
program, then generates the repairs: the GPU idles during execution and waits for the longest request of each round,
and a static business split leaves GPUs idle at the end. Here:
- requests go to vLLM's engine one by one (LLMEngine.add_request / step); finished requests are executed on the CPU
  pool asynchronously while the GPU keeps decoding; a repair is queued the moment its program fails;
- each business is a small state machine: reference transfer (CPU) -> LLM mapping samples (+ repairs) -> selection ->
  K answer samples per question (+ repairs / reflection) and the rule-based program -> vote -> answer written at once
  (Rule 8: a time-out keeps every finished answer);
- a business is admitted while fewer than `--active` businesses are in the answer stage or the GPU queue runs low, so
  the requests in flight share one or two prefixes (prefix cache hits) while the next business is already mapping;
- with `--claim-dir` (set by agent.run_dp), the GPU processes claim businesses in a shared order (most questions first)
  by atomic file creation: a GPU that finishes early takes the next business instead of idling.
Prompts, seeds, temperatures, repair messages and voting are those of the round-based path, so runs stay comparable."""
import os, time, json
from collections import Counter, defaultdict, deque
import agent, fewshot, canon as canonmod


def bind(mod):
    """use the agent module that is actually running (agent.py may run as __main__: its globals, e.g. CARDS and the
    worker context of the CPU pool, live in that module, not in a fresh `import agent`)."""
    global agent
    agent = mod


# ------------------------------------------------------------------ engines with a request / step interface
class StepLLM:
    """vLLM request-level interface: add(conv, ...) -> rid; step() -> [(rid, text, n_tokens)] of finished requests.
    Uses LLMEngine.add_request / step when the installed vLLM exposes them; otherwise each step() runs every pending
    request in one blocking generate() call (same results, no overlap)."""

    def __init__(self, engine):
        from vllm import SamplingParams
        self.e, self.SP, self.n, self.pending, self.unfinished = engine, SamplingParams, 0, {}, 0
        core = getattr(engine.llm, "llm_engine", None)
        self.core = core if core is not None and hasattr(core, "add_request") and hasattr(core, "step") else None
        try:
            from vllm.sampling_params import RequestOutputKind
            self.final_only = RequestOutputKind.FINAL_ONLY
        except Exception: self.final_only = None
        self.prio_ok = True
        print(f"[pipe] engine interface: {'add_request/step' if self.core is not None else 'batched generate (fallback)'}", flush=True)

    def _sp(self, temp, seed, max_tokens):
        kw = dict(temperature=temp, top_p=1.0 if temp == 0 else 0.95, max_tokens=max_tokens, seed=seed)
        if self.final_only is not None and self.core is not None: kw["output_kind"] = self.final_only
        return self.SP(**kw)

    def add(self, conv, temp, seed, max_tokens, think=None, priority=0):
        prompt, mt = self.e.fit(conv, max_tokens, think)
        self.n += 1; rid = f"r{self.n}"
        if prompt is None: self.pending[rid] = None; return rid        # too long: finishes empty at the next step
        sp = self._sp(temp, seed, mt)
        if self.core is None: self.pending[rid] = (prompt, sp); return rid
        kw = {"lora_request": self.e.lora} if self.e.lora is not None else {}
        if self.prio_ok:
            try: self.core.add_request(rid, prompt, sp, priority=priority, **kw); self.unfinished += 1; return rid
            except TypeError: self.prio_ok = False
            except ValueError as ex:                                    # FCFS scheduler refuses priorities
                if "priority" not in str(ex).lower(): raise
                self.prio_ok = False
        self.core.add_request(rid, prompt, sp, **kw); self.unfinished += 1
        return rid

    def busy(self): return self.unfinished + len(self.pending)

    def step(self):
        done = [(rid, "", 0) for rid, v in list(self.pending.items()) if v is None]
        for rid, _, _ in done: self.pending.pop(rid)
        if self.core is None:
            if self.pending:
                rids = list(self.pending); outs = self.e.llm.generate([self.pending[r][0] for r in rids], [self.pending[r][1] for r in rids],
                                                                    use_tqdm=False, lora_request=self.e.lora)
                done += [(r, o.outputs[0].text, len(o.outputs[0].token_ids)) for r, o in zip(rids, outs)]; self.pending.clear()
            return done
        if self.unfinished:
            for o in self.core.step():
                if getattr(o, "finished", False):
                    self.unfinished -= 1; done.append((o.request_id, o.outputs[0].text, len(o.outputs[0].token_ids)))
        return done


class FakeStepLLM:
    """tests without a GPU: responder(conv) -> text; requests finish one step after they are added."""
    def __init__(self, responder): self.r, self.n, self.q = responder, 0, []
    def add(self, conv, temp, seed, max_tokens, think=None, priority=0):
        self.n += 1; rid = f"r{self.n}"; self.q.append((rid, conv)); return rid
    def busy(self): return len(self.q)
    def step(self):
        out, self.q = [(rid, self.r(c), 10) for rid, c in self.q[:7]], self.q[7:]   # partial batches: exercises overlap
        return out


# ------------------------------------------------------------------ CPU tasks with deadlines
class CPU:
    """CPU tasks on the shared pool with callbacks. A task counts as lost (dead / hung worker) only when it has been
    RUNNING longer than its own alarm limit + slack: the worker writes its start time into agent._START[slot]
    (agent._timed), so time spent waiting in a busy queue never triggers a false time-out. Without the shared slots
    (should not happen) a generous limit from submission is used."""
    def __init__(self, pool, slack=120):
        self.pool, self.slack, self.ready, self.live, self.n = pool, slack, deque(), {}, 0
        self.start = agent._START

    def submit(self, func, task, deadline_s, cb):
        self.n += 1; tok = self.n; slot = tok % agent.START_SLOTS
        if self.start is not None: self.start[slot] = 0.0
        self.live[tok] = (slot, deadline_s + self.slack, time.time(), cb)
        self.pool.apply_async(agent._timed, ((slot, func, task),), callback=lambda res, t=tok: self.ready.append((t, res)),
                              error_callback=lambda ex, t=tok: self.ready.append((t, (None, {"ok": False, "error": f"{type(ex).__name__}: {ex}"[:300], "t": 0.0}))))

    def poll(self):
        n = 0
        while self.ready:
            tok, (_, r) = self.ready.popleft()
            if tok in self.live: cb = self.live.pop(tok)[3]; cb(r); n += 1
        now = time.time(); lost = []
        for tok, (slot, lim, t_sub, cb) in self.live.items():
            started = self.start[slot] if self.start is not None else 0.0
            if (started > 0 and now - started > lim) or (self.start is None and now - t_sub > lim + 1800): lost.append(tok)
        for tok in lost:                                                 # dead / hung worker: report a time-out, ignore it later
            cb = self.live.pop(tok)[3]; cb({"ok": False, "error": "timeout: the program ran too long (worker did not return)", "t": 0.0}); n += 1
        return n

    def busy(self): return len(self.live)


def claim(claim_dir, bid):
    """atomic claim of a business by one GPU process (False when another process has it)."""
    if not claim_dir: return True
    try: os.close(os.open(os.path.join(claim_dir, bid), os.O_CREAT | os.O_EXCL | os.O_WRONLY)); return True
    except FileExistsError: return False


# ------------------------------------------------------------------ the scheduler
class Pipeline:
    def __init__(self, a, llm, pool, rulebook, bizs, qs, sub, logf, clog, claim_dir=""):
        self.a, self.llm, self.cpu = a, llm, CPU(pool)
        self.rulebook, self.rb_map, self.bizs = rulebook, agent.rulebook_extract(rulebook), bizs
        self.sub, self.logf, self.clog, self.claim_dir = sub, logf, clog, claim_dir
        self.exclude_own = os.path.isfile(os.path.join(a.split, "answers.jsonl"))
        self.progs, self.meta = canonmod.ref_programs(), canonmod.load_meta()
        self.ex = fewshot.AnswerExamples()
        self.by_b = defaultdict(list)
        for q in qs: self.by_b[q["business_id"]].append(q)
        self.order = sorted(self.by_b, key=lambda b: (-len(self.by_b[b]), b))  # most questions first: short tail
        self.cdir = os.path.join(a.cache or a.out, "canon2_pipe"); os.makedirs(self.cdir, exist_ok=True)
        self.cb = {}                                                     # rid -> callback(text, ntok)
        self.active, self.answering, self.done_b = set(), set(), []
        self.t0, self.ntok, self.n_answered, self.n_q = time.time(), 0, 0, len(qs)
        self.mt_map = a.compile_tokens * (3 if a.think_map else 1); self.written = set()
        self.k_now = a.k
        # one absolute deadline for every GPU process (set by run_dp from the container start), else from now
        self.deadline = getattr(a, "deadline", 0) or ((self.t0 + a.budget) if a.budget else 0)
        self.n_procs = max(1, getattr(a, "claim_procs", 1) or 1)

    # -- plumbing
    def gen(self, conv, temp, seed, max_tokens, cb, think=None, prio=0):
        self.cb[self.llm.add(conv, temp, seed, max_tokens, think, prio)] = cb

    def sample(self, key, conv, greedy, max_tokens, runner, make_task, finish, think=None, reflect=False, prio=0, attempts=None):
        """one sample = generation + execution + up to a.repairs repair rounds; finish(attempts) when it is settled."""
        st = {"conv": list(conv), "rnd": 0, "att": attempts if attempts is not None else [], "reflected": False}
        def launch():
            temp = 0.0 if greedy else self.a.temp
            self.gen(st["conv"], temp, 1000 * st["rnd"] + 17 * key[1] + 1, max_tokens, on_text, think, prio)
        def retry(msg):
            if st["rnd"] < self.a.repairs:
                st["conv"].append({"role": "user", "content": msg}); st["rnd"] += 1; launch()
            else: finish(st["att"])
        def on_text(text, ntok):
            self.ntok += ntok
            st["conv"].append({"role": "assistant", "content": text})
            code = agent.extract_code(text)
            if code is None:
                st["att"].append({"round": st["rnd"], "ok": False, "error": "no python code block", "gen": text[-400:]})
                return retry("No ```python block was found. Output only the complete program in one ```python block.")
            self.cpu.submit(runner, make_task(key, code), self.a.timeout if runner is agent.run_code else 300,
                            lambda r, code=code: on_exec(code, r))
        def on_exec(code, r):
            att = {"round": st["rnd"], "code": code, **{x: r[x] for x in r if x not in ("t", "card")}, "exec_s": round(r.get("t", 0.0), 2)}
            if "card" in r: att["card"] = r["card"]
            st["att"].append(att)
            if not r.get("ok"):
                return retry(f"The program failed:\n{r.get('error')}\nstdout:\n{r.get('stdout', '')[-600:]}\nFix it and output the full corrected program.")
            if reflect and st["rnd"] < self.a.repairs and not st["reflected"] and agent.suspicious(r["answer"]):
                st["reflected"] = True; att["reflect"] = True
                return retry(f"The program returned {json.dumps(r['answer'], ensure_ascii=False)}. A zero or empty result often means "
                             "a wrong filter (status labels, date range, name spelling: use match_names, unit). Re-check each filter "
                             "against the actual values. If the result is really correct, output the same program again.")
            finish(st["att"])
        launch()

    # -- one business
    def start(self, b):
        self.active.add(b); prio = self.order.index(b)
        biz = self.bizs[b]; bs = {"cands": [], "pending": 0, "llm": {}, "prio": prio, "t": time.time()}
        tasks = [((b, rid), b, code, rid, os.path.join(self.cdir, f"{b}.ref_{rid.replace(':', '_')}.pkl")) for rid, code in self.progs.items()
                 if not (self.exclude_own and rid == b) and canonmod.resolvable(biz, "generic" if rid.startswith("fmt:") else self.meta.get(rid, {}))]
        bs["pending"] = len(tasks)
        def on_ref(rid, r):
            if r.get("ok"): bs["cands"].append(("ref:" + rid, r))
            bs["pending"] -= 1
            if bs["pending"] == 0: self.after_transfer(b, bs)
        for t in tasks: self.cpu.submit(agent.run_mapping, t, 300, lambda r, rid=t[3]: on_ref(rid, r))
        if not tasks: self.after_transfer(b, bs)

    def after_transfer(self, b, bs):
        if not (self.a.map_llm or not bs["cands"]):
            return self.select(b, bs)
        msgs = [{"role": "system", "content": agent.SYSTEM_MAP},
                {"role": "user", "content": agent.user_msg_map(self.rb_map, self.bizs[b].readme, agent.CARDS[b], agent.map_examples(self.bizs[b], b, self.exclude_own))}]
        bs["pending"] = self.a.kc
        def fin(s, atts):
            bs["llm"][s] = atts; r = agent.last_ok(atts)
            if r: bs["cands"].append((f"llm:{s}", r))
            bs["pending"] -= 1
            if bs["pending"] == 0:
                self.clog.write(json.dumps({"business_id": b, "llm_attempts": {x: bs["llm"][x] for x in range(self.a.kc)}}, ensure_ascii=False, default=str) + "\n")
                self.select(b, bs)
        for s in range(self.a.kc):
            self.sample((b, s), msgs, s == 0, self.mt_map, agent.run_mapping,
                        lambda k, code: (k, k[0], code, None, os.path.join(self.cdir, f"{k[0]}.llm{k[1]}.pkl")),
                        lambda atts, s=s: fin(s, atts), think=True if self.a.think_map else None, prio=bs["prio"])

    def select(self, b, bs):
        cs, can = bs["cands"], None
        if cs:
            cs.sort(key=lambda x: (not x[0].startswith("ref:"), x[0]))  # deterministic; references before LLM samples (as in agent)
            _, scores = agent.agreement([r["sig"] for _, r in cs])
            order = sorted(range(len(cs)), key=lambda i: (-cs[i][1]["coverage"], -scores[i], i))
            src, can = cs[order[0]]
            self.clog.write(json.dumps({"business_id": b, "chosen": src, "candidates": [[x, y["coverage"]] for x, y in cs]}, ensure_ascii=False) + "\n")
            print(f"  {b}: {len(cs)} candidates, chose {src} coverage {can['coverage']:.0f} ({time.time() - bs['t']:.0f}s)", flush=True)
        else: print(f"  {b}: no canonical tables ({time.time() - bs['t']:.0f}s)", flush=True)
        self.clog.flush(); self.answering.add(b); self.answer(b, can, bs["prio"])

    def answer(self, b, can, prio):
        a, k = self.a, self.k_now
        qs = self.by_b[b]; left = {"n": len(qs)}
        for q in qs:
            st = {"att": {}, "rule": None, "pending": k + (1 if (a.qw > 0 and can) else 0), "t": time.time()}
            path = "L" if can else "B"
            def settle(q=q, st=st, path=path):
                st["pending"] -= 1
                if st["pending"] == 0:
                    self.write(q, st, k, path); left["n"] -= 1
                    if left["n"] == 0: self.finish_b(b)
            if can:
                exs = self.ex.pick(q["question"], k=a.shots, exclude_business=b if self.exclude_own else None)
                msgs = [{"role": "system", "content": agent.SYSTEM_LIB},
                        {"role": "user", "content": agent.user_msg_lib(self.rulebook, self.bizs[b].readme, can["card"], fewshot.render_examples(exs), q)}]
            else:
                msgs = [{"role": "system", "content": agent.SYSTEM_RAW}, {"role": "user", "content": agent.user_msg(self.rulebook, self.bizs[b].readme, agent.CARDS[b], q)}]
            cpath = can["path"] if can else None
            for s in range(k):
                st["att"][s] = []
                self.sample((q["id"], s), msgs, s == 0, a.max_tokens, agent.run_code,
                            lambda kk, code, q=q, cpath=cpath: (kk, b, q["as_of_date"], code, a.timeout, cpath),
                            lambda atts, settle=settle: settle(), reflect=not a.no_reflect, prio=prio, attempts=st["att"][s])
            if a.qw > 0 and can:
                def on_rule(r, st=st, settle=settle): st["rule"] = r; settle()
                self.cpu.submit(agent.run_qparse, ((q["id"], "Q"), b, q["as_of_date"], q["question"], cpath, a.timeout), a.timeout, on_rule)
        if not qs: self.finish_b(b)

    def write(self, q, st, k, path):
        a = self.a; votes, rep = Counter(), {}; rr = st["rule"]; total = float(k)
        if rr and rr.get("ok"):
            c = agent.canon_of(rr["answer"]); votes[c] += a.qw; rep[c] = rr["answer"]; total += a.qw
        for s in range(k):
            r = agent.last_ok(st["att"][s])
            if r: c = agent.canon_of(r["answer"]); votes[c] += 1.0; rep.setdefault(c, r["answer"])
        if votes and total:
            top, n = votes.most_common(1)[0]; conf = n / total
            final = rep[top] if conf >= a.tau else {"type": "abstain"}
        else: conf, final = 0.0, {"type": "abstain"}
        samples = {f"{s}{path}": st["att"][s] for s in range(k)}
        if rr: samples["rule"] = rr
        if q["id"] in self.written: return                              # already abstained at the hard time limit
        agent.write_answer(self.sub, self.logf, a, q, final, conf, votes, samples, {}, time.time() - st["t"])
        self.sub.flush(); self.logf.flush(); self.n_answered += 1; self.written.add(q["id"])

    def finish_b(self, b):
        self.active.discard(b); self.answering.discard(b); self.done_b.append(b)
        el = time.time() - self.t0
        print(f"[pipe] {b} done | {len(self.done_b)} businesses, {self.n_answered}/{self.n_q} questions, {self.ntok} tok in {el:.0f}s "
              f"({self.ntok / max(el, 1e-9):.0f} tok/s)", flush=True)

    # -- admission and the main loop
    def next_business(self):
        for b in self.order:
            if b in self.active or b in self.done_b or b in self.skipped: continue
            if claim(self.claim_dir, b): return b
            self.skipped.add(b)                                          # another GPU process has it
        return None

    def abstain_rest(self, bids):
        for b in bids:
            for q in self.by_b[b]:
                if q["id"] in self.written: continue
                agent.write_answer(self.sub, self.logf, self.a, q, {"type": "abstain"}, 0.0, Counter(), {}, {}, 0.0); self.written.add(q["id"])
        self.sub.flush(); self.logf.flush()

    def want_more(self):
        """one business prefetching (transfer / mapping) while at most --active answer; more when the GPU queue runs low."""
        n_pre = len(self.active) - len(self.answering)
        if n_pre < 1 and len(self.active) < self.a.active + 1: return True
        return self.llm.busy() < self.a.low_water and len(self.active) < self.a.active + 2

    def unclaimed(self):
        return [b for b in self.order if b not in self.active and b not in self.done_b and b not in self.skipped
                and not (self.claim_dir and os.path.exists(os.path.join(self.claim_dir, b)))]

    def claim_rest(self):
        """claim every business nobody has started (so that each is abstained exactly once)."""
        rest = [b for b in self.unclaimed() if claim(self.claim_dir, b)]
        self.abstain_rest(rest); self.done_b.extend(rest); return rest

    def run(self):
        a = self.a; self.skipped = set(); exhausted = False
        span = (self.deadline - self.t0) if self.deadline else 0
        while True:
            now = time.time(); left_s = (self.deadline - now) if self.deadline else None
            if self.deadline and left_s <= 0.01 * span:                 # hard limit: every unwritten question is abstained
                self.claim_rest(); self.abstain_rest([b for b in self.order if b in self.active])
                print("[pipe] time budget reached", flush=True); break
            if not exhausted and (not self.deadline or left_s > 0.05 * span):
                if self.deadline and self.n_answered:                   # fast mode (K=1) when this process's share of the
                    mine = sum(len(self.by_b[b]) for b in self.active) + sum(len(self.by_b[b]) for b in self.unclaimed()) / self.n_procs
                    per_q = (now - self.t0) / self.n_answered           # remaining work would not fit before the deadline
                    self.k_now = 1 if per_q * mine > 0.9 * left_s else a.k
                while self.want_more():
                    b = self.next_business()
                    if b is None: exhausted = True; break
                    self.start(b)
            elif self.deadline and left_s <= 0.05 * span and not exhausted:   # out of time: nothing new is started
                self.claim_rest(); exhausted = True
            moved = self.cpu.poll()
            if self.llm.busy():
                for rid, text, ntok in self.llm.step(): self.cb.pop(rid)(text, ntok); moved += 1
            elif not moved:
                if not self.cpu.busy() and not self.active and exhausted: break
                time.sleep(0.01)
        print(f"[pipe] DONE {len(self.done_b)} businesses, {self.n_answered} questions, {self.ntok} tok in {time.time() - self.t0:.0f}s", flush=True)
