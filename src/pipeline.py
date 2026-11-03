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
import os, re, time, json, pickle
from collections import Counter, defaultdict, deque
import agent, industry, fewshot, dm, ingest, canon as canonmod


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

    THINK_END = "</think>"
    FORCE = "\n\nThe reasoning budget is used up, so I stop here and write the complete program now.\n" + THINK_END + "\n\n"

    def __init__(self, engine, think_cap=0):
        from vllm import SamplingParams
        self.e, self.SP, self.n, self.pending, self.unfinished = engine, SamplingParams, 0, {}, 0
        # budget forcing (s1, Muennighoff et al. 2025): a thinking request stops at think_cap tokens; when it is still
        # reasoning then (21 % of the first mapping attempts looped 'Wait, ...' to the token limit without code), the end
        # of the reasoning is appended and the same request continues with the rest of its budget for the program
        self.think_cap, self.req, self.cont = int(think_cap or 0), {}, {}
        core = getattr(engine.llm, "llm_engine", None)
        self.core = core if core is not None and hasattr(core, "add_request") and hasattr(core, "step") else None
        try:
            from vllm.sampling_params import RequestOutputKind
            self.final_only = RequestOutputKind.FINAL_ONLY
        except Exception: self.final_only = None
        self.prio_ok = True
        print(f"[pipe] engine interface: {'add_request/step' if self.core is not None else 'batched generate (fallback)'}", flush=True)

    @staticmethod
    def cut_loop(text, min_len=12, reps=3):
        """the reasoning up to where it starts repeating itself ('Wait, ... map_status handles it.' x 40): a line of at
        least min_len characters seen `reps` times is cut at its second occurrence, with everything after it."""
        lines, seen = text.split("\n"), {}
        for i, l in enumerate(lines):
            k = l.strip()
            if len(k) < min_len: continue
            seen.setdefault(k, []).append(i)
            if len(seen[k]) >= reps: return "\n".join(lines[:seen[k][1]])
        return text

    def _sp(self, temp, seed, max_tokens, think=False):
        kw = dict(temperature=temp, top_p=1.0 if temp == 0 else 0.95, max_tokens=max_tokens, seed=seed)
        if think:                                                       # Qwen3.5 thinking: no greedy decoding (it loops),
            kw.update(temperature=temp if temp > 0 else 0.6, top_p=0.95, top_k=20)   # top-k 20 (model card); seeded
        if self.final_only is not None and self.core is not None: kw["output_kind"] = self.final_only
        return self.SP(**kw)

    def _add_core(self, rid, prompt, sp, priority):
        kw = {"lora_request": self.e.lora} if self.e.lora is not None else {}
        if self.prio_ok:
            try: self.core.add_request(rid, prompt, sp, priority=priority, **kw); self.unfinished += 1; return
            except TypeError: self.prio_ok = False
            except ValueError as ex:                                    # FCFS scheduler refuses priorities
                if "priority" not in str(ex).lower(): raise
                self.prio_ok = False
        self.core.add_request(rid, prompt, sp, **kw); self.unfinished += 1

    def add(self, conv, temp, seed, max_tokens, think=None, priority=0):
        prompt, mt = self.e.fit(conv, max_tokens, think)
        self.n += 1; rid = f"r{self.n}"
        if prompt is None: self.pending[rid] = None; return rid        # too long: finishes empty at the next step
        cap = self.think_cap if think and self.think_cap and self.core is not None and mt >= self.think_cap + 1024 else 0
        sp = self._sp(temp, seed, cap or mt, bool(think))
        if self.core is None: self.pending[rid] = (prompt, sp); return rid
        if cap: self.req[rid] = (prompt, temp, seed, mt, priority)
        self._add_core(rid, prompt, sp, priority)
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
                if not getattr(o, "finished", False): continue
                self.unfinished -= 1; rid, out = o.request_id, o.outputs[0]; text, n = out.text, len(out.token_ids)
                if rid in self.cont:                                    # the continuation of a capped request: both parts
                    rid0, head, n0 = self.cont.pop(rid); done.append((rid0, head + text, n0 + n)); continue
                r = self.req.pop(rid, None)
                if r is not None and getattr(out, "finish_reason", None) == "length" and r[3] - n >= 512:
                    prompt, temp, seed, mt, prio = r                    # cut at the cap: continue (inside the reasoning:
                    head = text if self.THINK_END in text else self.cut_loop(text) + self.FORCE   # end it first) with the rest
                    self.cont[rid + "c"] = (rid, head, n)
                    self._add_core(rid + "c", prompt + head, self._sp(temp, seed + 1, mt - n, True), prio); continue
                done.append((rid, text, n))
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


def claim(claim_dir, name):
    """atomic claim of a work unit (business mapping `b`, question chunk `b.c<i>`) by one GPU process; the file holds
    the owner's pid. False when another process has it."""
    if not claim_dir: return True
    try: fd = os.open(os.path.join(claim_dir, name), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError: return False
    os.write(fd, str(os.getpid()).encode()); os.close(fd); return True


def owner_alive(claim_dir, name):
    try: pid = int(open(os.path.join(claim_dir, name)).read().strip() or 0)
    except (OSError, ValueError): return True                             # being written: assume alive
    if not pid: return True
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False
    except PermissionError: return True


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
        self.C = max(1, getattr(a, "steal_chunk", 16) or 16)             # questions per stealable chunk
        self.chunks = {b: [qs_[i:i + self.C] for i in range(0, len(qs_), self.C)] for b, qs_ in self.by_b.items()}
        self.mapping, self.maps, self.own = set(), {}, []                 # mapping in progress here; b -> can (None = raw path)
        self.started, self.taken, self.full = set(), set(), set()        # businesses mapped here; chunks seen taken; fully claimed
        self.mine, self.others = set(), set()                             # chunks claimed by THIS process; businesses mapped elsewhere
        self.inflight_q, self.done_chunks = 0, 0
        self.norms = {}                                                  # b -> question Normalizer
        self.vocabs = {}                                                 # b -> item name / code key -> the item's name keys
        self.t0, self.ntok, self.n_answered, self.n_q = time.time(), 0, 0, len(qs)
        self.mt_map = a.compile_tokens * (3 if a.think_map else 1); self.written = set()
        self.k_now = a.k
        # one absolute deadline for every GPU process (set by run_dp from the container start), else from now
        self.deadline = getattr(a, "deadline", 0) or ((self.t0 + a.budget) if a.budget else 0)
        self.n_procs = max(1, getattr(a, "claim_procs", 1) or 1)

    # -- plumbing
    def gen(self, conv, temp, seed, max_tokens, cb, think=None, prio=0):
        self.cb[self.llm.add(conv, temp, seed, max_tokens, think, prio)] = cb

    def sample(self, key, conv, greedy, max_tokens, runner, make_task, finish, think=None, reflect=False, prio=0, attempts=None,
               complete=None):
        """one sample = generation + execution + up to a.repairs repair rounds; finish(attempts) when it is settled.
        complete(r) -> message or None: one extra round asking a mapping to cover the raw tables it left unused (the last
        ok attempt is kept, so this round can only add)."""
        st = {"conv": list(conv), "rnd": 0, "att": attempts if attempts is not None else [], "reflected": False, "completed": False,
              "think": think, "mt": max_tokens}
        is_map = runner is agent.run_mapping
        reps = int(getattr(self.a, "map_repairs", -1)) if is_map and int(getattr(self.a, "map_repairs", -1)) >= 0 else self.a.repairs
        def launch():
            temp = 0.0 if greedy else self.a.temp
            seed = 1000 * st["rnd"] + 17 * key[1] + 1 + 7919 * int(getattr(self.a, "seed_base", 0) or 0)   # --seed-base: another run
            self.gen(st["conv"], temp, seed, st["mt"], on_text, st["think"], prio)
        def retry(msg):
            if st["rnd"] < reps:
                if is_map and st["think"] and not getattr(self.a, "map_repair_think", 1):
                    st["think"] = False; st["mt"] = self.a.compile_tokens   # --map-repair-think 0: a repair is a local fix
                st["conv"].append({"role": "user", "content": msg}); st["rnd"] += 1; launch()
            else: finish(st["att"])
        def on_text(text, ntok):
            self.ntok += ntok
            st["conv"].append({"role": "assistant", "content": text}); st["forced"] = StepLLM.FORCE in text
            code = agent.extract_code(text)
            if code is None:
                st["att"].append({"round": st["rnd"], "ok": False, "error": "no python code block", "gen": text[-400:], **({"forced": True} if st["forced"] else {})})
                if st["think"]:                                          # the reasoning ran out of tokens (23 of 203 mapping
                    st["think"] = False                                  # attempts of v4b): the retry writes the program
                    return retry("No ```python block was found: the reasoning used up the token budget. Do not reason further; "
                                 "output only the complete program in one ```python block.")
                return retry("No ```python block was found. Output only the complete program in one ```python block.")
            sp = None
            if st["rnd"] > 0:                                            # a repair that sent only the fixed statements
                prev = next((x["code"] for x in reversed(st["att"]) if x.get("code")), None)
                sp = agent.splice(prev, code) if prev else None
                if sp: code = sp
            task = make_task(key, code)
            self.cpu.submit(runner, task, (self.a.timeout * (1 + len(task[6]) if len(task) > 6 else self.a.timeout)) if runner is agent.run_code else 300,
                            lambda r, code=code, sp=bool(sp): on_exec(code, r, sp))
        def on_exec(code, r, spliced=False):
            att = {"round": st["rnd"], "code": code, **{x: r[x] for x in r if x not in ("t", "card")}, "exec_s": round(r.get("t", 0.0), 2)}
            if spliced: att["spliced"] = True
            if st.get("forced"): att["forced"] = True                    # the reasoning was ended by budget forcing
            if "card" in r: att["card"] = r["card"]
            st["att"].append(att)
            if not r.get("ok"):
                return retry(f"The program failed:\n{r.get('error')}\nstdout:\n{r.get('stdout', '')[-600:]}\nFix it and output the full corrected program.")
            if complete is not None and not st["completed"] and st["rnd"] < reps:
                msg = complete(r)
                if msg: st["completed"] = True; att["complete"] = True; return retry(msg)
            if reflect and st["rnd"] < reps and not st["reflected"] and agent.suspicious(r["answer"]):
                st["reflected"] = True; att["reflect"] = True
                return retry(f"The program returned {json.dumps(r['answer'], ensure_ascii=False)}. A zero or empty result often means "
                             "a wrong filter (status labels, date range, name spelling: use match_names, unit). Re-check each filter "
                             "against the actual values. If the result is really correct, output the same program again.")
            finish(st["att"])
        launch()

    # -- one business: transfer + mapping (exactly one process), then its chunks (any process)
    def start(self, b):
        self.started.add(b); self.mapping.add(b); prio = self.order.index(b)
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

    REF_GAP_W = 5.0                                                     # ranking points a reference loses when it misses README rules

    def adapt_target(self, b, bs):
        """(src, result, rid, gaps) of the best transferred reference when it misses conventions of b's README, else None."""
        refs = [(src, r) for src, r in bs["cands"] if src.startswith("ref:") and not src.startswith("ref:fmt:")]
        if not refs or not int(getattr(self.a, "ref_adapt", 0) or 0): return None
        src, r = max(refs, key=lambda x: canonmod.rank_score(x[1])); rid = src[4:]
        ref = agent.ref_readme(rid)
        if ref is None or rid not in self.progs: return None
        gaps = agent.readme_gaps(self.bizs[b].readme, ref)
        return (src, r, rid, gaps) if gaps else None

    def adapt(self, b, bs, target):
        """--ref-adapt LLM rewrites of the reference program that follow the README conventions it misses; they compete with it
        in the choice (the unadapted reference loses REF_GAP_W points)."""
        src, r, rid, gaps = target; r["gaps"] = len(gaps)
        code = self.progs[rid]
        for name, tgt in (r.get("tables_used") or {}).items():             # the reference's table names -> this shop's
            code = code.replace(f'T["{name}"]', f'T["{tgt}"]').replace(f"T['{name}']", f"T['{tgt}']")
        print(f"  {b}: {src} misses {len(gaps)} README convention(s); {self.a.ref_adapt} rewrites", flush=True)
        msgs = [{"role": "system", "content": agent.SYSTEM_MAP},
                {"role": "user", "content": agent.user_msg_adapt(self.rb_map, self.bizs[b].readme, agent.CARDS[b], code, gaps)}]
        n = int(self.a.ref_adapt); bs["pending"] = n; atts_all = {}
        def fin(s, atts):
            atts_all[s] = atts; r2 = agent.last_ok(atts)
            if r2: bs["cands"].append((f"adapt:{rid}:{s}", r2))
            bs["pending"] -= 1
            if bs["pending"] == 0:
                self.clog.write(json.dumps({"business_id": b, "adapt_of": src, "gaps": gaps, "adapt_attempts": atts_all}, ensure_ascii=False, default=str) + "\n")
                self.select(b, bs)
        for s in range(n):
            self.sample((b, 100 + s), msgs, s == 0, self.mt_map, agent.run_mapping,
                        lambda k, code: (k, k[0], code, None, os.path.join(self.cdir, f"{k[0]}.adapt{k[1]}.pkl")),
                        lambda atts, s=s: fin(s, atts), think=True if self.a.think_map else None, prio=bs["prio"])

    def after_transfer(self, b, bs):
        if not (self.a.map_llm or not bs["cands"]):
            t = self.adapt_target(b, bs)
            return self.adapt(b, bs, t) if t else self.select(b, bs)
        msgs = [{"role": "system", "content": agent.SYSTEM_MAP},
                {"role": "user", "content": agent.user_msg_map(self.rb_map, self.bizs[b].readme, agent.CARDS[b], agent.map_examples(self.bizs[b], b, self.exclude_own),
                                                               industry_doc=self.industry_doc(b, "canon"))}]
        bs["pending"] = self.a.kc
        def fin(s, atts):
            bs["llm"][s] = atts; r = agent.last_ok(atts)
            if r: bs["cands"].append((f"llm:{s}", r))
            bs["pending"] -= 1
            if bs["pending"] == 0:
                if len(bs["cands"]) < max(1, int(getattr(self.a, "map_min_cands", 1) or 1)) and getattr(self.a, "map_retry", 1) and not bs.get("retried"):
                    # every sample failed (biz_123 in v9, dev_014 in 3 of 4 public runs: luck of the sampling), or only one ran
                    # (biz_123 / biz_124 in v13: nothing to choose from, no fallback): one more batch
                    bs["retried"] = True; bs["pending"] = self.a.kc
                    print(f"  {b}: {len(bs['cands'])} mapping candidate(s); a second batch of {self.a.kc}", flush=True)
                    for s2 in range(self.a.kc, 2 * self.a.kc): launch(s2)
                    return
                self.clog.write(json.dumps({"business_id": b, "llm_attempts": {x: bs["llm"][x] for x in sorted(bs["llm"])}}, ensure_ascii=False, default=str) + "\n")
                self.select(b, bs)
        def complete(r):
            """one extra round for a mapping that runs: the soft checks it fails (canon.secondary_checks) and the raw
            tables it leaves unused."""
            if not getattr(self.a, "complete_map", 1): return None
            un = [t for t in r.get("unused") or [] if len(self.bizs[b].tables[t])]
            warns = r.get("warnings") or []
            if not un and not warns: return None
            parts = []
            if warns: parts.append("The mapping runs, but these checks of its content fail:\n- " + "\n- ".join(warns))
            if un:
                desc = "; ".join(f"`{t}` ({len(self.bizs[b].tables[t])} rows: {', '.join(map(str, list(self.bizs[b].tables[t].columns)[:12]))})" for t in un)
                parts.append(f"It does not use these raw tables: {desc}. If they hold information questions may ask about (purchases, "
                             "stock, lots, prices, payments, debts, deliveries, expenses, staff, customers, items, conversions ...), add "
                             "them to C under the canonical names.")
            return ("\n\n".join(parts) + "\n\nFix this and output the complete program; keep everything that already works. If nothing "
                    "needs to change, output the same program again.")
        def launch(s):
            self.sample((b, s), msgs, s == 0, self.mt_map, agent.run_mapping,
                        lambda k, code: (k, k[0], code, None, os.path.join(self.cdir, f"{k[0]}.llm{k[1]}.pkl")),
                        lambda atts, s=s: fin(s, atts), think=True if self.a.think_map else None, prio=bs["prio"], complete=complete)
        for s in range(self.a.kc): launch(s)

    SEL_FAIL_W = 30.0                                                   # ranking points per unit of rule failure share

    @staticmethod
    def answer_agreement(answers):
        """[{qid: rule answer | None} per candidate] -> per candidate, the share of questions with a majority answer (given
        by >= 2 candidates) where the candidate gives that answer: an outlier mapping disagrees with the others."""
        ag, nq = [0.0] * len(answers), 0
        for q in set().union(*[set(a) for a in answers]) if answers else ():
            v = Counter(a[q] for a in answers if a.get(q) is not None)
            if not v: continue
            top, k = v.most_common(1)[0]
            if k < 2: continue
            nq += 1
            for i, a in enumerate(answers):
                if a.get(q) == top: ag[i] += 1
        return [x / nq for x in ag] if nq else ag

    def rule_questions(self, b):
        """(qid, as_of, text the rule reader parses) for every question of b; the cleaned question is kept on q."""
        nz = self.normalizer(b) if getattr(self.a, "qclean", 1) else None
        out = []
        for ch in self.chunks.get(b, []):
            for q in ch:
                if nz is not None and "_clean" not in q:
                    try: q["_clean"] = nz.clean(q["question"], q["as_of_date"])
                    except Exception: q["_clean"] = None
                out.append((q["id"], q["as_of_date"], agent.rule_text(q)))
        return out

    def select(self, b, bs):
        """with >= 2 candidates: first the rule programs of b's questions on each of them (CPU), then the choice."""
        cs = bs["cands"]
        if len(cs) >= 2 and getattr(self.a, "select_rules", 1) and self.a.qw > 0:
            qs = self.rule_questions(b)
            if qs:
                bs["evals"] = len(cs)
                def on_eval(res, d):
                    d["fail_share"] = res.get("fail_share", 0.0) if res.get("ok") else 0.0
                    d["rule_answers"] = res.get("answers") or {} if res.get("ok") else {}
                    bs["evals"] -= 1
                    if bs["evals"] == 0: self.choose(b, bs)
                for src, r in cs:
                    self.cpu.submit(agent.run_rule_eval, ((b, src), b, r["path"], qs, 150), 300, lambda res, d=r: on_eval(res, d))
                return
        self.choose(b, bs)

    def choose(self, b, bs):
        cs, can = bs["cands"], None
        if cs:
            cs.sort(key=lambda x: (not x[0].startswith("ref:"), x[0]))  # deterministic; references before LLM samples (as in agent)
            _, scores = agent.agreement([r["sig"] for _, r in cs])
            ag = self.answer_agreement([r.get("rule_answers") or {} for _, r in cs]); W = float(getattr(self.a, "sel_agree", 0.0) or 0.0)
            for (_, r), x in zip(cs, ag): r["agree"] = x
            order = sorted(range(len(cs)), key=lambda i: (-(canonmod.rank_score(cs[i][1]) - self.SEL_FAIL_W * cs[i][1].get("fail_share", 0.0) + W * ag[i]
                                                            - (self.REF_GAP_W if cs[i][1].get("gaps") else 0.0)),
                                                          -scores[i], i))   # coverage - warnings - rule failures + answer agreement
            src, r = cs[order[0]]; can = {"path": r["path"], "card": r["card"], "src": src, "unused": r.get("unused") or []}
            M = max(1, int(getattr(self.a, "map_vote", 1) or 1), int(getattr(self.a, "map_fallback", 1) or 1))   # next ranked candidates
            if M > 1: can["alts"] = [cs[i][1]["path"] for i in order[1:M]]; can["alt_src"] = [cs[i][0] for i in order[1:M]]
            self.clog.write(json.dumps({"business_id": b, "chosen": src, "candidates": [[x, y["coverage"], len(y.get("warnings") or []),
                                                                                        round(y.get("fail_share", -1.0), 3), round(y.get("agree", 0.0), 3)] for x, y in cs],
                                        "chosen_penalty": round(sum(canonmod.warning_penalty(w) for w in r.get("warnings") or []), 1),   # vote.py: a split
                                        "chosen_warnings": [str(w)[:120] for w in r.get("warnings") or []]},                             # goes to the cleanest run
                                       ensure_ascii=False) + "\n")
            print(f"  {b}: {len(cs)} candidates, chose {src} coverage {r['coverage']:.0f} ({time.time() - bs['t']:.0f}s)", flush=True)
        else: print(f"  {b}: no canonical tables ({time.time() - bs['t']:.0f}s)", flush=True)
        self.clog.flush(); self.maps[b] = can; self.mapping.discard(b); self.own.append(b)
        if self.claim_dir:                                               # publish: other GPU processes may now take chunks
            tmp = os.path.join(self.claim_dir, f".{b}.map.{os.getpid()}")
            json.dump(can if can else {"none": True}, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
            os.replace(tmp, os.path.join(self.claim_dir, f"{b}.map"))

    def mapping_of(self, b):
        """(known, can): the mapping chosen for b, here or by another process (map file)."""
        if b in self.maps: return True, self.maps[b]
        if not self.claim_dir: return False, None
        f = os.path.join(self.claim_dir, f"{b}.map")
        if not os.path.isfile(f): return False, None
        m = json.load(open(f, encoding="utf-8")); self.maps[b] = None if m.get("none") else m
        return True, self.maps[b]

    def unused_card(self, b, can):
        """cards of the raw tables the chosen mapping left out (dev_014: instalments, trade-ins, warranty visits answered
        'unanswerable' because only C was shown, though programs can read T). Cached per business."""
        if not hasattr(self, "_ucard"): self._ucard = {}
        if b not in self._ucard:
            un = [t for t in (can.get("unused") or []) if t in self.bizs[b].tables and len(self.bizs[b].tables[t])][:8]
            try: cards = [ingest.table_card(t, self.bizs[b].tables[t]) for t in un]
            except Exception: cards = []
            self._ucard[b] = ("\n\n# RAW TABLES NOT IN C (in T, original column names)\n" + "\n\n".join(cards)) if cards else ""
        return self._ucard[b]

    def industry(self, b):
        """(title, body, new) of b's rulebook industry section, or None (agent.industry_of; cached)."""
        if not hasattr(self, "_ind"): self._ind = {}
        if b not in self._ind:
            try: self._ind[b] = agent.industry_of(self.rulebook, self.bizs[b].readme)
            except Exception: self._ind[b] = None
            if self._ind[b] is None and getattr(self.a, "industry_kit", 1):   # the README does not name it: the data shows it
                try:
                    t = industry.detect(self.bizs[b]); secs = agent.rulebook_sections(self.rulebook)
                    t = next((x for x in secs if t and dm.key(x) == dm.key(t)), None)
                    if t:
                        try: train = {dm.key(x) for x in agent.rulebook_sections(open(agent.TRAIN_RULEBOOK, encoding="utf-8").read())}
                        except OSError: train = set()
                        self._ind[b] = (t, secs[t], dm.key(t) not in train)
                except Exception as e: print(f"[pipe] {b}: industry detection failed ({type(e).__name__}: {e})", flush=True)
            if self._ind[b]: print(f"[pipe] {b}: industry section '{self._ind[b][0]}'" + (" (not in the training rulebook)" if self._ind[b][2] else ""), flush=True)
        return self._ind[b]

    def industry_doc(self, b, kind):
        """the canonical tables ('canon') / toolkit ('api') of b's rulebook industry when src/industry.py implements it, else ''."""
        if not getattr(self.a, "industry_kit", 1): return ""
        ind = self.industry(b); k = industry.industry_key(ind[0]) if ind else None
        return (industry.INDUSTRY_CANON if kind == "canon" else industry.INDUSTRY_API).get(k, "") if k else ""

    def readme_for(self, b, toolkit=True):
        """README of b + its rulebook industry section (+ the industry toolkit for programs over C; raw-table programs have no
        toolkit). Cached."""
        if not hasattr(self, "_rdm"): self._rdm = {}
        if (b, toolkit) not in self._rdm:
            ind = self.industry(b); api = self.industry_doc(b, "api") if toolkit else ""
            self._rdm[(b, toolkit)] = self.bizs[b].readme + (f"\n\n# RULES OF THIS SHOP'S INDUSTRY (rulebook section '{ind[0]}': they define the "
                                                             f"terms of its questions and take precedence)\n{ind[1]}" if ind else "") + \
                                      (f"\n\n# TOOLKIT FOR THIS INDUSTRY (implements these rules; prefer it to raw-table code)\n{api}" if api else "")
        return self._rdm[(b, toolkit)]

    def normalizer(self, b):
        """question cleaning for business b (vocabulary: rulebook + training questions of OTHER businesses + b's names)."""
        if b not in self.norms:
            import qnorm
            try: self.norms[b] = qnorm.Normalizer.for_split(self.a.split, exclude_business=b, extra_texts=qnorm.Normalizer.entity_texts(self.bizs[b]))
            except Exception as e: print(f"[pipe] {b}: question cleaning unavailable ({type(e).__name__}: {e})", flush=True); self.norms[b] = None
        return self.norms[b]

    def answer_chunk(self, b, ci, can):
        a, k = self.a, self.k_now; prio = self.order.index(b)
        qs = self.chunks[b][ci]; left = {"n": len(qs)}; self.inflight_q += len(qs)
        nz = self.normalizer(b) if getattr(a, "qclean", 1) else None
        for q in qs:
            if nz is not None and "_clean" not in q:
                try: q["_clean"] = nz.clean(q["question"], q["as_of_date"])
                except Exception: q["_clean"] = None
        ind = self.industry(b); new_ind = bool(ind and ind[2] and getattr(a, "industry_raw", 0))
        readme = self.readme_for(b)
        n_raw = int(getattr(a, "industry_raw", 0) or 0) if new_ind and can else 0
        for q in qs:
            confirm = bool(getattr(a, "confirm", 0)) and a.qw > 0 and bool(can) and k > 1 and not new_ind
            # --confirm: phase 1 = rule program + greedy sample; phase 2 = the other k-1 samples, only when phase 1 disagrees.
            # A shop of a new industry: the rule reader does not know its definitions -> weight 1, every sample, raw-table samples
            st = {"att": {}, "rule": None, "pending": (1 if confirm else k) + (1 if (a.qw > 0 and can) else 0) + n_raw, "t": time.time(), "w": {},
                  "phase": 1 if confirm else 2, "q": q, "b": b if can else None}
            if new_ind: st["qw"] = min(1.0, a.qw)
            path = "L" if can else "B"
            def settle(q=q, st=st, path=path):
                st["pending"] -= 1                                       # st["settle"]: THIS question's settle (a bare name
                if st["pending"] > 0: return                             # would bind late)
                if st["phase"] == 1 and not self.confirmed(st):          # rule and greedy sample disagree: the other samples
                    st["phase"] = 2; st["pending"] = k - 1; st["launch"](1, k); return
                if not self.escalate(q, st, k, path, can, b, prio, st["settle"]):
                    self.write(q, st, k, path); self.inflight_q -= 1; left["n"] -= 1
                    if left["n"] == 0: self.chunk_done(b, ci)
            st["settle"] = settle
            if can:
                exs = self.ex.pick(q["question"], k=a.shots, exclude_business=b if self.exclude_own else None)
                msgs = [{"role": "system", "content": agent.SYSTEM_LIB},
                        {"role": "user", "content": agent.user_msg_lib(self.rulebook, readme, can["card"] + self.unused_card(b, can), fewshot.render_examples(exs), q)}]
            else:
                msgs = [{"role": "system", "content": agent.SYSTEM_RAW}, {"role": "user", "content": agent.user_msg(self.rulebook, self.readme_for(b, False), agent.CARDS[b], q)}]
            if n_raw:
                msgs_raw = [{"role": "system", "content": agent.SYSTEM_RAW}, {"role": "user", "content": agent.user_msg(self.rulebook, self.readme_for(b, False), agent.CARDS[b], q)}]
                for j in range(n_raw):
                    s_ = 50 + j; st["att"][s_] = []; st["w"][s_] = 1.0
                    self.sample((q["id"], s_), msgs_raw, False, max(a.max_tokens, 2048), agent.run_code,
                                lambda kk, code, q=q: (kk, b, q["as_of_date"], code, a.timeout, None),
                                lambda atts, settle=settle: settle(), reflect=False, prio=prio, attempts=st["att"][s_])
            cpath = can["path"] if can else None
            alts = tuple(can.get("alts") or ()) if can else ()
            def launch(lo, hi, q=q, st=st, msgs=msgs, cpath=cpath, settle=settle):
                for s_ in range(lo, hi):
                    st["att"][s_] = []
                    self.sample((q["id"], s_), msgs, s_ == 0, a.max_tokens, agent.run_code,
                                lambda kk, code, q=q, cpath=cpath: (kk, b, q["as_of_date"], code, a.timeout, cpath, alts),
                                lambda atts, settle=settle: settle(), reflect=not a.no_reflect, prio=prio, attempts=st["att"][s_])
            st["launch"] = launch
            launch(0, 1 if confirm else k)
            if a.qw > 0 and can:
                def on_rule(r, st=st, settle=settle):
                    st["rule"] = r
                    if st["phase"] == 1 and not (r.get("ok") and r.get("answer")):   # no rule answer: the other samples now
                        st["phase"] = 2; st["pending"] += k - 1; st["launch"](1, k)
                    settle()
                self.cpu.submit(agent.run_qparse, ((q["id"], "Q"), b, q["as_of_date"], agent.rule_text(q), cpath, a.timeout, alts),
                                a.timeout * (1 + len(alts)), on_rule)
        if not qs: self.chunk_done(b, ci)

    @staticmethod
    def confirmed(st):
        """phase 1 of --confirm: the rule program and the greedy sample give the same answer."""
        rr = st["rule"]; g = agent.last_ok(st["att"].get(0) or [])
        return bool(rr and rr.get("ok") and rr.get("answer") and g and g.get("answer") is not None
                    and agent.canon_of(g["answer"]) == agent.canon_of(rr["answer"]))

    _UNIT = r"(?:ml|l|g|kg|mg|cm|mm|m|m2|m3|gr|lit|oz|cc|w|kw|v)\b"
    GEN = set("hop chai goi vi vien lon ly cai tuyp ml g kg mg l thung bao loc x va size m s".split())

    @classmethod
    def _ck(cls, s): return re.sub(r"(?<=\d) (?=" + cls._UNIT + ")", "", dm.key(s))   # 'chai 30 ml' = 'chai 30ml'

    def vocab(self, b):
        """item name / code key -> name keys of that item, from the mapping chosen for b (grounding check)."""
        if b in self.vocabs: return self.vocabs[b]
        V = {}
        try:
            known, can = self.mapping_of(b)
            C = pickle.load(open(can["path"], "rb"))["C"] if known and can and can.get("path") else {}
            for t in ("lines", "items"):
                df = C.get(t)
                if df is None or "item_name" not in df.columns: continue
                sub = df[[c for c in ("item_code", "item_name") if c in df.columns]].drop_duplicates()
                for cd, nm in zip(sub["item_code"] if "item_code" in sub.columns else [None] * len(sub), sub["item_name"]):
                    if nm is None or (isinstance(nm, float) and nm != nm): continue
                    nk = self._ck(nm)
                    if len(nk) >= 3: V.setdefault(nk, set()).add(nk)
                    if cd is not None and not (isinstance(cd, float) and cd != cd) and len(self._ck(cd)) >= 3: V.setdefault(self._ck(cd), set()).add(nk)
        except Exception as e: print(f"[pipe] {b}: no item vocabulary for the grounding check ({type(e).__name__}: {e})", flush=True)
        self.vocabs[b] = V; return V

    @classmethod
    def _mentioned(cls, text, nk):
        """the item name (or 3 consecutive words of it, not only units / numbers) occurs in the question text."""
        if re.search(r"(^|\s)" + re.escape(nk) + r"(\s|$)", text): return True
        w = nk.split()
        return any(any(x not in cls.GEN and not re.search(r"\d", x) for x in w[i:i + 3])
                   and re.search(r"(^|\s)" + re.escape(" ".join(w[i:i + 3])) + r"(\s|$)", text) for i in range(len(w) - 2))

    def grounded(self, q, code, b):
        """False when an LLM program names an item of the shop that the question does not mention ('siro tiffy chai 30ml'
        answered with 'Tiffy Dey hộp 25 vỉ'): such samples were wrong 85 % of the time on train (99 of 116, v6-v9), the
        others 7 %. Such a sample gets no vote and is left out of the total."""
        if b is None or not code: return True
        V = self.vocab(b)
        if not V: return True
        used = [V[u] for u in (self._ck(x or y) for x, y in re.findall(r'"([^"\n]{3,80})"|\'([^\'\n]{3,80})\'', code)) if u in V]
        if not used: return True
        c = q.get("_clean") if isinstance(q.get("_clean"), dict) else {}
        texts = [self._ck(q["question"])] + ([self._ck(c["clean"])] if c.get("clean") else [])
        return all(any(self._mentioned(t, nk) for t in texts for nk in names) for names in used)

    def vote(self, st):
        """the vote on the chosen mapping; --map-fallback M: when it abstains or says unanswerable, the vote of the same
        programs on the next ranked mapping that answers (replay of v11 train: PA 0.967 -> 0.974; --map-vote, every mapping
        voting at once, was worse: 0.942)."""
        mv = max(1, int(getattr(self.a, "map_vote", 1) or 1))
        res = self.vote_on(st, None if mv > 1 else 0)
        st["fallback"] = None
        if mv <= 1 and res[0].get("type") in ("abstain", "unanswerable"):
            for j in range(1, max(1, int(getattr(self.a, "map_fallback", 1) or 1))):
                r2 = self.vote_on(st, j)
                if r2[0].get("type") not in ("abstain", "unanswerable"): st["fallback"] = j; return r2
        if res[0].get("type") == "unanswerable" and getattr(self.a, "unans_unused", 1) and st.get("b"):
            known, can = self.mapping_of(st["b"])                        # 'not in the data' from an LLM mapping that left raw
            if known and can and str(can.get("src", "")).startswith("llm") and can.get("unused"):   # tables unused: the data may
                st["unans_abstain"] = True; return ({"type": "abstain"},) + tuple(res[1:])         # hold it (v6-v13: 16 right / 42 wrong)
        return res

    NEW_RULES = {"orders_by_trips", "delivered_qty", "due_in", "due_after"}   # (with every 'ind_*' template of qparse.parse_industry)
    NEW_RULE_W = 2.0                                                    # their vote: more than one LLM sample, less than a checked rule

    def vote_on(self, st, idx):
        """weighted vote: rule program qw, every LLM sample its weight (1); conf = top / total weight of all voters. An LLM
        sample naming an item the question does not mention (grounded() False) neither votes nor counts. idx: the answers
        on that mapping (0 = chosen, j = j-th alternate); None = every mapping an equal share (--map-vote)."""
        a = self.a; votes, rep = Counter(), {}; rr = st["rule"]; total = sum(st["w"].get(s, 1.0) for s in st["att"])
        def add(r, w):
            ans = [r.get("answer")] + list(r.get("alt") or [])
            if idx is not None: ans = [ans[idx] if idx < len(ans) else None]
            for x in ans:
                if x is not None: c = agent.canon_of(x); votes[c] += w / len(ans); rep.setdefault(c, x)
        ralt = (rr.get("alt") or []) if rr else []
        qw = st.get("qw", a.qw)
        tpl = str((rr or {}).get("tpl") or "")
        if a.qw > 0 and (tpl.startswith("ind_") or tpl in self.NEW_RULES): qw = self.NEW_RULE_W   # templates no train label checked
        if a.qw > 0 and tpl == "subjective": qw = a.qw                   # a quality no data records: in every shop (§7)
        if rr and (rr.get("ok") if idx in (0, None) else (idx - 1 < len(ralt) and ralt[idx - 1] is not None)):
            if not rr.get("ok"): rr = {"answer": None, "alt": ralt}      # failed on the chosen mapping, runs on alternate idx
            add(rr, qw); total += qw
        st["ungrounded"] = []
        for s in sorted(st["att"]):
            r = agent.last_ok(st["att"][s])
            if r and st.get("q") is not None and not self.grounded(st["q"], r.get("code"), st.get("b")):
                total -= st["w"].get(s, 1.0); st["ungrounded"].append(s); continue
            if r: add(r, st["w"].get(s, 1.0))
        if votes and total:
            top, n = votes.most_common(1)[0]; conf = n / total
            final = rep[top] if conf >= a.tau else {"type": "abstain"}
            best = rep[top]
        else: conf, final, best = 0.0, {"type": "abstain"}, {"type": "abstain"}
        return final, best, conf, votes

    def escalate(self, q, st, k, path, can, b, prio, settle):
        """post-hoc check (user rule): a low-confidence, abstained, unanswerable or flagged answer gets --escalate extra
        samples WITH thinking (+ raw-table programs when it is unanswerable and the mapping left tables unused); the
        question is decided on the vote of all samples. Returns True when samples were launched."""
        a = self.a
        if not getattr(a, "escalate", 0) or st.get("esc") or self.k_now < a.k: return False   # never in fast mode
        final, best, conf, _ = self.vote(st)
        c = q.get("_clean") or {}
        flags = agent.answer_flags(c.get("key") or q["question"].lower(), best, c.get("period"))
        if conf >= a.esc_conf and not flags: return False
        st["esc"] = flags or [f"conf {conf:.2f}"]
        raw = a.esc_raw if (best.get("type") == "unanswerable" and can and can.get("unused")) else 0
        st["pending"] = a.escalate + raw
        msgs_l = None
        if can:
            exs = self.ex.pick(q["question"], k=a.shots, exclude_business=b if self.exclude_own else None)
            msgs_l = [{"role": "system", "content": agent.SYSTEM_LIB},           # same context as the first samples (industry
                      {"role": "user", "content": agent.user_msg_lib(self.rulebook, self.readme_for(b), can["card"] + self.unused_card(b, can),   # rules,
                                                                     fewshot.render_examples(exs), q)}]                                      # raw tables)
        msgs_b = [{"role": "system", "content": agent.SYSTEM_RAW}, {"role": "user", "content": agent.user_msg(self.rulebook, self.readme_for(b, False), agent.CARDS[b], q)}]
        cpath = can["path"] if can else None
        alts = tuple(can.get("alts") or ()) if can else ()
        for e in range(a.escalate + raw):
            s = k + e; st["att"][s] = []; is_raw = e >= a.escalate or msgs_l is None
            st["w"][s] = 1.0
            self.sample((q["id"], s), msgs_b if is_raw else msgs_l, False, a.esc_tokens, agent.run_code,
                        lambda kk, code, cp=(None if is_raw else cpath), al=(() if is_raw else alts): (kk, b, q["as_of_date"], code, a.timeout, cp, al),
                        lambda atts: settle(), think=True, reflect=False, prio=prio, attempts=st["att"][s])
        return True

    def write(self, q, st, k, path):
        a = self.a
        final, best, conf, votes = self.vote(st)
        samples = {(f"{s}{path}" if s < k else f"{s}E"): st["att"][s] for s in sorted(st["att"])}
        if st.get("esc"): samples["escalated"] = st["esc"]
        if st.get("ungrounded"): samples["ungrounded"] = st["ungrounded"]
        if st.get("fallback"): samples["fallback"] = st["fallback"]          # answered on the j-th alternate mapping
        if st.get("unans_abstain"): samples["unans_abstain"] = True
        rr = st["rule"]
        if rr: samples["rule"] = rr
        if q["id"] in self.written: return                              # already abstained at the hard time limit
        agent.write_answer(self.sub, self.logf, a, q, final, conf, votes, samples, {}, time.time() - st["t"])
        self.sub.flush(); self.logf.flush(); self.n_answered += 1; self.written.add(q["id"])

    def chunk_done(self, b, ci):
        self.done_chunks += 1; el = time.time() - self.t0
        print(f"[pipe] {b} chunk {ci + 1}/{len(self.chunks[b])} done | {self.n_answered} questions here, {self.ntok} tok in {el:.0f}s "
              f"({self.ntok / max(el, 1e-9):.0f} tok/s)", flush=True)

    # -- work units
    def next_business(self):
        """claim the next unmapped business (most questions first)."""
        for b in self.order:
            if b in self.started or b in self.others: continue
            if claim(self.claim_dir, b): return b
            self.others.add(b)
        return None

    def takeover(self):
        """a business claimed by a process that died before publishing its mapping is mapped here (idle check only)."""
        if not self.claim_dir: return None
        for b in self.others:
            if b in self.started or os.path.isfile(os.path.join(self.claim_dir, f"{b}.map")): continue
            if not owner_alive(self.claim_dir, b) and claim(self.claim_dir, f"{b}.takeover"):
                print(f"[pipe] {b}: mapping owner died, taking over", flush=True); self.start(b); return b
        return None

    def next_chunk(self):
        """the next unclaimed question chunk of a mapped business: own businesses first (warm prefix), then others'."""
        for b in self.own + [x for x in self.order if x not in self.own]:
            if b in self.full: continue
            known, can = self.mapping_of(b)
            if not known: continue
            free = False
            for ci in range(len(self.chunks[b])):
                if (b, ci) in self.taken: continue
                if claim(self.claim_dir, f"{b}.c{ci}"): self.taken.add((b, ci)); self.mine.add((b, ci)); return b, ci, can
                self.taken.add((b, ci))                                  # someone else has it
            if not free: self.full.add(b)
        return None

    def unclaimed_units(self):
        """(businesses not mapped by anyone, chunks nobody holds) according to the claim files."""
        if not self.claim_dir:
            bs = [b for b in self.order if b not in self.started]
            return bs, [(b, ci) for b in self.order for ci in range(len(self.chunks[b])) if (b, ci) not in self.taken]
        names = set(os.listdir(self.claim_dir))
        return [b for b in self.order if b not in names], [(b, ci) for b in self.order for ci in range(len(self.chunks[b])) if f"{b}.c{ci}" not in names]

    def abstain_qs(self, qs):
        for q in qs:
            if q["id"] in self.written: continue
            agent.write_answer(self.sub, self.logf, self.a, q, {"type": "abstain"}, 0.0, Counter(), {}, {}, 0.0); self.written.add(q["id"])
        self.sub.flush(); self.logf.flush()

    def claim_rest(self):
        """claim every chunk nobody holds and abstain it (each question is written by exactly one process)."""
        _, chunks = self.unclaimed_units(); n = 0
        for b, ci in chunks:
            if claim(self.claim_dir, f"{b}.c{ci}"): self.taken.add((b, ci)); self.mine.add((b, ci)); self.abstain_qs(self.chunks[b][ci]); n += 1
        return n

    def fill(self):
        """keep one business mapping ahead (two when the GPU queue runs low) and --q-inflight questions answering."""
        while len(self.mapping) < (2 if self.llm.busy() < self.a.low_water else 1):
            b = self.next_business()
            if b is None: break
            self.start(b)
        while self.inflight_q < self.a.q_inflight:
            ch = self.next_chunk()
            if ch is None: break
            self.answer_chunk(*ch)

    def run(self):
        a = self.a; span = (self.deadline - self.t0) if self.deadline else 0; stopped = False; last_check = last_fill = last_est = 0.0
        while True:
            now = time.time(); left_s = (self.deadline - now) if self.deadline else None
            if self.deadline and left_s <= 0.01 * span:                 # hard limit: every unwritten question is abstained
                self.claim_rest(); self.abstain_qs([q for (b, ci) in self.mine for q in self.chunks[b][ci]])   # own chunks only
                print("[pipe] time budget reached", flush=True); break
            if self.deadline and left_s <= 0.05 * span:
                if not stopped: self.claim_rest(); stopped = True       # out of time: nothing new is started
            else:
                if self.deadline and self.n_answered and now - last_est > 5:    # fast mode (K=1) when this process's share of
                    last_est = now; ub, uc = self.unclaimed_units()               # the remaining work would not fit in time
                    mine = self.inflight_q + sum(len(self.chunks[b][ci]) for b, ci in uc) / self.n_procs
                    self.k_now = 1 if (now - self.t0) / self.n_answered * mine > 0.9 * left_s else a.k
                if now - last_fill > 0.1: last_fill = now; self.fill()
            moved = self.cpu.poll()
            if self.llm.busy():
                for rid, text, ntok in self.llm.step(): self.cb.pop(rid)(text, ntok); moved += 1
            elif not moved and not self.cpu.busy() and not self.mapping and self.inflight_q == 0:
                if now - last_check > 0.5:                              # idle: finished only when every unit is claimed
                    last_check = now; ub, uc = self.unclaimed_units()
                    if not ub and not uc: break
                    self.fill(); self.takeover()                         # waits for mappings of other processes, or takes over
                time.sleep(0.05)
            elif not moved: time.sleep(0.005)
        print(f"[pipe] DONE {self.done_chunks} chunks, {self.n_answered} questions here, {self.ntok} tok in {time.time() - self.t0:.0f}s", flush=True)
