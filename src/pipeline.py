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
        st = {"conv": list(conv), "rnd": 0, "att": attempts if attempts is not None else [], "reflected": False, "completed": False}
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
            if complete is not None and not st["completed"] and st["rnd"] < self.a.repairs:
                msg = complete(r)
                if msg: st["completed"] = True; att["complete"] = True; return retry(msg)
            if reflect and st["rnd"] < self.a.repairs and not st["reflected"] and agent.suspicious(r["answer"]):
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
        def complete(r):
            un = [t for t in r.get("unused") or [] if len(self.bizs[b].tables[t])]
            if not un or not getattr(self.a, "complete_map", 1): return None
            desc = "; ".join(f"`{t}` ({len(self.bizs[b].tables[t])} rows: {', '.join(map(str, list(self.bizs[b].tables[t].columns)[:12]))})" for t in un)
            return (f"The mapping runs and passes the checks, but it does not use these raw tables: {desc}. If they hold information "
                    "questions may ask about (purchases, stock, lots, prices, payments, debts, deliveries, expenses, staff, customers, "
                    "items, conversions ...), add them to C under the canonical names; keep everything that already works. If none "
                    "of them is needed, output the same program again.")
        for s in range(self.a.kc):
            self.sample((b, s), msgs, s == 0, self.mt_map, agent.run_mapping,
                        lambda k, code: (k, k[0], code, None, os.path.join(self.cdir, f"{k[0]}.llm{k[1]}.pkl")),
                        lambda atts, s=s: fin(s, atts), think=True if self.a.think_map else None, prio=bs["prio"], complete=complete)

    def select(self, b, bs):
        cs, can = bs["cands"], None
        if cs:
            cs.sort(key=lambda x: (not x[0].startswith("ref:"), x[0]))  # deterministic; references before LLM samples (as in agent)
            _, scores = agent.agreement([r["sig"] for _, r in cs])
            order = sorted(range(len(cs)), key=lambda i: (-cs[i][1]["coverage"], -scores[i], i))
            src, r = cs[order[0]]; can = {"path": r["path"], "card": r["card"], "src": src, "unused": r.get("unused") or []}
            self.clog.write(json.dumps({"business_id": b, "chosen": src, "candidates": [[x, y["coverage"]] for x, y in cs]}, ensure_ascii=False) + "\n")
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
        for q in qs:
            st = {"att": {}, "rule": None, "pending": k + (1 if (a.qw > 0 and can) else 0), "t": time.time(), "w": {}}
            path = "L" if can else "B"
            def settle(q=q, st=st, path=path):
                st["pending"] -= 1                                       # st["settle"]: THIS question's settle (a bare name
                if st["pending"] == 0 and not self.escalate(q, st, k, path, can, b, prio, st["settle"]):   # would bind late)
                    self.write(q, st, k, path); self.inflight_q -= 1; left["n"] -= 1
                    if left["n"] == 0: self.chunk_done(b, ci)
            st["settle"] = settle
            if can:
                exs = self.ex.pick(q["question"], k=a.shots, exclude_business=b if self.exclude_own else None)
                msgs = [{"role": "system", "content": agent.SYSTEM_LIB},
                        {"role": "user", "content": agent.user_msg_lib(self.rulebook, self.bizs[b].readme, can["card"], fewshot.render_examples(exs), q)}]
            else:
                msgs = [{"role": "system", "content": agent.SYSTEM_RAW}, {"role": "user", "content": agent.user_msg(self.rulebook, self.bizs[b].readme, agent.CARDS[b], q)}]
            cpath = can["path"] if can else None
            for s_ in range(k):
                st["att"][s_] = []
                self.sample((q["id"], s_), msgs, s_ == 0, a.max_tokens, agent.run_code,
                            lambda kk, code, q=q, cpath=cpath: (kk, b, q["as_of_date"], code, a.timeout, cpath),
                            lambda atts, settle=settle: settle(), reflect=not a.no_reflect, prio=prio, attempts=st["att"][s_])
            if a.qw > 0 and can:
                def on_rule(r, st=st, settle=settle): st["rule"] = r; settle()
                self.cpu.submit(agent.run_qparse, ((q["id"], "Q"), b, q["as_of_date"], q["question"], cpath, a.timeout), a.timeout, on_rule)
        if not qs: self.chunk_done(b, ci)

    def vote(self, st):
        """weighted vote: rule program qw, every LLM sample its weight (1); conf = top / total weight of all voters."""
        a = self.a; votes, rep = Counter(), {}; rr = st["rule"]; total = sum(st["w"].get(s, 1.0) for s in st["att"])
        if rr and rr.get("ok"):
            c = agent.canon_of(rr["answer"]); votes[c] += a.qw; rep[c] = rr["answer"]; total += a.qw
        for s in sorted(st["att"]):
            r = agent.last_ok(st["att"][s])
            if r: c = agent.canon_of(r["answer"]); votes[c] += st["w"].get(s, 1.0); rep.setdefault(c, r["answer"])
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
            msgs_l = [{"role": "system", "content": agent.SYSTEM_LIB},
                      {"role": "user", "content": agent.user_msg_lib(self.rulebook, self.bizs[b].readme, can["card"], fewshot.render_examples(exs), q)}]
        msgs_b = [{"role": "system", "content": agent.SYSTEM_RAW}, {"role": "user", "content": agent.user_msg(self.rulebook, self.bizs[b].readme, agent.CARDS[b], q)}]
        cpath = can["path"] if can else None
        for e in range(a.escalate + raw):
            s = k + e; st["att"][s] = []; is_raw = e >= a.escalate or msgs_l is None
            st["w"][s] = 1.0
            self.sample((q["id"], s), msgs_b if is_raw else msgs_l, False, a.esc_tokens, agent.run_code,
                        lambda kk, code, cp=(None if is_raw else cpath): (kk, b, q["as_of_date"], code, a.timeout, cp),
                        lambda atts: settle(), think=True, reflect=False, prio=prio, attempts=st["att"][s])
        return True

    def write(self, q, st, k, path):
        a = self.a
        final, best, conf, votes = self.vote(st)
        samples = {(f"{s}{path}" if s < k else f"{s}E"): st["att"][s] for s in sorted(st["att"])}
        if st.get("esc"): samples["escalated"] = st["esc"]
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
