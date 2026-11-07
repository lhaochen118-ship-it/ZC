# Tests (CPU, no GPU) — run after every change; all must PASS

```bash
python3 tests/test_ingest_robust.py
```

```bash
python3 tests/test_engine_batching.py
```

```bash
python3 tests/test_agent_v2.py data/raw/train/train <scratch>
```

```bash
python3 tests/eval_lib.py data/raw/train/train <cache> --show 3
```

`eval_lib.py` = toolkit + rule-based programs with each business's reference mapping (oracle mapping): 2651 ok / 51 wrong /
298 unparsed on train (2026-11-02). `agent.py <split> <out> --no-llm --cache <cache>` = same without the own reference
(leave-one-out): train PA 0.506, 15/30 businesses mapped by transfer.

```bash
python3 tests/test_pipeline.py data/raw/train/train <scratch>
```

`test_pipeline.py`: the pipelined scheduler gives the same answers as the round-based path (fake engine finishing 7
requests per step: generation / execution / repairs interleave), claims are exclusive. `test_engine_batching.py` also
checks the vLLM request-level adapter (`StepLLM`) against a mock `LLMEngine`.

```bash
python3 tests/test_sandbox.py
```

`test_sandbox.py`: a generated `except Exception:` cannot swallow the time-out; a late SIGALRM is ignored; `pmap`
turns a crashing / lost task into an error result (the 02/11 Kaggle crash: a stray time-out aborted 2 of 4 GPU shards).

`bash tests/run_all.sh` runs the whole CPU suite. Before any Kaggle launch, `automation/preflight.py "<jobs>"` must print
`PREFLIGHT OK` (fanout.py enforces it).

```
python3 tests/test_mapping_fixes.py
```

`test_mapping_fixes.py` (no data files): a code block indented as a whole runs (dedent); a repair reply holding only the
fixed statements is spliced into the previous program (also `C["x"] = ...` after `C = ...`); a `KeyError` on a name
gets the closest real table / column names; `fillna(None)` is a no-op; the soft checks (`canon.secondary_checks`) flag
a revenue that differs from a "Tiền hàng" column and lots without lot sales, and nothing on a right mapping; the
ranking puts a wrong revenue below any candidate; stock comes from the lots when there is no snapshot.
`test_plausibility.py` also requires zero soft warnings for every reference mapping on its own business.

## Rule-first stack (2026-11-05)

```bash
python3 tests/test_rules_first.py data/raw/train/train <scratch>
```

`test_rules_first.py`: `src/rules_first.py` (stage 1 of `docker/predict.sh`: automap + qnorm + qparse + toolkit, no
model) on 60 training questions + 2 unreadable ones: every question ends answered xor pending, the answered ones match
`answers.jsonl` (strict judge), unreadable ones are pending; `src/merge_sub.py` keeps a rule answer first, fills the
others from the model's file, writes abstain for the rest, in the split's order.

`test_canon_flow.py` (stage 1.5, `predict_flow.py --canon 1`, fake model): two training questions asked in a new way
(restatements from the real v33 run) must be answered by the rules on the restatement, only the unreadable question
reaches stage 2, the stage-1 answer is kept.

```bash
S=<scratch with cache_train / cache_pub> WORKERS=4 bash tests/eval_all.sh [quick]
```

`eval_all.sh` (~20 min on 4 cores): regenerates the train question cleaning, `test_qnorm`, then `tests/eval_split.py`
(mapping by `src/automap.py`, no reference programs, no model) on train and public, and the stress tests: questions
accent-free / with chat noise (3 seeds, `src/perturb.py`), every README blanked. 2026-11-05: train 3000 / 3000, public
1000 / 1000, train README-blanked 3000 / 3000, public README-blanked 0.991, accent-free 0.981, noise 0.948-0.962 (the
gap is mostly abstains, which the model stage receives). 2026-11-06 (final code): accent-free 0.989, noise 0.971 /
0.974 / 0.978, README blanked public 0.993 / train 1.000, columns renamed (`--perturb-data rename:<seed>`, another
export of the same data) public 1000 / 1000, train 3000 / 3000. The public reference file only scores; nothing in
`src/` reads it. Under the Docker image's pandas 2.3.3 + calamine (local `--target` install) public is 1000 / 1000 as
well.
