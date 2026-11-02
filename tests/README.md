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
