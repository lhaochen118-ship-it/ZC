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
