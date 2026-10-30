# DataMate solution — README (draft, to be finalised before the 16/11 submission)

## Models used at inference (Rule 2)

| Model | Parameters (total) | Licence | Public release | Use |
|---|---|---|---|---|
| Qwen3-14B (official FP8 / AWQ quantisation by the Qwen team) | 14.8 B | Apache-2.0 | 2025-04-29 | the only model: writes the data-preparation and query programs |

Total parameters of all models used at inference: **14.8 B ≤ 15 B**. No embedding model, reranker or classifier is
used (any extra model would exceed the 15 B budget). No fine-tuning yet; if a LoRA is trained it is merged into the same
14.8 B weights and the training code is added to `train/`.

## What the system does (no human in the loop, nothing keyed on question or business ids)

1. **Ingest** (`src/ingest.py`, deterministic): reads every CSV / Excel sheet / SQLite table of the business, detects
   the header row, drops repeated headers and note / total rows, parses Vietnamese numbers (`10.000đ`, `105k`,
   `1tr440`, `531 500`, `1 bao 25kg` → value + unit column), mixed date formats (`dd/mm/yyyy`, `1 thg 6 2025`, Excel
   serials, `dd/mm` completed with the year of the sheet / file), and merges the files / sheets of one list.
2. **Compile** (`src/agent.py`, once per business): the LLM writes a program building canonical tables (orders, lines,
   items, purchases, payments, …) and notes on what the data does not contain, following the split's `rulebook.md`
   and the business README. Several candidates are executed and schema-checked; errors are sent back for repair; the
   candidate whose monthly totals agree most with the others is kept.
3. **Answer**: for each question, several programs (greedy + fixed-seed samples) over the canonical tables and over the
   cleaned raw tables are executed in a sandbox (time limit); failures are repaired with the error message; zero /
   empty results get one self-check.
4. **Decide**: answers are grouped with the organisers' matching rules and voted; the system answers only when the
   majority is strong enough (Penalized Accuracy: a wrong answer costs 1, silence costs 0), otherwise `abstain`.
5. **Format** deterministically (JSON numbers, rounding, `YYYY-MM-DD` / `YYYY-MM`, list order, table columns).

## Running

```bash
docker run --gpus all --network none -v /path/to/test:/data:ro -v /path/to/result:/result <image>
```

- Output: `/result/submission.jsonl` (written chunk by chunk, so a time-out keeps every finished answer), one log per
  question in `/result/logs/<id>.json` (question, final answer, elapsed seconds, every generated program, its output
  or error, votes), plus `/result/logs/run.log`.
- Time: a 5 h 40 min budget for the agent (the rest of the 6 h covers loading); when the estimate runs over, the
  remaining businesses switch to a single greedy program, and at the limit the remaining questions abstain.
- Hardware used for the estimate: 2 × RTX 5090, one model copy per GPU (`--dp 2`).

## Reproducibility (Rule 8)

Greedy decoding for the first program of each path and fixed seeds for the others; fixed chunking and business order;
deterministic formatting. Remaining non-determinism comes from GPU batching; the logs record every program and output
so that any difference between two runs can be traced.

## Training

No training data is used at inference. Development used only the provided training split (and data we synthesise);
the public / private test sets are never used for training or tuning.
