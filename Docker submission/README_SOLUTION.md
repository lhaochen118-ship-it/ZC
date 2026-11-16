# DataMate solution — Zalo AI Challenge 2026 (README for the organisers)

## Models used at inference (total ≤ 15B parameters)

| Model | Parameters | Licence | Released | Use |
|---|---|---|---|---|
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) + our LoRA fine-tunes (merged, no extra parameter) | 9.65B (language model + vision tower + MTP head, all counted) | Apache-2.0 | before 01/10/2026 | every model call (one backbone) |

Three LoRA adapters of the same backbone (trained on the competition's training split only):
- **S (solver)** V3A, merged at build time with weight 0.70 (WiSE-FT interpolation) — the served weights `/models/solver`;
- **W2 (linker)** WL1 and **W3 (judge)** WJR1, merged at run time into a copy of the S weights (`training/merge.py`, ~80 s each).
sha256 of every adapter file: `src/adapter_hashes.json`, `src/adapter_hashes_w.json`. No other model; no network (`--network none`).

## Method (`src/predict_flow.py`, configuration `configs/flow_p5_final.json`)

The model writes every answer program; no logic keyed on question or business identifiers, no hard-coded answer.

1. **Ingest + schema** (`src/ingest.py`, `src/automap.py`): generic cleaning of CSV / Excel / SQLite, mapping to canonical tables.
2. **Toolkit** (`src/dm.py`, `src/industry.py`): periods, valid orders, aggregation, ranking with ties, formatting, and the rulebook's
   industry definitions as tested functions (generic per industry, never per shop or question).
3. **W1 — rule cards and shop functions written by the model** (`src/lrl_gen.py`, `src/units.py`): rulebook rows compiled into cards,
   value aliases, and one Python function per shop-specific row, kept only when it runs on the shop's data and samples agree.
4. **W2 — linker** (`src/link.py`): the model links each question to the rulebook rows it uses.
5. **S — generation** (`src/hybrid.py`): 3 retrieved checked examples of training shops; the model writes a program; sandboxed
   execution, repair turns, up to 16 samples (half reasoning first), vote with the organisers' matching rules.
6. **W3 — judge** (`src/judge.py`): on questions whose samples disagree, the model judges the candidates (list / point / point+think
   ensemble); abstain when no candidate is reliable.

## Output

`/result/submission.jsonl` (one line per question, appended during the run) and `/result/logs/<id>.json` per question: id, final
answer, elapsed time, the programs written, execution output, votes, judge verdict; `/result/logs/run.log` for the whole run.

## Results (public test, 1000 questions, organisers' matching rules, Kaggle 4×L4)

This configuration: **0.952** (file `submission.jsonl` of this folder = the output of that run, D1);
an independent re-run with the current code: 0.942 (run-to-run spread ~±0.01, Rule 8 tolerance 0.03). Wall clock 69–110 min.
