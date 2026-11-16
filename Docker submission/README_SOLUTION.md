# DataMate solution — Zalo AI Challenge 2026 (README for the organisers)

## Models used at inference (total ≤ 15B parameters)

| Model | Parameters | Licence | Released | Use |
|---|---|---|---|---|
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) + our LoRA fine-tunes (merged, no extra parameter) | 9.65B (language model + vision tower + MTP head, all counted) | Apache-2.0 | before 01/10/2026 | every model call (one backbone) |

Three LoRA adapters of the same backbone (trained on the competition's training split only):
- **S (solver)** V3A, merged at build time with weight 0.70 (WiSE-FT interpolation) — the served weights `/models/solver`;
- **W2 (linker)** WL1 and **W3 (judge)** WJR1, merged at run time into a copy of the S weights (`training/merge.py`, ~80 s each).
sha256 of every adapter file: `src/adapter_hashes.json`, `src/adapter_hashes_w.json`. No other model; no network (`--network none`).

## Method (`src/predict_flow.py`, configuration `configs/flow_v2_final.json`)

No hand-written business rule: the model writes every answer program and every shop-specific rule (mode H: the hand-written
industry functions of `src/industry.py` are hidden from every prompt and from the program namespace).

1. **Ingest + schema** (`src/ingest.py`, `src/automap.py`): generic cleaning of CSV / Excel / SQLite, mapping to canonical tables.
2. **Neutral toolkit** (`src/dm.py`): periods, valid orders, aggregation, ranking with ties, formatting (rulebook general definitions).
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

Container path of this configuration: **0.863, 0.863, 0.858, 0.857** in four independent runs (mean 0.860; wall clock 110–125 min).
