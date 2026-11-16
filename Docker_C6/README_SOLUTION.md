# DataMate solution

## Models (total ≤ 15B parameters)

| Model | Parameters | Licence | Released |
|---|---|---|---|
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) + LoRA fine-tunes merged into it | 9.65B | Apache-2.0 | 27/02/2026 |

LoRA adapters of the same backbone, trained on the competition's training split only:
- **Solver** V3A, merged at build time with weight 0.70 (served weights `/models/solver`).
- **Judge** WJR1, merged at run time into a copy of the solver weights.

sha256 of every file: `hashes/`. Inference is fully offline (`--network none`).

## Method (`app/src/predict_flow.py`, configuration `app/configs/solution.json`)

The model writes every answer program. No logic is keyed on a question or business identifier and no answer is hard-coded;
every rule card and shop function of a test split is written at run time from that split's own rulebook and data.

1. **Ingest and schema** (`src/ingest.py`, `src/automap.py`, `src/canon.py`): CSV / Excel / SQLite files are cleaned and mapped to canonical tables; the model writes the mapping program when no known layout fits.
2. **Toolkit** (`src/dm.py`, `src/industry.py`): periods, valid orders, aggregation, ranking with ties, rounding and formatting. Industry-specific definitions are not built in: only general text tools (product line, brand, vehicle plate, service task) are given; every business definition comes from the split's rulebook.
3. **Rule cards and shop functions written by the model at run time** (`src/lrl_gen.py`, `src/units.py`): the split's rulebook rows become rule cards, the shop's data gives value aliases, and each shop-specific row becomes a Python function kept only when it runs on the shop's data and its samples agree.
4. **Vietnamese glossary** (`src/refs/lrl/glossary_vi.jsonl`): meanings of common abbreviations, chat spellings, money and unit words, shown when the question uses the word.
5. **Consistency check** (`src/consistency.py`, `src/precheck.py`): each program is compared with what the question states (periods, thresholds, answer form); a program that ignores them counts half in the vote.
6. **Generation** (`src/hybrid.py`): retrieved examples of training shops; the model writes a program, which is executed in a sandbox with repair turns; several samples are voted with the official matching rules.
7. **Judge** (`src/judge.py`): on questions whose samples disagree, the judge adapter chooses among the candidates (list / point / point with reasoning, ensembled); it abstains when no candidate is reliable.

## Input / output

Input: the test split mounted read-only at `/data` (`questions.jsonl`, `rulebook.md`, `calendar.csv`, `data/<business_id>/`).
Output: `/result/submission.jsonl` (one line per question, appended while running) and `/result/logs/`: `run.log` and one
`<id>.json` per question with `id`, `final_answer`, `elapsed_s`, the programs written, their execution output, the votes and
the judge verdict.

Training code: `/app/training/` (LoRA fine-tuning `sft.py` / `full_job.py`, training-data builders `data/`, `merge.py`).

## Public test result

PA **0.834** (right / wrong / abstained: 891 / 57 / 52), file `submission.jsonl` of this folder (Kaggle 4 × L4).
