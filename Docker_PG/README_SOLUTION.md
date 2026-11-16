# DataMate solution

## Models (total ≤ 15B parameters)

| Model | Parameters | Licence | Released |
|---|---|---|---|
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) + LoRA fine-tunes merged into it | 9.65B | Apache-2.0 | 27/02/2026 |

LoRA adapters of the same backbone, trained on the competition's training split only:
- **Solver** WG1, merged at build time with weight 0.35 (served weights `/models/solver`).

sha256 of every file: `hashes/`. Inference is fully offline (`--network none`).

## Method (`app/src/predict_flow.py`, configuration `app/configs/solution.json`)

The model writes every answer program. No logic is keyed on a question or business identifier and no answer is hard-coded;
every rule card and shop function of a test split is written at run time from that split's own rulebook and data.

1. **Ingest and schema** (`src/ingest.py`, `src/automap.py`, `src/canon.py`): CSV / Excel / SQLite files are cleaned and mapped to canonical tables; the model writes the mapping program when no known layout fits.
2. **Toolkit** (`src/dm.py`, `src/industry.py`): periods, valid orders, aggregation, ranking with ties, rounding and formatting. Industry-specific definitions are not built in: only general text tools (product line, brand, vehicle plate, service task) are given; every business definition comes from the split's rulebook.
3. **Rule cards and shop functions written by the model at run time** (`src/lrl_gen.py`, `src/units.py`): the split's rulebook rows become rule cards, the shop's data gives value aliases, and each shop-specific row becomes a Python function kept only when it runs on the shop's data and its samples agree.
4. **Vietnamese glossary** (`src/refs/lrl/glossary_vi.jsonl`): meanings of common abbreviations, chat spellings, money and unit words, shown when the question uses the word.
5. **Question reading** (`src/hints.py`, `src/qparse.py`, `src/qnorm.py`): the period(s), thresholds, entities of the shop's data, expected answer form and the rulebook rows the question mentions are added to the prompt.
6. **Generation** (`src/hybrid.py`): retrieved examples of training shops; the model writes a program, which is executed in a sandbox with repair turns; several samples are voted with the official matching rules. A review turn re-checks suspicious results.

## Input / output

Input: the test split mounted read-only at `/data` (`questions.jsonl`, `rulebook.md`, `calendar.csv`, `data/<business_id>/`).
Output: `/result/submission.jsonl` (one line per question, appended while running) and `/result/logs/` (`run.log` and one JSON
per question with the programs written, execution output, votes and final answer).

## Public test result

PA **0.880** (right / wrong / abstained: 920 / 40 / 40), file `submission.jsonl` of this folder (Kaggle 4 × L4).
