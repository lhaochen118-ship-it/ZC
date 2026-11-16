# DataMate solution — Zalo AI Challenge 2026 (README for the organisers)

## Models used at inference (total ≤ 15B parameters)

| Model | Parameters | Licence | Released | Use |
|---|---|---|---|---|
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) + our LoRA fine-tunes (merged, no extra parameter) | 9.65B (language model + vision tower + MTP head, all counted) | Apache-2.0 | before 01/10/2026 | every model call (one backbone) |

One LoRA adapter of the backbone (trained on the competition's training split only):
- **S (solver)** WG1, merged at build time with weight 0.35 (WiSE-FT interpolation) — the served weights `/models/solver`.
sha256 of every adapter file: `0.994_Docker_submission/adapter_hashes.json`. No other model; no network (`--network none`).

## Method (`src/predict_flow.py`, configuration `configs/flow_p_final.json` (system P))

The model writes every answer program; no logic keyed on question or business identifiers, no hard-coded answer.

1. **Ingest + schema** (`src/ingest.py`, `src/automap.py`): generic cleaning of CSV / Excel / SQLite, mapping to canonical tables.
2. **Toolkit** (`src/dm.py`, `src/industry.py`): periods, valid orders, aggregation, ranking with ties, formatting, and the rulebook's
   industry definitions as tested functions (generic per industry, never per shop or question).
3. **Hints and checks** (`src/hints.py`, `src/precheck.py`, `src/qparse.py`): generic parsing of periods / entities in the question,
   retrieval of the rulebook rows it mentions, and execution checks on the candidate programs (counting, sizes, averages).
4. **Generation** (`src/hybrid.py`): retrieved checked examples of training shops; the model writes a program; sandboxed execution,
   repair turns with the error message, several samples, vote with the organisers' matching rules; abstain when the top group is
   below half.

## Output

`/result/submission.jsonl` (one line per question, appended during the run) and `/result/logs/<id>.json` per question: id, final
answer, elapsed time, the programs written, execution output, votes, `/result/logs/run.log` for the whole run.

## Results (public test, 1000 questions, organisers' matching rules, Kaggle 4×L4)

This configuration: **0.994** (file `submission.jsonl` of this folder = the output of that run; 3 seeds 0.993 / 0.994 / 0.993).
