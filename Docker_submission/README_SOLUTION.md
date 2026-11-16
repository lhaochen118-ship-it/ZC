# DataMate solution — Zalo AI Challenge 2026 (README for the organisers)

## Models used at inference (total ≤ 15B parameters)

| Model | Parameters | Licence | Released | Use |
|---|---|---|---|---|
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) + our LoRA fine-tunes (merged, no extra parameter) | 9.65B (language model + vision tower + MTP head, all counted) | Apache-2.0 | 27/02/2026 (Hugging Face) | every model call (one backbone) |

Two LoRA adapters of the same backbone, trained on the competition's **training split only** (questions, data and rulebook of the
training split, their paraphrases and noisy rewrites; no public or private test data):
- **S (solver)** V3A, merged at build time with weight 0.70 (WiSE-FT interpolation) — the served weights `/models/solver`;
- **W3 (judge)** WJR1, merged at run time into a copy of the S weights (`training/merge.py`, ~80 s).
sha256 of every adapter file: `src/adapter_hashes.json`, `src/adapter_hashes_w.json`. No other model; no network (`--network none`).

## Method (`src/predict_flow.py`, configuration `configs/flow_p5c4.json`)

The model writes every answer program; nothing is keyed on a question or business identifier and no answer is hard-coded. Nothing
derived from a test split is shipped: every rule card and shop function of a split is written by the model at run time from that
split's own inputs (its rulebook and its shops' data).

1. **Ingest + schema** (`src/ingest.py`, `src/automap.py`, `src/canon.py`): generic cleaning of CSV / Excel / SQLite and mapping to
   canonical tables (column-name synonyms of common Vietnamese sales software / spreadsheets; the model writes the mapping program
   when no known layout fits).
2. **Toolkit** (`src/dm.py`, `src/industry.py`): periods, valid orders, aggregation, ranking with ties, formatting; industry toolkits
   (phone shop, motorbike repair, mother & baby) shown only to shops whose rulebook has that industry section.
3. **W1 — rule cards and shop functions written by the model at run time** (`src/lrl_gen.py`, `src/units.py`): the split's rulebook
   rows compiled into cards, value aliases from the shop's data, and one Python function per shop-specific row, kept only when it
   runs on the shop's data and samples agree. Rows of the training rulebook count as known; every other row is the split's own rule.
4. **General Vietnamese glossary** (`src/refs/lrl/glossary_vi.jsonl`, built by `training/data/build_glossary.py`): word meanings only
   (abbreviations, chat spellings, money / unit slang, terms of the training split's kinds of shop), shown when the question uses the
   word; computation rules always come from the split's rulebook.
5. **S — generation** (`src/hybrid.py`): 3 retrieved checked examples of training shops; the model writes a program; sandboxed
   execution, repair turns, up to 16 samples (half reasoning first), vote with the organisers' matching rules. A general consistency
   check (`src/consistency.py`, `src/precheck.py`) compares each program with what the question states (its period(s), numeric
   thresholds, answer form): a program ignoring them counts half in the vote; `len()` of a boolean mask is sent back for repair.
6. **W3 — judge** (`src/judge.py`): on questions whose samples disagree, the model judges the candidates (list / point / point+think
   ensemble); abstain when no candidate is reliable.

## Output

`/result/submission.jsonl` (one line per question, appended during the run) and `/result/logs/<id>.json` per question: id, final
answer, elapsed time, the programs written, execution output, votes, judge verdict; `/result/logs/run.log` for the whole run.

## Results (public test, 1000 questions, organisers' matching rules, Kaggle 4×L4)

This configuration: **0.939** (963 right / 24 wrong / 13 abstained; run as two halves of the shops on two machines, 69 min each;
file `submission_flow_p5c4.jsonl` of this folder). Run-to-run spread ~±0.01 (Rule 8 tolerance 0.03). The time budget of the
container is 5 h 40 min (`BUDGET`), the remaining phases are skipped when it runs short.
