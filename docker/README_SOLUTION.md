# DataMate solution (Zalo AI Challenge 2026) — v3

## Models used at inference (total ≤ 15B parameters)

| Model | Parameters | Licence | Public since | Use |
|---|---|---|---|---|
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9.65B (language model + vision tower + MTP head, all counted) | Apache-2.0 | 2026-02-27 | the only model: writes mapping programs and answer programs |

Quantisation: FP8 weights computed by vLLM at load time (`QUANT=fp8`; `QUANT=` for bf16). No other model, no
embedding / reranker model, no external service; the container runs with `--network none`.

## Method (no logic keyed on business or question ids)

1. **Ingest** (`src/ingest.py`): generic cleaning of CSV / Excel / SQLite (header detection, repeated headers, total
   rows, Vietnamese / US numbers, `105k`, `1tr440`, dates in many forms, sheets of one list merged).
2. **Rulebook semantics as tested code** (`src/dm.py`): periods, valid orders, revenue, ranking with ties by code, debt,
   stock, lots, purchases, prices, profit; rulebook §7 "no data is not zero" raises `Unanswerable`.
3. **Mapping** (`src/canon.py`): per business, a short program maps the raw tables to canonical tables and calls
   `finalize` (rulebook §8 "how files are written"). Sources: generic reference programs from the training split whose
   tables resolve by column names, else K programs written by the model (few-shot: the two most similar training
   references), each executed, checked and repaired; the candidate with the best coverage / agreement is used.
4. **Answer programs**: K short programs per question written by the model with retrieved verified training examples
   (`src/fewshot.py`), plus a rule-based reading of common question patterns (`src/qparse.py`); programs run in a sandbox;
   answers are voted with the organisers' matching rules; below the vote threshold the agent abstains.
5. **Scheduler** (`src/pipeline.py`): one model replica per GPU, generation / execution / repairs overlap, GPUs take the
   next business from a shared queue; every answer is written to `/result/submission.jsonl` as soon as it is decided.

Training data used: the training split only (reference mapping programs and verified question → program examples
built from it). No public or private test data is used for training or tuning.

## Run

`/app/predict.sh` (entry point): reads `/data`, writes `/result/submission.jsonl` and `/result/logs/<id>.json`
(id, final answer, elapsed time, generated programs, execution output, votes) plus `/result/logs/run.log`.
Defaults: `K=4 KC=4 BUDGET=20400` (seconds, 5 h 40 min for the agent), `QUANT=fp8`, `PIPE=1`.
