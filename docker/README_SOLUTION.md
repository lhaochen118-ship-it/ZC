# DataMate solution (Zalo AI Challenge 2026) — v3, rule-first

## Models used at inference (total ≤ 15B parameters)

| Model | Parameters | Licence | Public since | Use |
|---|---|---|---|---|
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9.65B (language model + vision tower + MTP head, all counted) | Apache-2.0 | 2026-02-27 | the only model: answers the questions the rule stage does not read (stage 2) |

Quantisation: FP8 weights computed by vLLM at load time. No other model, no embedding / reranker model, no external
service; the container runs with `--network none`.

## Method (no logic keyed on business or question ids, no answers in the code)

**Stage 1, rules (CPU, minutes, no model)** — `src/rules_first.py`:

1. **Ingest** (`src/ingest.py`): generic cleaning of CSV / Excel / SQLite (header detection, repeated headers, total
   rows, Vietnamese / US numbers, `105k`, `1tr440`, dates in many forms, sheets of one list merged).
2. **Mapping** (`src/automap.py`): every raw table gets column roles from the business README (column descriptions,
   bold definitions, unit rules), a lexicon of column names (exact, then by head words), value profiles (statuses,
   payment methods, channels, phones, codes) and money identities of an order (paid = total − debt …); tables are
   classified by their roles (orders, lines, receipts, lots, payments, instalment / debt / repair books …) and assembled
   into canonical tables. Shop conventions are handled generically (status written in notes, voucher / points /
   shipping pseudo-lines, toppings, abbreviation sheets, bundle units, chained unit factors, debts in notes).
3. **Rulebook semantics as tested code** (`src/dm.py`, `src/industry.py`): periods, valid orders, revenue, ranking
   with ties, debt, stock, lots, purchases, prices, profit, industry books; rulebook "no data is not zero" raises
   `Unanswerable`.
4. **Question reader** (`src/qnorm.py` + `src/qparse.py`): cleaning (accents restored, chat abbreviations, Telex,
   typos, glued words, regional words), then templates that write a short program over the toolkit; a precision guard
   drops a reading that ignores what the question states (an amount, a count, a qualifier), and a question about data
   the shop does not record is answered `unanswerable`. A rule answer is final.

**Stage 2, model (GPU)** — only the questions stage 1 leaves (not read, program failed, mapping failed): `src/agent.py`
with Qwen3.5-9B on a split holding just those questions. Mapping = the stage-1 mapper when it runs cleanly, else
mapping programs written by the model (executed, checked, repaired); K answer programs per question with retrieved
verified training examples; answers voted with the organisers' matching rules, abstain below the vote threshold;
independent runs voted while time allows (`src/vote.py`). Scheduler `src/pipeline.py`: one replica per GPU.

**Development data.** Rules and templates were written from the training split and checked on the public split: train
3000 / 3000 and public 1000 / 1000 (strict matching), and robustness tests on rewritten questions (accent-free, chat
noise, paraphrases by the model) and on READMEs blanked. The public reference answers were used only to score offline;
no solution file contains or reads them, and nothing is keyed on a business or question id.

## Run

`/app/predict.sh` → `src/predict_flow.py`: reads `/data`, writes `/result/submission.jsonl` (complete after stage 1,
re-merged every 20 s during stage 2, rule answers first) and `/result/logs/<id>.json` (id, final answer, elapsed time,
template / program or the model's programs and votes) plus `/result/logs/run.log`.
Environment overrides: `MODEL`, `BUDGET` (default 20400 s for both stages), `RUNS` (3), `LLM` (0 = rules only), `DP`,
`EXTRA` (extra `agent.py` flags).
