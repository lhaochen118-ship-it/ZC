# DataMate solution (Zalo AI Challenge 2026) — hybrid flow: the model writes every answer program

## Model used at inference (total ≤ 15B parameters)

| Model | Parameters | Licence | Use |
|---|---|---|---|
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) + our LoRA fine-tunes merged into its weights | 9.65B (language model + vision tower + MTP head, all counted; merging adds no parameter) | Apache-2.0 | writes the answer program of every question |

The served weights are the base checkpoint plus LoRA deltas merged at image build time (`soup.json`): one fine-tune (WF3)
scaled by 0.35 (WiSE-FT, Wortsman et al. 2022) and the mean of 4 LoRA branches trained on top of that scaled model (RS1-RS4,
x 0.25 each), merged (`docker/merge_assets.py` with `training/merge.py`; adapters: Kaggle dataset
`nhatlong1103/zc-final-adapter`, sha256 of every file in `src/adapter_hashes.json`). vLLM serves them with FP8 weights
computed at load time. No other model, no external service; the container runs with `--network none`.

## Method (`src/predict_flow.py` → `src/agent.py --flow hybrid`; design: `docs/06_HYBRID_FRAMEWORK.md`, `docs/08_MASTER_PLAN.md`)

Rules prepare and check; **the model writes the answer program of every question** (no rule writes an answer).

1. **Ingest + schema** (`src/ingest.py`, `src/automap.py`): generic cleaning of CSV / Excel / SQLite and mapping of every
   shop's tables to canonical tables (orders, lines, purchases, lots, debts …) from the README and value profiles.
2. **Toolkit** (`src/dm.py`, `src/industry.py`): the rulebook's definitions as tested functions (periods, valid orders,
   revenue, ranking with ties, debt, stock, lots, profit, industry books); "no data" raises `Unanswerable`.
3. **Hints** (`src/hints.py`): the question's period(s), entities linked to data values (items, customers, units, sizes, repair
   jobs …), the expected answer type, what the question counts and its thresholds, rulebook terms (matched with accents), data
   the shop does not record — read by general rules, shown to the model as hints.
4. **Examples** (`src/retrieval.py`): 3 checked (question → program) pairs of other training shops, retrieved by the
   question's skeleton.
5. **Generation** (`src/hybrid.py`, `src/prompting.py`): Qwen3.5-9B writes `<plan>` + `<code>` (a short program over the
   toolkit). Two samples first; when they disagree, more samples, half of them reasoning first (thinking mode).
6. **Checks**: precheck before running (unknown names, entities not in the data, periods, safety; a size the question does not
   name, a unit count without its unit filter → repair turn), execution in a sandbox, repair turns with the error message, a
   review turn when the hints say the shop's data does not record what is asked, postcheck flags (empty / zero results, answer
   kind, a period / threshold of the question not used by the program); answers wrapped in the hinted answer type.
7. **Vote + abstain**: answers grouped with the organisers' matching rules; abstain when the top group's share is below
   the threshold (a wrong answer costs a point).

## Training (`training/README.md`)

LoRA SFT (r 32, all linear layers of the language model) on the competition's training split only:
D1 = 2,119 checked (question → program) pairs of the training shops, D2 = paraphrases of those questions written by
Qwen3.5-9B itself, D2N = accent-free / chat-noise variants; prompts in the inference format.
- WF3: one LoRA fine-tune on D1 + a D2 sample (4,238 records), scaled by 0.35 toward the base weights (the scaling keeps the
  base model's robustness on rewritten questions). Of 4 such fine-tunes (2 seeds of D1, D1 + D2, D1 + D2N), the one with
  paraphrases generalised best; averaging all four was weaker (the D1-only ones pull the soup down).
- RS1-RS4 (self-distillation, STaR / rejection-sampling fine-tuning): the earlier model (the soup of the 4 fine-tunes)
  answered the training-split questions (training shops not held out, their paraphrases, teacher-written questions about them); its own outputs whose
  programs ran to the reference answer were kept (710 from questions where its votes split, failed or abstained; a correct
  program found only by reasoning first or by a repair turn is written as the direct answer), plus easy ones / D1 records;
  4 LoRA branches (2 seeds, two mixes) trained on top of WF3 x 0.35 and averaged.
No public or private test data was used for training; public answers were used only to score offline.

## Results (offline, organisers' matching rules)

| Set | PA |
|---|---|
| Public test, original 1000 questions | 0.981 / 0.980 (two seeds) |
| Public questions rewritten by a model (held out, 1373) | 0.63 |
| Training shops held out of training (1826) | 0.89 |

(The previous model, a soup of the 4 fine-tunes x 0.35: public 0.982 / 0.981, held out 0.61 / 0.87; the current one is equal
on the public test within run-to-run noise and better on both held-out sets, the estimate of the private test.)

About 0.6 s / question on 4×L4 (1000 questions ≈ 10 minutes plus model load).

## Run

`/app/predict.sh` → `src/predict_flow.py --config /app/configs/flow_final.json`: reads `/data`, writes
`/result/submission.jsonl` and `/result/logs/<id>.json` (id, final answer, elapsed time, programs, execution output, votes).
Environment overrides: `MODEL`, `CONFIG`, `BUDGET` (default 20400 s), `RUNS` (1), `DP` (one model copy per GPU), `EXTRA`.
Build: `docker/fetch_assets.sh` (base weights from Hugging Face + adapters from Kaggle, both sha256-checked), then
`docker/build.sh`.
