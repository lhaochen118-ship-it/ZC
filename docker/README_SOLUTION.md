# DataMate solution (Zalo AI Challenge 2026) — hybrid flow: the model writes every answer program

## Model used at inference (total ≤ 15B parameters)

| Model | Parameters | Licence | Use |
|---|---|---|---|
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) + our LoRA fine-tunes merged into its weights | 9.65B (language model + vision tower + MTP head, all counted; merging adds no parameter) | Apache-2.0 | writes the answer program of every question |

The served weights are the base checkpoint plus one LoRA fine-tune (WG1) scaled by 0.35 (WiSE-FT, Wortsman et al. 2022), merged (`docker/merge_assets.py` with `training/merge.py`; adapters: Kaggle dataset
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
- WG1: one LoRA fine-tune on D1 + all D2 paraphrases (3,818; 14.7M tokens, 66 steps), scaled by 0.35 toward the base
  weights (the scaling keeps the base model's robustness on rewritten questions). Chosen among 8 fine-tunes evaluated alone
  and in soups (D1 only, D1 + D2 sample, D1 + D2N, D1 + D2 + D2N, two seeds): paraphrase data generalised best; averaging
  members (soups) and self-distillation on top (RFT) did not beat it on the public test.
No public or private test data was used for training; public answers were used only to score offline.

## Results (offline, organisers' matching rules)

| Set | PA |
|---|---|
| Public test, original 1000 questions | 0.993 / 0.994 / 0.993 (three seeds; container path 0.994; before the three checks below 0.986 / 0.985 / 0.987) |
| Public questions rewritten by a model (held out, 1373) | 0.63–0.64 |
| Training shops held out of training (1826) | 0.88 |

(The first submitted model, a soup of 4 fine-tunes x 0.35: public 0.982 / 0.981, held out 0.61 / 0.87.) The flow also
treats `len()` of a True / False mask as an error and asks for a repair (`mask_check`), and (12/11) three more prechecks: a counting
question must not use a money metric (`count_check`), a catalogue size the question names must filter the program (`size_use_check`),
`aov()` in a shop whose industry counts tickets is replaced by `avg_ticket()` (`ticket_avg_check`).

## Run

`/app/predict.sh` → `src/predict_flow.py --config /app/configs/flow_final.json`: reads `/data`, writes
`/result/submission.jsonl` and `/result/logs/<id>.json` (id, final answer, elapsed time, programs, execution output, votes).
Environment overrides: `MODEL`, `CONFIG`, `BUDGET` (default 20400 s), `RUNS` (1), `DP` (one model copy per GPU), `EXTRA`.
Build: `docker/fetch_assets.sh` (base weights from Hugging Face + adapters from Kaggle, both sha256-checked), then
`docker/build.sh`.
