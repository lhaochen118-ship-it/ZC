# DataMate solution (Zalo AI Challenge 2026) — the model writes every answer program and the shop's rule functions

## Model used at inference (total ≤ 15B parameters)

| Model | Parameters | Licence | Use |
|---|---|---|---|
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) + our LoRA fine-tune merged into its weights | 9.65B (language model + vision tower + MTP head, all counted; merging adds no parameter) | Apache-2.0 | writes the rule functions of the split (W) and the answer program of every question (S) |

The served weights are the base checkpoint plus one LoRA fine-tune (v3a) scaled by 0.70 (WiSE-FT, Wortsman et al. 2022), merged
(`docker/merge_assets.py` with `training/merge.py`; adapter: Kaggle dataset `nhatlong1103/zc-final-adapter`, sha256 of every file in
`src/adapter_hashes.json`). vLLM serves them with FP8 weights computed at load time. No other model, no external service; the
container runs with `--network none`.

## Method (`src/predict_flow.py` → `src/agent.py --flow hybrid --pure 1`; design: `docs/17_PLAN4_SWV.md`)

No hand-written rule reads the question: the model writes every program, and the shop-specific rules are written by the model too.

1. **Ingest + schema** (`src/ingest.py`, `src/automap.py`): generic cleaning of CSV / Excel / SQLite and mapping of every shop's
   tables to canonical tables (orders, lines, purchases, lots, debts …) from the README and value profiles.
2. **Toolkit** (`src/dm.py`, `src/industry.py`): the rulebook's general definitions as tested functions (periods, valid orders,
   revenue, ranking with ties, debt, stock, lots, profit, industry books); "no data" raises `Unanswerable`.
3. **W — rule functions written by the model** (`src/lrl_gen.py --mode unit`, `src/units.py`): before answering, the model reads each
   rulebook row of a shop's industry section and every row no known rulebook has, with the shop's table card, and writes one Python
   function per row (constrained JSON: plan, name, signature, code as lines; 5 samples). A function is kept only when it runs on the
   shop's own data over 13 periods (last 12 months + the whole data) with a non-degenerate single value, and at least 2 samples agree
   on its results; a failed function goes back to the model once with its real error (self-debugging). Kept functions appear in the
   prompt as `# SHOP FUNCTIONS` and are callable like toolkit metrics. Time cap 45 min; past it the run continues without them.
4. **Rule cards compiled by the model** (`src/lrl.py`, `src/refs/lrl/`): rulebook definitions, value aliases and transliterations,
   written by the model from the rulebook / data (no answer read), matched to the question.
5. **Examples** (`src/retrieval.py`): 3 checked (question → program) pairs of training shops, retrieved by the question's skeleton.
6. **S — generation** (`src/hybrid.py`, `src/prompting.py`): the model writes `<spec>` + `<plan>` + `<code>` (a short program over the
   toolkit, the shop functions and the tables). 4 samples first; when they disagree, up to 16, half of them reasoning first.
7. **Execution + repair**: generic precheck (unknown names, safety), execution in a sandbox, repair turns with the error message.
8. **Vote + abstain**: answers grouped with the organisers' matching rules; abstain when the top group's share is below 0.5.

## Training (`training/README.md`)

LoRA SFT (r 32, all linear layers of the language model) on the competition's training split only (v3a, 16.7M tokens, 77 steps,
FSDP on 4 L4, checkpoints chained across Kaggle sessions): checked (question → `<spec>` + program) pairs of the training shops,
rejection-sampled programs of the model itself on hard questions, and style variants of the training questions (abbreviated periods,
lexical-normalisation noise in the ViLexNorm style) written by Qwen3.5-9B. No public or private test data was used for training; public answers were used only to score offline.

## Results (offline, organisers' matching rules)

| Set | PA |
|---|---|
| Public test, original 1000 questions | **0.938** (this configuration); container path (`predict_flow`, 4×L4, 39 min) **0.922**; run-to-run spread ~±0.01 |
| 614 questions of the 5 public shops with an industry section, industry functions hidden (a new industry) | W written by the model: 0.476 → 0.595 (originals), 0.415 → 0.493 (paraphrases) |

## Run

`/app/predict.sh` → `src/predict_flow.py --config /app/configs/flow_final.json`: reads `/data`, writes `/result/submission.jsonl` and
`/result/logs/<id>.json` (id, final answer, elapsed time, programs, execution output, votes). Environment overrides: `MODEL`, `CONFIG`,
`BUDGET` (default 20400 s), `RUNS` (1), `DP` (one model copy per GPU), `EXTRA`. Build: `docker/fetch_assets.sh` (base weights from
Hugging Face + adapter from Kaggle, both sha256-checked), then `docker/build.sh`.

Fallback (the earlier system P with hand-written hints and checks, public 0.994): `CONFIG=/app/configs/flow_p_final.json` with the
previous adapter version (WG1 × 0.35; hashes `src/adapter_hashes_p.json`).
