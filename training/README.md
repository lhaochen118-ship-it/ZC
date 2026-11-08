# training/ — how the fine-tuned model is reproduced

Everything here runs offline from the organisers' **training** split only (never the public / private test: the factory's
tests check it). Order (docs/08_MASTER_PLAN.md §5.9, §7, WP4–WP7):

1. **Canonical tables** of every training shop: `python3 tests/build_canon.py data/raw/train/train <ingest_cache> <canon_dir>`
   (`src/automap.py`, deterministic).
2. **Data factory** (`training/data/`, CPU; records = the exact inference prompt of `src/shopctx.py` + the labelled output
   `<plan>…</plan><code>…</code>`; the 6 held-out shops `biz_105 biz_109 biz_114 biz_123 biz_127 biz_130` are never used):
   - D1 `d1_train_pairs.py` — training question → checked program (`src/refs/answer_examples.jsonl`); also writes the example
     bank with question skeletons (`src/refs/bank/examples.jsonl`).
   - D2 `d2_paraphrase.py` — paraphrases of the training questions (`src/paraphrase_llm.py`), period-changing ones dropped.
   - D2N `d2n_noise.py` — accent-free / chat noise / regional words (`src/perturb.py`).
   - D3 `d3_sample_programs.py` → `d3_teacher_prompts.py` → GPU job `training/d3_job.py` — checked programs re-instantiated on
     other shops / days / periods / entities, a teacher model writes the questions, the student reads them back through the
     hybrid flow and only questions answered like the program are kept.
   - D4T `d4_tool_traj.py` — repair turns (a realistic mistake, the flow's precheck message, the checked program).
   - Tests: `tests/test_data_factory.py`, `tests/test_sft_format.py`.
3. **LoRA SFT** — `training/sft_unsloth.py` (one GPU per branch: Unsloth, bf16 LoRA, peak 18.8 GB at micro-batch 4 on an L4)
   or `training/sft.py` (TRL 1.15 + PEFT 0.21; FSDP with `accelerate launch --config_file training/fsdp{2,4}.yaml`). LoRA on
   every linear layer of the language model (attention, linear attention, MLP; not the vision tower, MTP head or lm_head),
   loss on the assistant output only, cosine schedule, seed 20261108. Branches: `training/arms_job.py` (4 at once on a
   4×L4 session, each merged and evaluated on `quick400`).
4. **Merge** — `training/merge.py merge <base> <out> <adapter>[,…] [--weights …]`: LoRA deltas written into a copy of the
   original checkpoint (same tensor names: vLLM serves it like the base, `--quantization fp8`); `merge.py vllm` writes an
   adapter vLLM can load with `--lora`.

Versions (Kaggle GPU image + `nhatlong1103/zc-assets` vLLM wheels + `yeshenhi/zc-train-wheels`): Python 3.13, torch 2.13.0,
vLLM 0.32.0, transformers 5.17.0, peft 0.21.2, trl 1.15.0, accelerate 1.15.0, datasets 4.8.5, flash-linear-attention 0.6.0,
unsloth 2026.11.3 (isolated `--target` install; the image's torchao 0.10 must be uninstalled for PEFT).
Kaggle jobs: `automation/jobs/*.jobs` through `automation/fanout.py` (`--train` installs the training wheels offline).
