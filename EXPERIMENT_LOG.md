# Experiment log — DataMate v3

Mỗi entry: ngày, thí nghiệm (mã V# trong `docs/04_DECISION_V3.md` §6), giả thuyết, cấu hình, dữ liệu, kết quả (PA theo
dev / holdout / type, % abstain, s/câu), phân tích lỗi, quyết định. Nhật ký v1: `docs/history/v1_experiment_log.md`.

## Entry 0 — 2026-11-02: đánh giá phương án cũ, quyết định v3

- v1 (`claude/peaceful-carson-0dny90`): PA train dev +0.47 / holdout +0.44 (+0.51 với thinking, 354 câu); public 764/1000
  câu sau 12 giờ trên 4×L4. Public LB SLM top-10 ở 0.90–1.0. → **bỏ**.
- v2 toolkit (`claude/funny-euler-ceut3o`) kiểm lại trên CPU, ánh xạ tham chiếu của chính DN: train **2651 đúng / 51 sai /
  298 chưa đọc** (tái lập đúng số đã báo). Cần sửa tên file `#Uxxxx` trong 4 tham chiếu (biz_112/115/118/130: tên file
  bị mã hoá khi giải nén ở máy khác) → ingest sẽ chuẩn hoá tên.
- Quyết định và kiến trúc v3: `docs/04_DECISION_V3.md`.

## Entry 1 — 2026-11-02: V1 (v3 pipeline với LLM, 4×L4), Qwen3-14B-AWQ vs Qwen3.5-9B

Cấu hình chung: `--k 4 --kc 4 --repairs 2 --shots 4 --qw 2 --max-tokens 1024 --compile-tokens 3072`, `--dp 4`, đường
round-based; train **leave-one-out** (chương trình ánh xạ tham chiếu của chính DN bị loại). Shard tĩnh `shard_of`:
s0 = biz_102/106/116/120/124/125/130, s1 = 101/103/107/108/110/117/121/126/127, s2 = 104/109/112/113/118/122/128,
s3 = 105/111/114/115/119/123/129.

| Run | Mô hình | s0 | s1 | s2 | s3 | Toàn train (3000) | Wall |
|---|---|---|---|---|---|---|---|
| v1q14-1 | Qwen3-14B-AWQ | 0.741 | 0.454 | 0.944 | 0.761 | **0.725** (2533 đúng / 359 sai / 108 abstain) | 4170 s |
| v1q35-1 | Qwen3.5-9B (FP8 online) | **0.900** | crash | crash | **0.877** | — | s0 2424 s, s3 2549 s |

- **Qwen3.5-9B > Qwen3-14B** trên cùng 1499 câu (s0 + s3): ≈ 0.889 vs 0.751. → mô hình mặc định: Qwen3.5-9B.
- v1 (LLM viết pandas cho từng câu) cùng giao thức: ≈ 0.45. Không LLM (chỉ tham chiếu + mẫu câu): 0.506.
- Public (v1q14-2, Qwen3-14B): xong 1000 câu trong 3390 s nhưng **đáp án mất** (lỗi đường dẫn bên dưới). Shard
  3355 / 2804 / 2146 / 2023 s: chia tĩnh lệch ~40 % → lý do của work stealing (`--pipeline`).
- Hai lỗi đã sửa (commit 6b79322): (1) `_Timeout` thoát khỏi worker → `imap_unordered` ném lại ở tiến trình chính,
  chết 2/4 shard của v1q35-1; (2) `make_kernel` chỉ thay token `OUT` đứng riêng → `OUT/train` ghi vào thư mục code bị
  xoá cuối kernel (chỉ còn log). Lần chạy pipeline đầu (p1q14) chết vì `agent.py` chạy dưới tên `__main__` (sửa 4f9bc0e).
- Đang chạy (code đã sửa): v2q35 (Qwen3.5, round-based, train + public, linhmai24), p2q35 (Qwen3.5, `--pipeline`,
  duylnng), p1q14b (Qwen3-14B, `--pipeline`, thnliu: A/B tốc độ với v1q14).

## Entry 2 — 2026-11-02: pipeline (không rào chắn) + think-map (Qwen3.5-9B FP8, 4×L4, train 3000 LOO)

| Run | dev | holdout | all | wall |
|---|---|---|---|---|
| v2q35 round-based | 0.897 | 0.644 | 0.810 | 7174 s |
| p3q35 `--pipeline` (sau khi sửa) | 0.901 | 0.745 | 0.848 | 3234 s |
| t1q35-1 `--think-map` | 0.926 | 0.761 | **0.870** | 4065 s |
| t1q35-2 `--think --think-map` | 0.915 | 0.691 | 0.839 | 8124 s |

- Pipeline: nhanh 2.2× so với round-based, PA cao hơn (repair ngay, không chờ vòng).
- Sai số gần như toàn bộ do **chọn sai ánh xạ** ở vài DN (107, 110, 117, 123, 124, 126, 127). DN ánh xạ đúng đạt 0.90–1.00.
- Đã thêm: kiểm tra hợp lý ánh xạ (`canon.plausibility`), vòng "đầy đủ" (`--complete-map`), hậu kiểm + escalation
  (`--escalate`), câu đã làm sạch trong prompt (`--qclean`), trọng số luật `--qw 4` (re-vote offline tốt hơn qw 2).
  Đo trong v4 (đang chạy).

## Entry 3 — 2026-11-02: làm sạch câu hỏi (`question_cleaning/`, `src/qnorm.py`)

Đo: leave-one-business-out trên 2232 câu người dùng gõ có dấu: bỏ dấu, làm sạch lại, so với bản làm sạch câu gốc (bỏ qua
vị trí dấu thanh và hoa thường).

| Phiên bản | Từ đúng | Câu đúng | Thay đổi |
|---|---|---|---|
| lần 1 | 99.3 % | 93.6 % | bigram + viết tắt + Telex |
| từ vựng chỉ từ câu có dấu, tên trong dữ liệu | 99.66 % | 96.1 % | |
| luật ngữ cảnh, lexicon cụm từ ngắn | 99.75 % | 97.3 % | (đo cũ, so cả vị trí dấu) |
| bảo vệ tên / mã, entity linking chỉ đổi dấu + khớp ≥ 3 từ của tên, `e`/`nhieu`/`đ` | **99.79 %** | **98.1 %** | 42 câu còn sai: mơ hồ thật |

- Duyệt tay toàn bộ thay đổi trên 3000 câu train. Lỗi tìm được đều đã sửa bằng luật chung và có test (29 trường hợp):
  - `Max` → Mã;
  - `SP` → SẢN PHẨM;
  - khách Lê → khách lẻ;
  - BÁNH CROISSANT;
  - `e` → ế;
  - `CT-11A` → CT-11Ạ.
- Bước LLM (zc-v4-5) sẽ xử lý phần mơ hồ còn lại: tên đường không có trong dữ liệu; lon / lớn; vỉ / vị.

## Entry 4 — 2026-11-02: v4b (plausibility + completeness + qclean + qw 4, ± escalation), Qwen3.5-9B FP8, 4×L4

Cấu hình chung: `--pipeline --think-map --k 4 --kc 6 --repairs 2 --shots 4 --max-tokens 1024 --qw 4`, `--complete-map 1`,
`--qclean 1` (bản qnorm mới), train 3000 câu leave-one-out.

| Run | dev | holdout | all | wall |
|---|---|---|---|---|
| t1q35-1 (entry 2, tốt nhất cũ) | 0.926 | 0.761 | 0.870 | 4065 s |
| **zc-v4b-1** escalate 4 | **0.976** | 0.913 | **0.955** (2902 đúng / 38 sai / 60 bỏ) | 6046 s |
| zc-v4b-3 escalate 0 | 0.955 | **0.918** | 0.943 (2890 / 62 / 48) | 4774 s |
| zc-v4b-2 public (escalate 4) | — | — | (không có nhãn) | 7599 s |

- Kiểm tra ánh xạ + vòng đầy đủ sửa các DN thảm hoạ: biz_110 −0.20 → +1.00, biz_127 −0.05 → +0.97, biz_124 0.66 → 0.97.
- Escalation: +0.021 dev, −0.005 holdout, +0.012 toàn bộ, +27 % thời gian. Kiểu `table` tụt (0.58 vs 1.00 không escalation).
- Còn yếu: biz_123 0.57 (nhà thuốc, pos.sqlite: lô / nhập / tồn), biz_111 0.86 (escalation làm tệ hơn 0.93), biz_117 0.87,
  biz_121 0.88, biz_122 0.90.
- v4 (14:59) chết cả 5 kernel do môi trường dựng sẵn import từ mount > 900 s (đã sửa, xem commit de732a8).

## Entry 5 — 2026-11-02: offline (CPU) measurements on the v4b logs; robustness + rule-reader fixes

Measured without GPU:
- re-executing the 441 logged mapping attempts: 171 → 222 valid (dedent, `fillna(None)`, repair splicing);
- replaying the escalation run: top1 0.942 → 0.949 (stock from lots, warning-aware ranking).
  Cross-mapping voting (vote2 0.935, agreement-first 0.867) and fallback (= top1) bring nothing → dropped.

Train oracle, `tests/eval_lib.py` (reference mappings, rule programs only):

| Toolkit / rule reader | Right | Wrong | Unparsed |
|---|---|---|---|
| morning (02/11) | 2666 | 36 | 298 |
| + stock from lots | 2675 | 27 | 298 |
| + cleaned question as input (`--clean`) | 2737 | 23 | 240 |
| + general pattern fixes, fee lines, debts without data | **2754** | **3** | 243 |

LLM cleaning pass: measured and rejected (Qwen3.5-9B 70.0 % vs 97.7 % word accuracy; all 14 of its changes wrong).
Runs: v5 (18:49, mapping robustness + compile-tokens 6144, ± escalation, repeat for determinism, public ×2),
v6-train (rule reader fixes).

## Entry 6 — 2026-11-02: v5 (mapping robustness, compile tokens 6144), train

| Run | dev | holdout | all | wall |
|---|---|---|---|---|
| v4b-1 escalate 4 (entry 4) | 0.976 | 0.913 | 0.955 | 6046 s |
| **v5-1** escalate 4 | **0.979** | 0.881 | 0.946 | — |
| v4b-3 escalate 0 | 0.955 | 0.918 | 0.943 | 4774 s |
| v5-4 escalate 0 | 0.942 | 0.919 | 0.934 | 5538 s |

- Gains as expected: biz_121 0.88 → 0.99 and biz_122 0.90 → 0.99 (stock from lots); biz_111 0.86 → 0.98.
- The total did **not** rise, because of **variance in the sampled mappings** of a few LLM-mapped businesses. The same
  selection rule (highest coverage) picks a different program each run: biz_123 0.57 → 0.19, biz_105 0.97 → 0.82
  (escalation off), biz_117 0.88 → 0.74.
  - biz_123 (v5-1) errors: revenue +0.1 % (returns table ignored), debts ×16 (payments not matched), group revenue 0
    (groups not on the lines).
  - These now trigger soft checks (6dc6cbb). The study of selection by the questions' rule programs is running.
- Mapping variance is also the main **determinism** risk (|ΔPA| ≤ 0.03 between runs). The repeat run v5-3 measures it.

## Entry 7 — 2026-11-02: public v5 ×2 (determinism), mapping selection by rule programs, --confirm

- Public v5-2: 113 abstains (v4b-2: 136). dev_003 went from 20 abstains to 0. dev_014 still has no mapping (complex
  multi-sheet workbooks; every attempt failed).
- **Determinism** (v5-2 vs v5-5, identical config): the same final answer on only **88.3 %** of the 1000 public
  questions. Different mappings were chosen for 7 of the 13 businesses (dev_009: 17 / 21 answers differ). Cause:
  the mapping samples are not reproducible across runs (batch composition → numerics).
  - Response 1: selection by the rule programs of the questions (fb0020f; offline +0.014 rule PA, never worse) —
    a stable outcome even when the sampled set differs.
  - Response 2: v9i tests `VLLM_BATCH_INVARIANT=1`.
- `--confirm` (rule + greedy sample first): offline on v4b, same PA with 61 % fewer LLM samples; v9 validates it on GPU.

## Entry 8 — 2026-11-02/03: v6–v9 (train + public), v9i

| Run | Adds | dev | holdout | all | wall | public abstain |
|---|---|---|---|---|---|---|
| v6-1 | rule reader on the cleaned question, parser patterns, toolkit fixes | 0.974 | **0.961** | **0.969** | 5787 s | — |
| v7-1 / -2 | + soft checks, CANON_DOC, aliases | 0.961 | 0.959 | 0.960 | 5838 s | 76 |
| v8-1 / -2 | + import hook (`import datamate`), tolerances | 0.976 | 0.948 | 0.966 | **4641 s** | 108 |
| v9-1 / -2 | + selection by rule programs, `--confirm 1` | **0.989** | 0.809 | 0.928 | 5748 s | **51** |
| v9-3 | public repeat of v9 | | | | | 78 |
| v9i | `VLLM_BATCH_INVARIANT=1` | crashed at ~470 s (not supported): ruled out | | | | |

- Every run from v6 on is at 0.96–0.99 on dev. The total moves with the luck of the sampled mappings of 2–3 holdout
  businesses: in v9, all 6 samples of biz_123 failed (no mapping, PA −0.08) and biz_124 got 0.48.
- `--confirm` works on GPU: 2633 / 3000 train and 657 / 1000 public questions were settled by rule + greedy sample.
  Wall time is dominated by the thinking mapping phase.
- Determinism, public: v9 vs v9-3 gave the same answer on 80.4 % of questions (v5: 88.3 %). Mapping success itself
  is random for dev_014 (mapped in v7 / v9, not in v9-3), and near-tied candidates flip in dev_004 / dev_012.
- Next (v10): when every sample of a business fails, a second batch (`--map-retry`); kc 8 now that selection is better.

## Entry 9 — 2026-11-03: multi-run vote, errors shared by every run, looping thinking, grounding

**Multi-run vote** (`src/vote.py`, `docker/predict.sh` runs up to 3 times while the budget allows): per-question
plurality over independent runs (seed bases 0, 1, 2), abstaining runs give no vote, the top answer needs ≥ 0.4 of the
runs that reached the question. Train, every subset of the runs v6–v9:

| Runs | PA |
|---|---|
| 1 (each) | 0.928 – 0.969 |
| 2 (each pair) | 0.956 – 0.972 (about the mean of the two) |
| 3 (each triple) | 0.976 – 0.981 |
| 4 | 0.980 (2960 correct / 19 wrong / 21 abstain) |

**Errors every run shares** (the vote cannot fix them; 40 non-correct questions of the 4-run vote):
- biz_123 (sqlite, 17 tables): 22 of the 40. Its LLM mappings fail 2 of 3 times; the chosen one in v6 built purchases
  from `chi_tiet_nhap` with `NgayNhapKho` / `MaNhaCungCap`, which are columns of `phieu_nhap`. `frame()` skipped them
  silently, so there was no receipt date and no supplier, and every purchase question abstained. → An LLM mapping now
  gets a warning for each column `frame()` skipped, naming the table that has it. The warning goes into the
  completeness round and into the ranking.
- "hôm qua bán được bao nhiêu ly": all 4 runs answered 0 with `size_qty(p, "ly")` or counted every unit, and the
  rule reader had no pattern. → New rule `unit_qty` → `item_qty(p, base_unit=…)` (rulebook: only items of that base
  unit), plus a `size_qty` fallback.
- "số kg nếp cái hoa vàng": the rule reader read `kg` as the negation "không". → `kg` after a number or a quantity word
  is now the unit.
- One item code with two spellings ("Siro Tiffy chai 30 ml" / "30ml", "Red Bull 250 ml" / "250ml") counted half the
  sales. → Item filters now match every code and spelling of the item.
- Oracle (rules on reference mappings): 2754 / 3 / 243 → **2791 correct / 1 wrong / 208 unparsed**, no question lost.

**Mapping failures** (v6–v9, 600 LLM mapping samples; 375 run):
- OK at round 0 / 1 / 2: 207 / 99 / 69.
- "no python code block" is 153 of the 895 failed attempts: the thinking loops ("Wait, …") up to the 18k-token limit.
  This hits 21 % of the first attempts and 35 % of the greedy sample, and costs about 10 min of an L4 each.
  → StepLLM no longer decodes thinking greedily (temperature 0.6, top-k 20, per the Qwen3.5 card). It also uses budget
  forcing (s1, Muennighoff et al. 2025): at `--think-budget` 12288 tokens, the end of the reasoning is appended and the
  same request continues with the rest of its budget for the program.
- Other fixes:
  - a sales book without order codes gets one record per row (58 attempts stopped at "lines need an order_id");
  - `frame()` accepts functions (13 pickling errors);
  - a KeyError on a skipped column names the missing raw column;
  - table names work as variables.

**Grounding of answer programs**: an LLM sample that names an item the question does not mention was wrong 85 % of the
time on train (99 of 116 samples in v6–v9; other samples: 7 %). Example: "siro tiffy chai 30ml" answered with
"Tiffy Dey hộp 25 vỉ" by all 8 samples, which outvoted the correct rule. Such a sample now neither votes nor counts.
Re-voting the logged runs reproduces 3000 / 3000 answers and gains +0.0003 to +0.002 PA per run, never less.

**v10** (commit 277cfd6: kc 8 + `--map-retry`):

| Run | PA | Correct / wrong / abstain | Wall |
|---|---|---|---|
| v10-1 train | 0.957 | 2927 / 55 / 18 | 5292 s |
| v10-3 public (repeat) | — | 40 abstain (lowest so far); 89.5 % of answers equal the 4-run vote | 4694 s |

- biz_123 recovered (0.72; v9: −0.08).
- biz_110 fell to −0.20 (24 wrong). Both mapping candidates carried the revenue warning (revenue +100 % against
  "Tiền hàng"), so the least bad one was chosen.
- Vote over the 5 train runs v6–v10: **0.9817** (2966 / 21 / 13). Best triple: v8 + v9 + v10, 0.980.
- The mapping phase still takes 1100–2950 s per LLM-mapped public business; v11 targets this with budget forcing.

## Entry 10 — 2026-11-03: v11 (all fixes of entry 9 + sheet alias), v11s

| Run | dev | holdout | all | Correct / wrong / abstain | Wall |
|---|---|---|---|---|---|
| v10-1 | 0.966 | 0.940 | 0.957 | 2927 / 55 / 18 | 5292 s |
| **v11-1** | **0.990** | **0.946** | **0.975** | 2950 / 25 / 25 | **4172 s** |

- v11 is the best single run so far (previous best: v6, 0.969) and 21 % faster than v10.
- Weak businesses in v11: biz_123 0.63, biz_114 0.70 (mapping luck this time), biz_124 0.92.
- Vote over the 6 train runs v6–v11: 0.982. Best triple: v8 + v10 + v11, 0.9823.
- v11s (public `--subset 28`, GPU check of budget forcing and thinking sampling):
  - ran cleanly, the mapping phase was 25 % shorter, and the mapping success rate was unchanged (52 %);
  - whether forcing fired cannot be read from those logs. Attempts now carry `forced`, and the loop is cut before
    forcing (9bb6191).
- After v11 (not yet on GPU):
  - 9bb6191: `--map-repairs` / `--map-repair-think 0`, under test in v12;
  - 69c0d8d: rulebook templates for expiring items and out of stock (oracle 2798 / 1 / 201).
- v11 public (seeds 0 / 1 / 2): 42–45 abstains per run; any two runs give the same answer on 85 % of questions.
  - Vote of the 3 runs: 103 abstain (`submissions/public_v11_vote3.jsonl`).
  - Vote of 5 runs (v10-2, v10-3 + the 3 v11 runs): 34 abstain (`submissions/public_v10-v11_vote5.jsonl`).
  - On train, 5-run votes beat 3-run votes (0.982 vs 0.976–0.982), so vote5 is the first candidate to submit.
- **v12** (`--map-repairs 4 --map-repair-think 0`):
  - public subset: mapping success 64 % (v11s: 52 %), mapping 900–1650 s per business; budget forcing fired on 18
    samples;
  - train: 0.941 (dev 0.991; holdout 0.844, because biz_123 got a wrong mapping, −0.21); mapping success on train
    89 / 112, the same as v11 (87 / 112).
  - Not proven better: the defaults stay as tested (repairs with reasoning, 2 rounds).

## Entry 11 — 2026-11-03 05:30: mapping choice per question (offline study), v13 / v14

- The v11 train errors are almost all mapping errors:
  - biz_123: 35 questions; the rule programs raise AttributeError / KeyError on the chosen mapping, which has no
    purchase or prescription columns;
  - biz_114: 6 questions; an outlier candidate with coverage 14, while the other 7 had 10–12;
  - biz_124: 5 questions.
- **Selection by answer agreement**, study of v9–v12 (56 business-runs; `scratchpad/consensus_study.py`):
  - method: run each candidate's rule programs on the business's questions; a candidate gains W × the share of
    questions where its answer equals the majority answer of the other candidates;
  - mean rule PA of the chosen candidate: 0.919 → 0.930 at W 30 (2 choices changed, none worse);
  - W 100 and 300 made some choices worse;
  - oracle (best candidate): 0.949.
- **Replay of v11 train** (`src/replay.py`, all 30 businesses):

| Strategy | PA | Correct / wrong / abstain |
|---|---|---|
| run (logged answers) | 0.975 | 2950 / 25 / 25 |
| top1 (replay ranking) | 0.967 | 2932 / 32 / 36 |
| vote2 / vote3 (top 2 / 3 mappings voting together) | 0.964 / 0.942 | worse |
| **fb2 / fb3** (next mapping only when the first abstains or says unanswerable) | **0.974** | 2943 / 21 / 36 |

- Implemented (b628bc9):
  - `--sel-agree 30`;
  - `--map-fallback 2`: the programs also run on the next mapping inside the same CPU task, so no extra LLM call;
  - the rule program also runs on the alternate when it fails on the chosen mapping;
  - `--map-vote` stays off.
- Runs:
  - v13 (69c0d8d: expiry / out-of-stock templates), train + public ×3, thnliu / linhmai24;
  - v14 (b628bc9: + agreement + fallback), train + public ×3.

## Entry 12 — 2026-11-03 08:00: public score 0.637 (v13 vote), v14 / v15, industry sections

- **Portal (user): v13 vote of 3 runs = 0.637** on public. Each of v10, v11 and v13 scored a little higher than the
  one before.
  - With 67 abstains, 0.637 means about 785 right and 148 wrong.
  - The train LOO estimate (0.955–0.98) does not carry over: public needs things train never tested.
- The public rulebook adds industry sections that train has none of: Mẹ và bé, Sửa xe, Cửa hàng điện thoại.
  - The shops of those industries are the unstable ones: 3-run agreement dev_012 12 %, dev_014 25 %, dev_011 72 %,
    dev_013 75 %; the other shops 93–100 %. dev_009 has 57 %, from 'Tổng ngày …' summary rows.
  - Seen in dev_012: 'thợ nào hoa hồng cao nhất' was answered by revenue (the commission % is in DanhSachTho);
    'thợ nhiệt tình nhất' was answered by order count ('nhiệt tình' is not defined; 'chăm chỉ' = most working days).
- Changes:
  - a0e5c08 industry sections: the section is shown next to the README; for a section absent from the training
    rulebook, rule weight 1, 2 raw-table programs per question, no --confirm shortcut;
  - 084407b summary rows (biz_110, dev_009, dev_010), now matched with accents;
  - ce0c688 abstain on 'unanswerable' from LLM mappings that left tables unused;
  - 234e2f8 a second mapping batch when only one candidate ran;
  - 4b85565 --kv-dtype fp8 / MTP fallbacks.

| Train run | Code | dev | holdout | all | Wall |
|---|---|---|---|---|---|
| v14-1 | b628bc9 (agreement + fallback) | 0.974 | **0.951** | 0.966 | 4252 s |
| **v15-1** | 7c28f66 (+ rule reader coverage) | **0.999** | 0.949 | **0.982** (2967 / 21 / 12) | **3209 s** |

- v14: fallback answered 20 questions (17 right, 3 wrong); biz_110 was −0.20 because of summary rows (fixed after it).
- v14 public: 38–50 abstains per run; 3-run agreement 89.1 %.
- Running:
  - v15 public, v16 (ce0c688 + 234e2f8);
  - v17 (speed: `--spec-mtp 2 --kv-dtype fp8` vs plain, public subset);
  - v18 (a0e5c08 + 4b85565: industry sections, summary rows, all fixes), train + public ×3.

**v17 (speed test: public `--subset 28`, same code, lelynhan)**

| | Plain | `--spec-mtp 2 --kv-dtype fp8` |
|---|---|---|
| fp8 KV cache | — | **not supported** by vLLM 0.32 for Qwen3.5 (engine init fails; the fallback drops it) |
| KV cache / GPU | 201k tokens | 176k (the MTP drafter takes memory) |
| Load | ~280 s | 806 s (two refused inits) |
| Mapping phase, summed over 10 LLM shops | 11361 s | 10637 s (−6 %) |
| Mapping samples that ran | 55 / 80 | 51 / 80 |
| Wall | 3160 s | 3944 s |

→ Neither is worth it: the defaults stay plain (no MTP, model-dtype KV).

**v16 / v18 / v19 (train) and v18 public**

| Train run | Code | dev | holdout | all | Correct / wrong / abstain |
|---|---|---|---|---|---|
| v16-1 | 234e2f8 | 0.997 | 0.917 (biz_124 0.56) | 0.970 | 2947 / 38 / 15 |
| v18-1 | 4b85565 (industry sections, summary rows) | 0.9985 | 0.944 | 0.980 | 2955 / **15** / 30 |
| **v19-1** | a246b0b (+ unused raw tables in the prompt) | **0.999** | **0.961** | **0.986** | 2972 / 14 / 14 |

v18 public vote: 69 abstains; 3-run agreement up on every shop that was unstable.

| Shop | v13 | v18 |
|---|---|---|
| dev_009 | 57 % | 100 % |
| dev_008 | 62 % | 91 % |
| dev_012 | 12 % | 60 % |
| dev_013 | 75 % | 84 % |
| dev_014 | 25 % | 47 % |

'Unanswerable' answers in the new-industry shops fell: dev_012 3 → 0, dev_013 4 → 0, dev_014 20 → 15.
Submissions: `submissions/v15`, `submissions/v18` (pushed); v19 when its public runs end.

## Entry 13 — 2026-11-03 12:00: public scores v15 / v18 / v19, where the public errors are, parser and mapping-check fixes

**Portal scores (user).** PA = right − wrong over 1000, so with the abstain count known:

| Submission | PA | Right | Wrong | Abstain | Unanswerable answers |
|---|---|---|---|---|---|
| v13 vote | 0.637 | 785 | 148 | 67 | 61 |
| v15 vote | 0.634 | 792 | 158 | 50 | 53 |
| v18 vote | 0.669 | 800 | 131 | 69 | 48 |
| **v19 vote** | **0.687** | **801** | **114** | 85 | 30 |

- Right answers are flat at ~800 since v15; every gain since came from **fewer wrong answers**. v18 → v19 is mostly
  dev_014: 15 'unanswerable' answers (the mapping had left out the trade-in / instalment tables) became abstains.
- The versions answer **the same way on almost every question**: v18 vs v19, only 4 questions both answered differently.
  The ~114 remaining errors are therefore shared by every run: a vote over more runs cannot remove them.
- v19 3-run agreement: 820 unanimous, 95 majority (2 of 3), 85 split (abstain). The split / majority questions are in
  dev_003 (35 majority), dev_009 (9), dev_012 (28), dev_013 (13), dev_014 (48).
- The 30 'unanswerable' answers of v19 look right by rulebook §7 (no group column in dev_002 / dev_004, no customer
  gender / street / cancel reason in any shop; subjective or medical questions).

**Where the shared errors can come from (no labels; found from the logs).**
1. Rule programs that misread the question (55 public questions where every LLM sample disagreed with the rule; the
   LLM won 42 and 13 abstained): slang ('10 củ', '3tr7', 'o đờ', 'bq', 'đh', 'cất hàng'), top-N after the period,
   reversed growth for 'so với A thì B', several months answered with the first one, an unlinked item / name answered
   with the shop's revenue. → general parser fixes (d0007a9); train oracle unchanged (2967 / 1 / 32).
2. Mapping errors no check saw. dev_003, one LLM mapping: order revenue minus the order voucher on 34 of 61k orders
   (every monthly revenue answer off by 10–30k VND) and order_time without the time of day (afternoon-shift revenue 0).
   → new soft checks: revenue vs the sum of the order's line amounts (penalty 100) and order_time all 00:00 while the
   data has times (penalty 10); shift / peak hour raise on such data instead of answering 0. On the 195 candidate
   mappings of the 3 v19 public runs they flag exactly that mapping (chosen for dev_003 in 1 of the 3 runs; the vote
   had overruled it) and 2 dev_010 candidates that were not chosen; 36 train references: 0 flagged.
3. Reference transfers that ignore the shop's README conventions (dev_002 toppings in the dish cell, dev_005 chained
   unit factors, dev_006): v20 (`--ref-adapt 3`, de097f9) rewrites the reference for these gaps.

**Runs.** v20 = de097f9 + 5dba631 (weekday averages per the public rulebook), train + public ×3 (caotienminh1,
duylnng). v21 = d0007a9 (parser fixes + mapping checks), launched next.

**v20 train (12:27; 5dba631 = ref-adapt + weekday averages; caotienminh1).** PA **0.991** (2975 / 2 / 23): dev 0.998
(1974 / 1 / 1), **holdout 0.977** (1001 / 1 / 22; v19 0.961). Holdout shops are unseen, the closest proxy for public: the
README-gap adaptation of reference mappings is what moved. Wall 3844 s; GPU busy 84–90 %, peak 21.6 / 22.5 GB.

**More mapping checks found on public (a052d46, e7f79fd; train references 0 flagged, oracle unchanged).**
- dev_014: lines keep the sales book's abbreviations ('CL 9D') although a lookup sheet maps them to full names + type
  (every chosen mapping of the 3 v19 runs) → warning "lines.item_name holds the short names of …" (penalty 20).
- dev_012: mechanics written 'Đạt' / 'a Phúc' / 'A.Phúc' while DanhSachTho lists the full names (most candidates) →
  warning "orders.staff holds given names …" (penalty 10); dm.person_key drops honorifics so a full name matches every
  spelling of the same person (1555 tickets instead of 523).

**v20 public (13:22).** Vote of 3 runs → `submissions/v20/` (117 abstain, 22 unanswerable). Each run 6762–7071 s.
3-run agreement 788 unanimous / 95 majority / 117 split. Against v19's submission: 918 identical, 52 newly abstained
(dev_003 18, dev_009 9 — mapping choice differing between runs; v21's revenue-vs-lines / README-revenue checks target
exactly these), 20 newly answered, 10 changed. The README adaptation changed no answer of the reference-mapped
public shops (dev_005 identical, dev_002 one change): its train holdout gain does not show on public. Expected portal
score ≈ v19's; v21 is the candidate.

**Portal: v20 = 0.650** (user, 14:10) → ~767 right / ~117 wrong / 117 abstain (v19: 801 / 114 / 85). The loss is the 52
questions v20 abstained on where v19 answered (mostly right in v19): LLM-mapped shops whose 3 runs chose 3 different
mappings, so the vote split. dev_009: the 3 v20 runs read revenue 3 ways (run 1 = v19's majority; runs 2 / 3 +1.5 % / −0.9 %);
v21's README-revenue check flags exactly runs 2 / 3's mappings (penalty 100) and not run 1's. Lessons:
- the public score now moves ±0.03 with mapping luck alone, as much as a real fix → mapping choice must not depend on which
  samples a run drew (checks that reject wrong readings; pool candidates across runs);
- the vote should not count a run whose mapping is detectably wrong as much as one whose mapping is clean.
- caotienminh1 reached the weekly GPU quota (30 h); v22-1 / v22-2 moved to thnliu.

**14:25 — handoff.** Mapping-aware vote (7fa70f2): splits go to the one run whose chosen mapping is clearly cleaner
(penalty gap ≥ 10); on v20 offline 15 splits resolved, 10 to v19's answer. v21 (8dca4a2) and v22 (20cfec7) still running
on Kaggle at the end of this chat; their outputs must be downloaded in the next one (handoff/STATUS.md lists the kernels
and the submission recipe). Routine disabled.

## Entry 14 — 2026-11-03 16:00: v21 / v22 results (new chat)

- **v21** (8dca4a2): train 0.984 (2965 / 13 / 22; dev 0.991, holdout 0.971). Public 3 runs: 856 unanimous / 72 majority /
  72 split (v19: 820 / 95 / 85). Vote with mapping-aware tie-break → `submissions/v21` (70 abstain, 42 unanswerable, 11
  zero answers). Vs v19: 927 identical, 37 newly answered (dev_014 19), 22 newly abstained (dev_012 7), 14 changed.
- **v22** (20cfec7): train 0.988 (2969 / 5 / 26; dev 0.998, holdout 0.969; wall 4136 s). Public: 856 / 68 / 76 → `submissions/v22`
  (76 abstain, 33 unanswerable, 3 zero answers: the regional-word fixes removed most 0 answers). Vs v19: 929 identical,
  27 newly answered, 18 newly abstained, 26 changed (dev_012 9, dev_008 5, dev_013 5). Vs v21: 937 identical.
- Pooled mapping choice (`automation/jobs/pool_study.py`, v19 public, current checks): in 7 of 10 LLM-mapped shops the 3
  runs' choices fall in the same rule-answer cluster; the pooled ranking sometimes picks a singleton cluster (dev_008,
  dev_014), so pooling with the present ranking is not obviously more stable. Train (labelled) study running.
- Waiting for the user's portal scores of v21 / v22 before any new run (user instruction).

## Entry 15 — 2026-11-03 19:00: portal v22 = 0.71; v23 (industry toolkit); v24 fixes

**Portal (user): v22 = 0.71** (best so far; v19 0.687). User: continue from v22, two more versions, then stop for scores.

**v23 = 92dd902: toolkit for the public rulebook's new industries** (`src/industry.py`, mixed into `dm.Shop`; tests
`tests/test_industry.py`). Phone shops: devices vs accessories, model lines (`…GB`), brands, trade-ins, khách phải trả,
instalments (remaining / overdue / terms due / overdue customers) counted in the debts, warranty visits and rate. Repair
shops: tickets (warranty included), jobs of the 'Làm gì' cell (taxonomy with the rulebook's plain 'nhớt' / 'ruột' / 'sửa điện'),
plates, vehicle-line spellings, regular vehicles, mechanics across spellings, commission table. Mother & baby: vouchers /
points / shipping, new members, diaper pack shares, brands. The industry's canonical tables go into the MAPPING prompt and the
toolkit into the ANSWER prompts of shops of that industry only; `industry.detect()` finds the industry from the data when the
README does not name it (dev_010; no train shop detected). Train oracle 2967 / 1 / 32 unchanged.
- Train (linhmai24): 0.978; only biz_123 fell (0.48: 15 of 16 mapping attempts failed on column names, known hard shop).
- Public 3 runs: 834 unanimous / 101 majority / 65 split; the LLM mappings now build the industry tables (dev_014:
  instalments, instalment_payments, tradeins, warranty; dev_012 staff; dev_013 customers). → `submissions/v23` (49 abstain,
  v22 76). Vs v22: 903 identical, 48 newly answered (dev_014 29), 21 newly abstained, 28 changed.
- Ops: phuongnehe and nguynduylm out of weekly GPU quota; the old fanout failover only tried the listed accounts and raced a
  manual relaunch (two sessions of zc-v23-3 on hai1108). Fixed in fanout (all accounts with a token, quota remembered per week).

**Found on the v22 / v23 public logs (no labels), fixed generally in v24:**
- subjective qualities the rulebook does not define ('ngon nhất', 'thân thiện', 'khó tính', 'nhiệt tình', 'uy tín', 'đáng tin
  cậy', 'trung thực', 'chất lượng tốt') → unanswerable (§7); 9 public questions, 0 train; v22 answered 6 with a name.
- debts written only as free-text notes (dev_008: 5 debt questions 'unanswerable'): `debt_from_note`, `orders.debt_end`,
  soft check against what C records (a v23 mapping made payments = the order total).
- `n_orders(p, status='huỷ')` always 0 (valid=True first): status filters look at every status; `prev_month(this_month)` =
  same days of last month (rulebook); 'năm nay tính tới hôm nay' = ytd; 'th7, th8, th9/2026' = three months; a year read as
  top-k; 'o đờ' = đơn, 'xi ô đi' = COD, 'quan thu về' = quán; 'Tại tiệm' = tại quầy; regional 'pho mát' = 'phô mai', 'xây trát' =
  'xây tô' (and the grounding check compares regional forms: 6 right samples of dev_004 were dropped as ungrounded).
- repair mapping that hand-wrote a wrong nickname → full-name map (dev_012): (day, person) check against the book, aliases for
  raw staff lists; raw-table samples of implemented industries removed and their standard rules keep weight 4 (dev_010
  confidence 0.98 → 0.84 in v23); rule templates for the industry functions (weight 2), deliveries by order date, due dates;
  customer-group check (dev_008); second mapping batch when every candidate has a heavy warning (dev_014 run 3).

## Entry 16 — 2026-11-03 22:40: v24 (3 public runs, thnliu) → submissions/v24; invoice adjustment rows

- v24 code f3d21c9 (entry 15 fixes). Most accounts were out of weekly GPU quota (caotienminh1, phuongnehe, nguynduylm,
  lelynhan, linhmai24, duylnng, hai1108, huyenco, mucnokieu; mxuanquy T4 only, nguyenmanh0404 rules not accepted): the 3 public
  runs ran on thnliu, no train run. **A public run now takes 65–70 min** on 4×L4 (v23 ~2 h): no raw-table samples and the
  confirm shortcut for the implemented industries.
- 3-run agreement **903 unanimous / 71 majority / 26 split** (v22 856 / 68 / 76, v23 834 / 101 / 65); per-run abstains 14 (v23 36).
- Found on the v24 logs, fixed generally: (1) dev_014 seed 1 chose a mapping without the instalment book (all instalment
  answers 0) → check for industry books present in T but missing from C (penalty 20); (2) dev_013 seed 1 assigned the WHOLE
  table's voucher / shipping total to every order and the first order's channel to all → check for order columns holding one
  value for every order (penalty 20); (3) **dev_013 revenue wrong in every version since v19**: the rows 'Voucher', 'Dùng N điểm',
  'Phí ship' (README: not items) stayed in the lines, so revenue subtracted vouchers / points and added shipping (§1: neither)
  and every voucher / ship / points question was 'unanswerable' → `finalize` moves such rows to orders.order_discount /
  points_discount / shipping_fee (only dev_013 has them among the 44 shops; oracle 2967 / 1 / 32).
- `submissions/v24`: vote of the 3 runs with dev_013 replayed offline (`automation/jobs/replay_biz.py`: the logged programs
  on the rebuilt mapping, vote redone; unaffected questions keep their logged answers) and map penalties of the latest
  checks: 14 abstain, 33 unanswerable, 0 zero answers. Vs v22: 855 identical, 66 newly answered (dev_014 35), 4 newly
  abstained, 75 changed (dev_013 32). Vs v23: 879 identical.
- **Stopped here (user): waiting for the portal scores of v23 / v24.**

## Entry 17 — 2026-11-04 05:50: portal v23 = 0.87, v24 = 0.747, v25 = 0.825; v25 / v26; consensus re-answer

- **v25** (commits 98d601f + f4be0c0; 6 kernels on loinguyentran / anderson1202 / phatle1578, 55–75 min each): train seed 0
  **PA 0.9893** (2973 / 5 / 22 — best train run); 5 public runs, 881 questions unanimous over the 5. `submissions/v25` =
  5-run vote (5 abstain). **Portal 0.825** (v23 0.87).
- **Consensus re-answer** (`src/consensus.py`): candidates of every run pooled, one mapping per business, every logged
  program re-executed on it. First try on the v23 runs: 945 same as v23, but wrong debt mappings chosen (dev_014: payments =
  debt-book repayments only -> 6334 orders owing 22.9 bn; dev_008: 233 M for 4.55 M in the notes). Fixed generally:
  checks 1a-owing (> 25% of orders owing; train refs <= 0.4%), settled notes still owed, unreadable debt book, returns /
  cancels in notes kept valid (1a-status, with an order-id column that must have many distinct values), plates dropped,
  empty item names; toolkit paid_until = paid_at_purchase + repayments; cleaning layer `resolve_short_names` (short names
  differing only in case / accents -> the lookup's full name); **topic slots**: debt / instalment / vehicle / staff
  questions use the best candidate without a heavy warning of that topic (none -> abstain), the general ranking leaves topic
  penalties out; identical logged programs executed once (-38% time). Oracle 2967 / 1 / 32, plausibility 0 / 36, ALL PASS.
- **v26** = topic-aware consensus over the 5 v25 runs (33 min CPU): vs v25 967 same, 28 changed (dev_014 19, dev_011 7),
  4 newly abstained; 8 abstain. Hand checks on the raw books (8 questions where versions differ): v23 0/8, v25 6/8, v26 7/8.
- **Open contradiction**: the 3 portal scores imply that the 62 questions where v24 = v25 != v23 lose >= 27 points for
  v25, yet every group I could recompute from the raw data and the rulebook favours v25 (dev_008 note debts, dev_013
  voucher / ship / points with §8 de-duplication, dev_004 credit due dates by §2 'tính đến hôm nay'). Either part of these
  readings differs from the organisers' or the portal scores a subset. `submissions/diag/X1–X3` (v23 + v26 answers of one
  rule group each) settle it with three portal scores.
- **Stopped (user): waiting for the portal scores of v26 (and X1–X3 if the user submits them).**

## Entry 18 — 2026-11-04 09:00: portal v26 = 0.825, X1 0.793, X2 0.756, X3 0.768 → the v23 score is not comparable

- **X2 = v23 + 6 changed answers scored 0.756** (v23 had scored 0.87): 6 answers move the score by at most 0.012, so the
  answer key / scored set changed after the v23 score. With today's scores (groups disjoint, covering all 91 v23 -> v26
  differences): v23 now ≈ 0.746–0.755; X1 rules (dev_004 due dates / deliveries, dev_008 note debts, dev_013 voucher / ship /
  points) **+0.038..+0.047** (almost every change right, as the raw-book checks said), subjective -> unanswerable
  **+0.001..+0.010**, phone / repair toolkit **+0.013..+0.022**. Every rule group of v24-v26 is net positive.
- v26 = v25 (0.825): the consensus fixed dev_011 (7 answers; raw book with forward-filled order numbers: used phones 26 / 34
  / 19, charger 6,825,000, Redmi Note 14 = 6, iPhone 16 Pro Max = 7) and dev_014 short names / impossible debts, but picked
  for dev_014 a mapping without device kinds (accessories as phone models, 0 used phones) and for dev_008 one with all costs 0.
- Fixes (all general): device-kind check against the lookup table (1a-kinds), zero catalogue costs (1a-cost) + stock_value
  needs recorded costs, empty item names (1a-noname), short-name cleaning layer, named vehicle models / phone brands keep the
  filter in the rule reader (dev_012: 'xe Air Blade khác nhau' = 368 = every vehicle), vehicle topic from the question text,
  consensus ranking with rule failures on non-topic questions and agreement among plausible candidates only.
- Per-question agreement among plausible candidates is >= 0.67 on almost every question: most remaining errors are
  systematic (every mapping agrees), not mapping luck.

## Entry 19 — 2026-11-05 02:30: v27 / v28 (dev set 37 câu), bàn giao PDF 4

- Kiểm tra / sửa sau entry 18 (đều chung, 36 ánh xạ tham chiếu train 0 cảnh báo sai, oracle 2967 / 1 / 32): biển số không
  khớp sổ (vehicle topic); cột bảo hành toàn False bị bỏ qua; slot chủ đề xếp theo khả năng trả lời câu của chủ đề, dự phòng
  4 slot, câu không chủ đề mà 2 ánh xạ đầu không trả lời được dùng thêm ứng viên hợp lý có chương trình luật trả lời được;
  sổ nợ còn nợ mà C không nợ; ngày trả đọc mềm (dòng tiêu đề lẫn vào làm hỏng hàm nợ và che cảnh báo).
- v27 lần đầu (gửi user) làm hỏng 6 câu ngày phải trả dev_004 (slot chủ đề lấy ứng viên không có ngày phải trả) → bản thay
  thế 36 / 37 dev set. v28 = consensus 8 lần chạy (v25 ×5 + v28 seed 5 / 8 / 9; seed 6 / 7 không chạy vì hết quota):
  **37 / 37 dev set**, 10 abstain (3 câu tổng nợ dev_014: mọi ứng viên thiếu sổ nợ hoặc sổ góp), so với v27 998 câu giống.
  Train v28 (code mới): PA 0.9897 (2977 / 8 / 15).
- Lệnh v28: `python3 src/consensus.py data/raw/public_test/public_test <cache_pub> <work> <work>/submission.jsonl
  <8 thư mục results> --workers 3` (≈ 85 phút) → `automation/make_submission.py`.
- Bàn giao: `handoff/04_SESSION_0411.{md,pdf}`, công cụ `automation/jobs/analysis/`, `goldkey/` (dev set + đáp án tham
  chiếu đang làm theo yêu cầu user 05/11; chỉ để chấm offline).

## Entry 20 — 2026-11-05 13:30: khung tất định (automap + qnorm + qparse + toolkit) — train 3000 / 3000, public 1000 / 1000

- **Kiến trúc mới (rule-first):** `src/automap.py` ánh xạ bảng thô → C từ README (mô tả cột, định nghĩa in đậm) + từ điển
  tên cột (khớp đúng, rồi khớp tiền tố) + hồ sơ giá trị (trạng thái, PTTT, kênh, SĐT, mã) + đồng nhất thức tiền của đơn
  (đã trả = tổng − nợ …); phân loại bảng theo vai trò cột (đơn, dòng, phiếu nhập, lô, thanh toán, sổ góp, sổ nợ, phiếu sửa
  …). `src/qnorm.py` làm sạch câu hỏi; `src/qparse.py` đọc câu → chương trình trên toolkit `dm` / `industry`; bộ chặn độ
  chính xác (`consistent`, thiếu thông tin → unanswerable). Không dùng ánh xạ tham chiếu train, không LLM.
- **Điểm (chấm nghiêm, `tests/eval_split.py`):** train 3000 / 3000, public 1000 / 1000; cả dưới pandas 2.3.3 + calamine
  (phiên bản của image Docker) lẫn pandas 3.0.5.
- **Kiểm tra tổng quát hoá (`tests/eval_all.sh`, câu biến đổi bằng `src/perturb.py`, không dùng để khớp):** README bị xoá
  trắng: train 0.862 → 3000 / 3000 (sau các quy tắc độc lập README), public 0.991; câu không dấu 0.981; nhiễu chat 3 seed
  0.948 / 0.957 / 0.962 — phần thiếu chủ yếu là abstain (câu chuyển cho LLM ở stage 2).
- **Sai dưới nhiễu → sửa chung:** từ dính không tách được (`cấthàng`, `củtrở`, `huybao`, `vebnhiu`, `hàngkokê`) →
  `glue_segment` (token ≥ 2 âm tiết chưa biết → tách thành từ đã biết; token không dấu cần từ viết tắt / Telex hoặc từ
  ≥ 3 chữ phổ biến để không tách từ ngoại `soda`, `combo`); từ vùng miền 2 chữ (`ke` = kệ) khi khớp nguyên tên hàng;
  câu tiền trả góp phải có từ hỏi tiền.
- **Luồng nộp mới (`docker/predict.sh` → `src/predict_flow.py`):** stage 1 `src/rules_first.py` (CPU, public 91 s với 3
  worker: 1000 / 1000 trả lời, 0 chờ) — mọi đáp án luật là cuối cùng; stage 2 chỉ cho câu luật bỏ lại: `agent.py`
  (Qwen3.5-9B, mô hình duy nhất, 9B ≤ 15B) với `--prefer-map fmt:automap --adj-rows move`, bỏ phiếu nhiều lần chạy, gộp
  lại mỗi 20 s (`src/merge_sub.py`, luật trước).
- **Kaggle:** trọng số Qwen3.5-9B của các acc cũ là dataset private (403 với acc mới) → dùng Kaggle Model
  `danbth/qwen3-5-9b` (bản sao HF) và `run_kaggle.py` kiểm sha256 từng file so với Hugging Face (`src/model_hashes.json`)
  khi chép, sai là dừng. Job xác nhận GPU: `automation/jobs/v30.jobs` (luồng nộp trên câu public / train biến đổi: đo chất
  lượng LLM ở đúng các câu luật bỏ lại); CPU: `v30cpu.jobs` (luật trên môi trường Kaggle + seed nhiễu mới chưa dùng khi
  phát triển: ước lượng không thiên lệch).

## Entry 21 — 2026-11-05 17:45: paraphrase (ước lượng tổng quát hoá thật), các bộ chặn, Kaggle v30, file nộp v30

- **Paraphrase bằng Qwen3.5-9B** (`src/paraphrase_llm.py`, Kaggle yuan1103, trọng số kiểm sha256 với HF): public 3000 →
  2065 giữ (kiểm số / ngày / mã / tên); train 6000 → 5689. Chấm bằng đáp án câu gốc (`eval_split.py --perturb-q para:`).
  Quy trình chống khớp: chỉ phân tích lỗi của biến thể k=0 public (dev); k=1,2 public và **toàn bộ train là held-out**.
- **Kết quả (chỉ luật, chưa có LLM):** lần đầu public 0.497 (1254 / 227 / 584) — kém xa 100 % canonical: bộ đọc câu khớp
  với cách hỏi đã thấy. Đọc 74 lỗi dev: ~2/3 do paraphrase đổi nghĩa ('dh' = đơn hàng thành doanh thu, 'Th nay' thành
  'Hôm nay', 'Quy 2/2026' thành tên người), ~1/3 lỗi thật, đều là lớp chung:
  - **Bộ chặn loại đáp án** (`qparse.answer_kind_ok`): hỏi 'nào … nhất' mà trả số, '% / tỷ lệ' mà trả đếm, 'bao nhiêu
    tiền' mà trả đếm, 'bao nhiêu mặt hàng' mà trả tên, có 'N ngày / đầu năm' mà dùng hôm nay, 'bao nhiêu phần' mà trả doanh
    thu, doanh thu không có từ bán, 'thu nhập' đọc thành nhập hàng, 'chia cho số đơn' đọc thành tổng.
  - **Bộ chặn tên không dùng** (`qparse.names_used`): câu nêu nhân viên / khách / mặt hàng mà chương trình không lọc theo.
  - **Cách hỏi chung:** khoảng 'từ thời điểm bắt đầu năm nay cho đến hôm nay', 'tính ngược lại N ngày', 'năm 2025 tháng 5';
    số sau 'tháng' không phải top N; nhóm sau 'món / các / loại'; xếp khách theo 'số lượng đơn'; 'tiêu thụ' = bán, 'chi trả /
    phải thanh toán' = phải trả, 'thu được máy cũ' = thu máy cũ; **lỗi 'sàn' khớp 'sản phẩm'** (lọc nhầm kênh TMĐT).
- **Sau sửa (canonical vẫn 3000 / 3000, 1000 / 1000):** public dev 0.533 → 0.588, held-out 0.479 → 0.506; **train held-out
  0.6655 → 0.6692** (4084 / 277 / 1328). Phần chênh lớn nhất là abstain (~23–30 %) → stage 2.
- **Kaggle v30 (quynhtran35, 4×L4, code trước các sửa trên):** public 4000 câu biến đổi: luật trả lời 3907, còn 93 cho
  mô hình (luật 179 s); train 6000: 5965 / 35 (116 s); mỗi lần chạy mô hình 18–25 phút (chủ yếu khởi động + ánh xạ).
  Chấm đáp án của mô hình: chờ tải file (API Kaggle giới hạn tốc độ vì 6000 file log nhỏ → stress_flow nay chỉ lưu
  submission + run.log). CPU v30c: chết vì `nvidia-smi` không có trên máy CPU → sửa `run_kaggle.py`, preflight thêm bước
  chạy qua wrapper như phiên CPU.
- **File nộp `submissions/v30/`:** stage 1 trên public (môi trường image), 1000 / 1000, định dạng 0 lỗi.
- Tiếp: v31 = luồng nộp đầy đủ (luật + Qwen3.5-9B) trên paraphrase public + train (dataset private `quynhtran35/zc-para`):
  độ chính xác của stage 2 trên đúng loại câu mà luật bỏ lại.

## Entry 22 — 2026-11-05 18:45: kết quả Kaggle v30 (luồng nộp đầy đủ) + CPU v30d; stage 2 nhanh hơn

- **Luồng nộp đầy đủ trên câu biến đổi (v30, quynhtran35 4×L4, code trước các sửa paraphrase):** stage 2 (Qwen3.5-9B)
  trên câu luật bỏ lại: **104 đúng / 14 sai / 10 abstain (81 % đúng)**. Toàn hệ thống: public không dấu 0.984 → **0.993**,
  nhiễu 1 / 2 / 3: 0.968 / 0.974 / 0.975 → **0.990 / 0.989 / 0.991**; train nhiễu 1: 0.986 → **0.996**, không dấu 1.000.
  Tăng ngưỡng abstain của stage 2 không có lợi (giữ chia sẻ phiếu ≥ 0.5: net +90; ≥ 0.75: +87; ≥ 0.9: +76).
- **CPU v30d (quynhtran35, môi trường Kaggle pandas 2.3.3 / numpy 2.1.3 / calamine):** luật public **1000 / 1000** (346 s
  trên 4 lõi Kaggle), train **3000 / 3000** (727 s). Seed nhiễu mới chưa dùng khi phát triển: public 4 / 5 / 6 =
  0.966 / 0.970 / 0.972, train 2 / 3 / 4 / 5 = 0.987 / 0.987 / 0.983 / 0.985 (0–4 sai mỗi bộ; phần còn lại abstain → stage 2).
- **Quy tắc giữ thay đổi:** với stage 2 đúng ~81 % / sai ~11 %, một luật mới chỉ nên lấy câu khỏi stage 2 khi độ chính xác
  của nó trên câu held-out ít nhất như vậy (từ đồng nghĩa doanh thu: 83 % đúng trên train held-out — giữ; 12 / 12 câu
  chuyển sang 'nhập hàng' sai → tìm ra bộ chặn 'thu nhập' bị slang làm mất tác dụng, sửa).
- **Stage 2 nhanh hơn:** `--prefer-map` chạy ánh xạ ưu tiên trước, chỉ chạy ~40 chương trình tham chiếu khi nó lỗi.
- quynhtran35 hết 30 h GPU tuần (18:00) → v31 (luồng đầy đủ trên paraphrase) chạy ở phatle1578 (dataset private
  `phatle1578/zc-para`).
- Bộ kiểm tra cục bộ (code mới nhất): canonical 3000 / 1000; không dấu 0.989; nhiễu 0.971 / 0.974 / 0.978; README trắng
  public 0.993, train 1.000; paraphrase public 0.541, train held-out 0.677 (chỉ luật).

## Entry 23 — 2026-11-05 21:40: v31 (luồng nộp đầy đủ trên paraphrase public), ngưỡng tin cậy stage 2

- **v31 public (phatle1578, 4×L4, code trước LookupError / prefer-first):** 2065 paraphrase; luật trả lời 1450 (134 s), mô
  hình nhận 615; 3 lần chạy mô hình, tổng 12 336 s. Kết quả: k=0 0.588 → **0.640**, k=1 0.513 → 0.577, k=2 0.498 → 0.539.
  Stage 2 trên các câu này: 324 đúng / 215 sai / 76 abstain (yếu hơn nhiều so với câu nhiễu: 81 % đúng) — một phần "sai" là
  paraphrase đổi nghĩa (mô hình trả lời đúng câu đã đổi).
- **Ngưỡng tin cậy:** bỏ đáp án mô hình khi trung bình chia sẻ phiếu của các lần chạy < 0.75 (`vote.py --min-conf`, chọn
  trên dữ liệu dev = v30 nhiễu + paraphrase k=0: net +126 → +137). Kiểm lại offline trên log đã tải: paraphrase public
  0.640 → **0.660** (k=0), held-out 0.577 → 0.588 (k=1), 0.539 → 0.569 (k=2); bộ nhiễu / không dấu không đổi (±0.003).
  `predict_flow.py` nay luôn bỏ phiếu qua `vote.py` (cả khi chỉ 1 lần chạy).
- **An toàn rulebook:** câu nhắc thuật ngữ mà rulebook của split thêm mới (không có trong rulebook train / public) → mô hình
  (`rules_first.changed_terms`); dòng chỉ viết lại vẫn để luật.
- Đang chạy: v31 train (phatle1578), v32 = luồng mới nhất trên paraphrase public + train (yuan1103).
- **Ước lượng sạch hơn (bỏ paraphrase đổi kỳ thời gian — gần như toàn đổi nghĩa: 'hôm qua' → 'ngày mai', 'Th 4/25' → 'thứ 4
  ngày 25', '3/26' → '26/3'):** dev k=0 (591 câu) luật 0.689 → **toàn luồng 0.790**; held-out k=1,2 (1173 câu) 0.610 →
  **0.705**. Phần đổi kỳ: âm (đáp án gốc không còn đúng với câu đã đổi). Vẫn còn đổi nghĩa trong phần giữ kỳ (doanh thu →
  lãi, số đơn → số tiền), nên độ chính xác thật với cách hỏi mới cao hơn các số này. Câu cách hỏi như train / public: ~1.0;
  gõ nhiễu / không dấu: 0.99.
- **v31 train (held-out hoàn toàn, 5689 paraphrase):** luật trả lời 4360 (151 s), mô hình nhận 1329; 3 lần chạy, tổng 15 389 s
  (4.3 h). Luật 0.669 → toàn luồng 0.752 → **+ ngưỡng 0.75: 0.768** (4877 / 510 / 302); phần giữ kỳ (5371): 0.707 → 0.806 →
  **0.816**. Mô hình trên câu còn lại (ngưỡng 0.75): 793 đúng / 234 sai / 302 abstain. Ngưỡng 0.75 cũng có lợi trên tập
  held-out này (chọn trước trên dev) → giữ.

## Entry 24 — 2026-11-06 00:05: dữ liệu xuất kiểu khác (đổi tên cột) — public 1000 / 1000, train 3000 / 3000

- `eval_split.py --perturb-data rename:<seed>`: mỗi cột có tên mang **một** vai trò trong từ điển automap được đổi sang tên
  khác cùng vai trò (cột `__unit` đi theo), thứ tự cột xáo trộn, README giữ tên cũ — mô phỏng một shop mới xuất file theo
  cách khác. Lần đầu: public 0.998; lỗi duy nhất: tên nhà cung cấp tra trong danh sách tên cột cố định → nay chọn theo giá
  trị (cột văn bản duy nhất nhất, không phải mã / SĐT). Sau sửa: **public 1000 / 1000, train 3000 / 3000** (2 seed public).
- Thêm vào `tests/eval_all.sh`. Container khởi động lại giữa chừng (OOM khi train chạy 4 worker + bảng sao chép): chạy train
  rename với 2 worker; poller Kaggle v32 gắn lại bằng `run_kernel.py --attach`.

## Entry 25 — 2026-11-06 00:30: v32 (code mới nhất) trên paraphrase public

- yuan1103, 4×L4: luật trả lời 1472 / 2065 (137 s), mô hình nhận 593 (v31: 615); 3 lần chạy, tổng 11 932 s (v31: 12 336 s —
  prefer-first tiết kiệm ít: thời gian chủ yếu là sinh token, không phải ánh xạ).
- Toàn luồng + ngưỡng 0.75 (chấm lại offline): dev k=0 **0.660 → 0.681** (giữ kỳ 0.790 → 0.814); held-out k=1,2
  **0.579 → 0.584** (giữ kỳ 0.705 → 0.712). Cải thiện nhỏ nhưng cùng chiều trên cả dev và held-out.
- Quota GPU tuần W45 gần như hết ở cả 3 acc L4 (một phiên 4×L4 tiêu ~4 h quota / giờ) → stage 1.5 (v33) chờ tuần sau.
- **v32 train (held-out hoàn toàn):** luật trả lời 4426 / 5689 (135 s), mô hình nhận 1263 (v31: 1329); 3 lần chạy, 14 011 s.
  Toàn luồng + ngưỡng 0.75: **0.768 → 0.780** (4910 / 474 / 305); giữ kỳ **0.816 → 0.829**. Các thay đổi của ngày 05/11
  (LookupError, prefer-first, thuật ngữ mới, từ đồng nghĩa, bộ chặn) cải thiện cùng chiều trên dev và held-out.
- 06/11 03:05: cả 3 acc L4 hết quota GPU tuần W45 (đẩy v33 lên yuan1103 bị từ chối). Khi reset: `python3 automation/fanout.py
  <S>/runs/v33 zc-v33 automation/jobs/v33.jobs 2 --accounts <acc> --l4 --ds <acc>/zc-pend --ms danbth/qwen3-5-9b/transformers/default/1
  --ks nhatlong1103/zc-assets`, rồi chấm: `canon_to_para.py <out>/canon_log.jsonl para.jsonl` + `eval_split.py --perturb-q para:para.jsonl`.

## Entry 26 — 2026-11-07 01:10: v33 (stage 1.5) — luật trên câu viết lại vượt mô hình → bật mặc định (`--canon 1`)

- Quota reset 07/11 00:00 UTC. yuan1103, 2 phiên L4 (1 GPU dùng): `canon_llm.py` viết lại câu chờ của v31 (public 615,
  train 1329) theo văn phong câu train; 700–743 s / phiên gồm nạp mô hình. Qua kiểm (giữ số / mã / tên viết hoa): public
  527 / 615, train 1256 / 1329.
- Luật đọc câu viết lại (`eval_split --perturb-q para:`, chấm theo đáp án câu gốc): public 163 đúng / 34 sai / 330 abstain,
  train 722 / 64 / 470. **Độ chính xác khi trả lời:** train (held-out hoàn toàn) 0.938 (k=0) / 0.902 (k=1); public dev k=0
  0.80, held-out k=1 / k=2 0.853 / 0.825 (một phần "sai" là paraphrase đã đổi nghĩa, mô hình cũng bị tính sai ở đó).
- So với mô hình (v32, ngưỡng 0.75) trên cùng câu chờ, chính sách "luật trên câu viết lại trước, mô hình cho phần còn lại":
  public dev k=0 net 54 → 54; held-out k=1 49 → 59, k=2 52 → 64; train held-out 584 → **706** / 1263 câu chờ.
  Khi cả hai cùng trả lời mà khác nhau: luật đúng / mô hình sai 10 vs 7 (public), 32 vs 8 (train).
- Toàn luồng (ước lượng từ v32 + chính sách trên; câu chờ của v32 không có bản viết lại tính như v32): train held-out
  **0.780 → 0.801**; public held-out k=1,2 **0.584 → 0.600**; dev k=0 0.681 (không đổi). Stage 2 nhận ít câu hơn
  (public 593 → ~417, train 1263 → ~537) → nhanh hơn.
- Đạt tiêu chí CLAUDE.md (held-out ≥ ~85 % khi lấy câu khỏi stage 2): `predict_flow.py --canon` mặc định 1: stage 1.5 =
  `canon_llm.py` → `canon_llm` ghi `<out>/restated/` (câu viết lại làm câu hỏi) → `rules_first.py` trên đó; đáp án của nó
  là cuối cùng (sau đáp án stage 1); chỉ câu luật vẫn không đọc được sang stage 2, với câu gốc như trước (đúng như đã đo).
  Lỗi stage 1.5 → stage 2 nhận mọi câu chờ như cũ. Test CPU mới `tests/test_canon_flow.py` (mô hình giả, câu viết lại thật
  của v33). Câu train / public gốc không có câu chờ → không đổi (3000 / 3000, 1000 / 1000).

## Entry 27 — 2026-11-07 04:55: v34 = luồng nộp mới (stage 1.5 bật) trên GPU thật — paraphrase public

- yuan1103, 4×L4, `stress_flow.py --modes para:` (2065 paraphrase public), code c97e2e8. Chạy trọn không lỗi: stage 1 1472
  (134 s) → stage 1.5 viết lại 506 / 593, luật trả lời 171 → stage 2 nhận 422 (v32: 593); 3 lần chạy; tổng **11 157 s**
  (v32: 11 932 s). Nạp mô hình hai lần (stage 1.5 rồi stage 2) không vấn đề bộ nhớ.
- Toàn luồng (ngưỡng 0.75), chấm theo đáp án câu gốc: held-out k=1,2 **0.584 → 0.602** (1039 / 212 / 122; ước lượng entry 26:
  0.600); dev k=0 0.681 → 0.676 (557 / 89 / 46: +6 đúng, +9 sai — trong dao động lấy mẫu của mô hình); toàn bộ 2065 câu
  0.616 → **0.627**. Giữ `--canon 1`.
- Phân tích lỗi dev k=0 (public) của luật trên câu viết lại: 12 / 13 "sai" là paraphrase đổi nghĩa (đáp án gốc là số đơn
  huỷ, câu viết lại hỏi %; "ngày mai" thay ngày cụ thể ...). Một lỗi code: mẫu khớp mà thiếu kỳ trả `(tên, None)` → lỗi khi
  chạy (trong luồng nộp chỉ chuyển câu sang mô hình) → `qparse.parse` nay trả None. Thử "tháng trước / tháng vừa qua" đứng
  một mình = tháng liền trước: 0 đúng / 3 sai trên public (cả ba là paraphrase đổi kỳ) → không có bằng chứng, bỏ.
  Train / public gốc vẫn 3000 / 3000, 1000 / 1000.

## Entry 28 — 2026-11-08: quyết định hybrid framework; tổng quan tài liệu; thống kê dataset (không chạy GPU)

- **User chốt:** hybrid = Agent sinh code cho mọi câu + luật tổng quát hỗ trợ (ý định, schema, toolkit, làm sạch câu,
  few-shot, kiểm tra); cấm luật sinh thẳng code / đáp án. Thiết kế + kế hoạch E0–E5: `docs/06_HYBRID_FRAMEWORK.md`.
- **Điểm hiện có của đường mô hình:** v28 public 0.921 (vẫn có luật bỏ phiếu `--qw 4`, `--confirm 1`, câu làm sạch bằng
  `qnorm`); bản mô hình sinh code mọi câu (`--qw 0`, không `--confirm`) **chưa từng đo** → E0.
- **Tổng quan tài liệu** (DS-STAR, ReFoRCE, CHASE-SQL, XiYan-SQL, CHESS, OmniSQL, Arctic-Text2SQL-R1, chưng cất CoT,
  TrustSQL, selective prediction, ViText2SQL): bài học — ngữ cảnh dữ liệu giàu (mô tả do agent viết), nhiều ứng viên +
  chọn, tự sửa theo thực thi, dữ liệu tổng hợp có kiểm thực thi + fine-tune / RL, tín hiệu tin cậy kết hợp cho abstain.
- **Dataset:** quy tắc user (public là dùng được; chỉ dataset uy tín; ưu tiên loại không cần xin phép; không bao giờ dùng
  test của cuộc thi) + bảng 35 dataset (citation Semantic Scholar 08/11, hạng CORE, truy cập): `docs/07_DATASETS.md`.
- **Hiện trạng so khớp / schema** (trả lời user): so khớp thực thể nhiều tầng (chính xác sau chuẩn hoá → quy tắc tổng
  quát → fuzzy chỉ để gợi ý tên gần nhất, không tự chọn); chấm đáp án không fuzzy. Qwen đọc schema trước khi sinh code:
  `CANON_DOC` + card schema chuẩn C (kiểu, ví dụ giá trị) + README + rulebook + bảng thô ngoài C; chưa có data card bằng
  lời do Agent viết. Chi tiết `docs/06_HYBRID_FRAMEWORK.md` §2b.

## Entry 29 — 2026-11-08: kiểm tra mọi giả định của kế hoạch (không GPU) — chi tiết `docs/08_MASTER_PLAN.md` §1

- Mô hình (HF API + Kaggle API, đối chiếu tên + kích thước file): Qwen3.5-9B 9.65B (24 lớp linear attention + 8 full);
  Gemma 4 12B-it 11.96B (vLLM 0.32 chạy, không LoRA serving); thầy Qwen3.6-27B-FP8 (`nsalzman/...`, 66/66 file),
  Qwen3.6-35B-A3B-FP8 (`zerabytex/...`, 42/42), Gemma 4 31B-it (Google chính thức trên Kaggle), gpt-oss-120b / 20b
  (`danielhanchen/...`); tất cả Apache-2.0, public trước 01/10/2026. Loại Qwen3.5-122B-GPTQ (thiếu VRAM KV trên 4×L4).
- Thư viện: vLLM 0.32.0 (18/10/2026) cần torch 2.13.0, transformers [5.16.1, 5.20); bộ huấn luyện tương thích:
  transformers 5.17.0, peft 0.21.2, trl 1.15.0, accelerate 1.15.0, datasets 4.x, (unsloth 2026.11.3). Qwen3.5: không nên
  QLoRA 4-bit; LoRA bf16 9B ~22 GB (sát L4 22.5 GB) → FSDP 2 GPU / nhánh mặc định. vLLM 0.32: LoRA có cho Qwen3.5 /
  Gemma 4 / GPT-OSS; FP8, GPTQ, AWQ, Marlin (gồm MXFP4) chạy trên Ada.
- Kernel L4 không có internet → wheels huấn luyện vào dataset; thầy từ Kaggle Models.
- Số đo dùng để lập lịch: v34 stage 2 ~8.5 s / câu khó / lần chạy trên 4×L4, ~4.5k token / câu, 171 tok/s / GPU.
- Ba điều chỉ đo được trên GPU (bộ nhớ / tốc độ huấn luyện, nạp thầy FP8 trên L4, tốc độ LoRA vs gộp) → WP1 với quy tắc
  quyết định QĐ1–QĐ3 viết sẵn.

## Entry 30 — 2026-11-08: phương án đào sâu (tool, ràng buộc ngữ pháp, mô hình phụ, chưng cất) — `docs/09_OPTIONS_DEEP_DIVE.md`

- Căn cứ đo từ log v34 (3 × 422 câu khó, Qwen3.5-9B gốc): 35 % mẫu chương trình không chạy — TypeError 1,019, AttributeError
  670, thiếu khối code 626, LookupError 394, ImportError 230, NameError 216, SyntaxError 183, ModuleNotFound 126, không gán
  ANSWER 104, KeyError 269; lần chạy 1: đúng 188, sai do bất đồng 118, sai dù đồng ý 63, abstain 53.
- Đề xuất P0: ngữ pháp DSL qua `StructuredOutputsParams(grammar=...)` (vLLM 0.32, xgrammar), tool dạng hàm `find` / `period`
  / `check` / `help`, postcheck bất biến, tăng cường nhiễu khi huấn luyện. P1: truy hồi bằng Qwen3-Embedding-0.6B, bộ kiểm
  Qwen3.5-0.8B (Best-of-N + tin cậy), gộp nhánh LoRA (TIES / DARE) và nhiều LoRA cùng phục vụ, quỹ đạo dùng tool, tự phê bình
  có tool, nhánh thoát DuckDB SQL. Bộ đánh giá mới: `pub_noind` (ẩn hàm ngành, 112 câu public), `rb_shift`, `pub_noacc`.
- Đã kiểm: Qwen3.6 và Qwen3.5 dùng chung tokenizer (vocab 248,044, cùng merges, cùng 26 token đặc biệt) → chưng cất on-policy
  mức token khả thi (P2). Tổng tham số lúc chạy với S1 + S3: 11.12B.

## Entry 31 — 2026-11-08: kế hoạch + đặc tả triển khai được duyệt (`docs/08_MASTER_PLAN.md`)

- User đồng ý toàn bộ (docs/08 + docs/09 P0 / P1). `docs/08` viết lại thành đặc tả triển khai: luồng §4, file / hàm / định dạng /
  prompt / ngữ pháp / tool / thuật toán chọn / cờ / huấn luyện / mô hình phụ / Docker §5, dữ liệu D1–D7 + D2N, D4T, DV §7, gói việc
  WP0–WP12 với test và nghiệm thu §8, lịch / quota §9, quy tắc quyết định QĐ1–QĐ15 §10, thứ tự cắt khi thiếu thời gian.
- **Việc của khung chat kế tiếp: triển khai code đúng theo `docs/08`, WP0 → WP12.**

## Entry 32 — 2026-11-08: ước lượng giờ GPU và số acc Kaggle (`docs/08_MASTER_PLAN.md` §9, §5.12)

- Số đo cơ sở: cài wheels 180–315 s, chép 9B 50–89 s, nạp vLLM 274–280 s (v33 / v34); đầu ra ngắn ≥ 5.5–7 prompt / s / L4 (v33);
  luồng cũ có suy nghĩ 8.5 s / câu trên 4×L4 (v34); chạy chương trình 0.08–0.24 s; KV ~58 chuỗi 2k token / L4.
- Bảng việc G1–G29 (bố trí 4 GPU / batch theo §5.12): W46 ≈ 36.3 giờ phiên 4×L4 (≈ 145 GPU-h; + 30 % dự phòng ≈ 189), W47 ≈ 5.7.
  Quota W46 còn 75.6 GPU-h ở 3 acc → **cần thêm 4 acc L4 mới (lý tưởng 5)** trước sáng 11/11; T4 hiện có đủ. Phân bổ theo đợt
  Đ1–Đ7, lấp đầy acc theo thứ tự; tổng GPU tường ≈ 13 h trên 09–14/11. Thay bước dò thầy bằng sinh D3 đầy đủ cho cả 4 thầy song
  song; thêm vòng 3 nhánh và F1 / F2 song song; QĐ6 chọn thầy theo nhánh vòng 2; QĐ16 chỉnh ngân sách theo tốc độ đo ở WP1.

## Entry 33 — 2026-11-08: WP0 xong; WP2 module P0 (CPU); WP1 smoke vòng 1 (yuan1103, 4×L4)

- WP0: 9 token (8 acc có 4×L4, probe `zc-probe`); `score_1.jsonl` → `data/raw/public_gt.jsonl` (v30 1000 / 1000, v28 921 đúng,
  PA 0.852); `tests/splits/` (pub_full 1000, pub_k0 692, pub_ho 1373, tr_ho 1826, quick400 400, pub_noind 112, pub_noacc 1000);
  `eval_all.sh quick` train 3000 / 3000, public 1000 / 1000. Wheels huấn luyện = output kernel public `yeshenhi/zc-train-wheels`
  (ảnh GPU Kaggle: python 3.13.16; sau wheels vLLM 0.32: torch 2.13.0, transformers 5.19.0, peft 0.21.2, accelerate 1.15.0,
  datasets 4.8.5, xgrammar 0.2.8, duckdb 1.3.2; tải thêm transformers 5.17.0, trl 1.15.0, flash-linear-attention 0.6.0 + fla-core;
  `wheels_unsloth/`: unsloth 2026.11.3 + unsloth_zoo, bitsandbytes 0.50.2, xformers 0.0.35, torchao 0.18.0, cut-cross-entropy).
- WP2 (CPU, đều có test): ngữ pháp nhận 2681 / 2681 chương trình đã kiểm theo DN (dựng < 0.01 s, biên dịch xgrammar CPU
  0.2–1.9 s / DN), từ chối hàm / tham số / thực thể bịa; gợi ý: kỳ 171 / 171, thực thể 50 / 50 (200 câu train có nhãn), không lộ
  tên hàm; precheck: 0 báo nhầm trên 2681 chương trình, 14 cảnh báo kiểu (0.5 %), bắt đủ 9 loại sai; prompt compact: system 1043
  token (full 1649), card ≤ ~700 token; D1 compact 2119 bản ghi (trung vị 2174 token, tối đa 2662; bỏ 562 câu `tr_ho`).
- WP1 smoke vòng 1 (`zc-smoke35-1`, 22 phút): **Unsloth 1 GPU chạy được**: micro-batch 8 hết bộ nhớ, **mb 4 vừa: đỉnh 18.84 GB
  (≤ 21.5 GB → QĐ1: 4 nhánh / phiên 4×L4), 844 tok/s / L4** (chuỗi ~2.2k token, 8 bước). Đường HF + PEFT (1 GPU, FSDP2, FSDP4)
  **hỏng**: ảnh Kaggle có torchao 0.10.0, PEFT 0.21 ném ImportError khi dựng lớp LoRA (không phải lỗi bộ nhớ) → sửa: gỡ torchao
  trong job huấn luyện + vá kiểm tra của PEFT trong `training/sft.py`. Loss của Unsloth (0.028 → 0.0005) nghi chuẩn hoá khác
  TRL → thêm loss tham chiếu (trung bình trên token đáp án, đo thủ công trước / sau) ở cả hai trainer (CPU: = loss TRL).
  Vòng 2 (`zc-smoke35b`) đo HF 1 GPU / FSDP2 / FSDP4, gộp + phục vụ LoRA; phần suy luận (`zc-smoke35i`: ngữ pháp, 2 thầy) đang chạy.

## Entry 34 — 2026-11-08: WP1 smoke phần suy luận (`yuan1103/zc-smoke35i-1`): ngữ pháp, 2 thầy → QĐ2, QĐ12

- Qwen3.5-9B gốc (FP8, 1 L4), 50 bản ghi D1 (prompt compact, ví dụ truy hồi từ DN khác, gợi ý): **50 / 50 chương trình hợp lệ
  và trùng nhãn** cả khi có / không ngữ pháp (câu train có khuôn lặp giữa các DN → trong mẫu; held-out mới là thước đo). Không
  ngữ pháp: 15.6 s cho 50 câu (3.2 câu / s / L4, 169 tok/s), nạp 266 s, KV 239,712 token. **Có ngữ pháp xgrammar theo DN
  (liệt kê thực thể): 507 s (chậm 32×), không tăng độ hợp lệ → QĐ12: tắt ngữ pháp** (mặc định `--grammar 0`; A/B G5 thử
  ngữ pháp nhẹ `--max-enum 0`, không liệt kê thực thể).
- Thầy trên 4×L4 (QĐ2: chạy được, không cần dự phòng): Qwen3.6-27B-FP8 TP 2 nạp 308 s, KV 160,720 token; Gemma 4 31B-it
  `--quantization fp8` TP 2 nạp 415 s, KV 44,992 token. Câu do thầy viết (50 chương trình D1): tiếng Việt tự nhiên, có văn chat
  / viết tắt; lỗi gặp: kỳ tương đối sai ("tháng 4 năm ngoái" cho tháng 4 của năm hiện tại), lộ tên cột ("đơn valid") → lọc
  vòng lặp ngược + lọc từ mã.

## Entry 35 — 2026-11-08: WP4 nhà máy dữ liệu (CPU)

- D1 2119 (trung vị 2174 token), **D2 3816** (paraphrase câu train có chương trình; loại 205 câu đổi kỳ theo `parse_period`),
  **D2N 5418** (nhiễu: chat 3972, không dấu 1374, từ địa phương 72), **D4T 2000** quỹ đạo sửa lỗi (tên hàm 666, by= 639, thực thể
  348, tham số 347; thông điệp sửa = đúng chữ của luồng hybrid). `test_data_factory` (nhãn chạy lại ra đúng đáp án, không id
  public / DN dev_* / DN tr_ho, lệch kiểu ≤ 0.5 điểm, trùng < 1 %) và `test_sft_format` (thứ tự khối §5.3, nhãn khớp ngữ pháp DN): PASS.
- D3 bước 1 (`d3_sample_programs.py`): đột biến chương trình đã kiểm sang DN / ngày / kỳ / thực thể / k khác, chạy bằng toolkit;
  thử 108 → giữ 42 (2 DN, 22 họ hàm); bản 15k đang chạy. D3 bước 2: prompt thầy (`d3_teacher_prompts.py`, kèm kỳ đã quy ra
  ngày); job GPU `training/d3_job.py` (thầy viết câu → học trò đọc lại bằng luồng hybrid, 1 mẫu tham lam → giữ câu cùng đáp án).
- WP8 bắt đầu: `src/confidence.py` (đặc trưng từ log, logistic numpy, ngưỡng θ tối đa PA chính xác, kiểm conformal), test PASS.

## Entry 36 — 2026-11-08: WP1 smoke vòng 2 (`yuan1103/zc-smoke35b-1`) → QĐ1, QĐ3, QĐ16; WP1 xong

| Cấu hình (chuỗi ~2.2k token, LoRA r 32 trên mô hình ngôn ngữ: 200 module, 10 loại, 80.2M tham số học) | micro-batch lớn nhất | VRAM đỉnh / GPU | tok/s |
|---|---|---|---|
| HF + PEFT, 1 GPU (gỡ torchao 0.10) | 1 (mb 2, 4 hết bộ nhớ) | 21.07 GB | 587 |
| Unsloth, 1 GPU | 4 | 18.84 GB | 801–844 |
| FSDP2 (2 GPU) | 4 (mb 8 hết bộ nhớ) | 18.12 GB | 904–945 |
| FSDP4 (4 GPU) | 8 | 18.04 GB | 2,023 |

- **QĐ1:** Unsloth 1 GPU ≤ 21.5 GB → **4 nhánh / phiên** (3.2k tok/s mỗi phiên, hơn 2 × FSDP2 ≈ 1.9k). Bản cuối (WP7): FSDP4 mb 8.
- **QĐ16:** nhánh < 1,200 tok/s → **ngân sách nhánh 4M token** (~80 phút / nhánh).
- **QĐ3:** vLLM FP8, 50 câu: gộp 16.3 s (3.06 câu / s, nạp 279 s, KV 239,712), `--lora` 19.2 s (2.6 câu / s, nạp 346 s, KV 213,312)
  → LoRA chậm 18 % > 10 % → **gộp trọng số + FP8** (`training/merge.py`: 200 tensor trong 23 s). Cả ba (gốc / gộp / LoRA) 50 / 50 trùng
  nhãn trên câu train (dễ); loss tham chiếu (token đáp án) của mô hình gốc trên 4 bản ghi 0.150 → 0.093 sau 8 bước Unsloth,
  → 0.0085 sau 30 bước FSDP2 (loss TRL in ra 0.029 là chuẩn hoá khác, không dùng).
- Khởi chạy (17:20–17:34 UTC, code chốt `bc7c1a3` cho WP3 qua preflight; job huấn luyện / D3 không qua preflight theo §11, đã
  chạy thử CPU): WP3 `e0` (quick400 + pub_full), `e0think`, `s0` (Gemma 4 12B), `abfull`, `abgram`, `abnoind`, `abnoacc`; D3
  `d3self` (9B tự viết), `d3qwen27`, `d3gemma31`, `d3qwen35a3b`, `d3gptoss`; WP6 vòng 1 `armsr1a` (A1 = D1, A2 = D1 + D2 + D2N,
  A2T = A2 + D4T, A2T64 = A2T với r 64; 4M token / nhánh).

## Entry 37 — 2026-11-08: WP3 kết quả (E0 / S0 / A/B), QĐ5, QĐ11, QĐ12; D3 hai thầy đầu; WP6 vòng 1 huấn luyện xong

Luồng hybrid, mô hình gốc, cùng code `bc7c1a3` (qua preflight), 4×L4 DP 4. PA theo tập (`tests/score_sets.py`; quick400 là
mốc chính, các tập khác là phần giao với các câu đã chạy):

| Run | quick400 | pub_full | pub_k0 | tr_ho | pub_noind | ghi chú |
|---|---|---|---|---|---|---|
| E0 Qwen3.5-9B, prompt compact | 0.720 | 0.864 (1000 câu: 898 / 34 / 68) | 0.493 | 0.940 | 0.679 | v28 public gốc 0.852 |
| E0 think | 0.765 | 0.893 (150) | 0.540 | 0.910 | — | chậm ~1.9× |
| A/B prompt full | **0.755** | 0.887 (150) | 0.520 | 0.910 | — | |
| S0 Gemma 4 12B-it, compact | **0.7675** | 0.873 (150) | 0.553 | 0.930 | — | KV 75k token |
| A/B `pub_noacc` (câu bỏ dấu) | — | 0.824 (1000) | | | | |
| A/B `pub_noind` (ẩn hàm ngành) | — | | | | **−0.0625** (26 / 33 / 53) | gần như không dùng nhánh `<python>` |
| A/B ngữ pháp nhẹ | huỷ sau 2.6 h | | | | | quá chậm (QĐ12) |

- **QĐ5:** prompt compact thua full 3.5 điểm > 1.0 → với mô hình gốc dùng **full**. Mô hình fine-tune học đúng prompt nó
  được huấn luyện (dữ liệu SFT đang ở compact); quyết định lại sau WP6 vòng 1.
- **QĐ4 (chờ):** Gemma compact hơn Qwen compact 4.75 điểm nhưng chỉ hơn Qwen full 1.25 (< 2.0). Gemma 12B bf16 (24 GB) không
  vừa 1 L4 khi LoRA → FSDP 2 GPU (2 nhánh / phiên) và chậm hơn ~1.3× mỗi token; giữ Qwen nếu bản fine-tune vòng 1 vượt Gemma gốc rõ.
- **QĐ11:** truy hồi embedding (Qwen3-Embedding-0.6B) kém n-gram (public hit@4 0.729 vs 0.846; k=0 0.702 vs 0.818) → giữ n-gram.
- **QĐ12:** ngữ pháp tắt (WP1: chậm 32×; bản nhẹ chạy > 2.6 h cho quick400 → huỷ).
- Phân tích lỗi E0 (dev = pub_full + pub_k0, 1150 câu; `tests/analyze_run.py`): đúng 86.8 %, top sai do mẫu bất đồng 4.5 %,
  đồng thuận sai 4.3 %, đúng nhưng bỏ (conf < 0.5) 4.1 %. Nhóm lỗi lặp: "bán được bao nhiêu gói / ly / đĩa" quên
  `base_unit=` (13 / 16 sai: gợi ý không liên kết từ đơn vị), câu chủ quan / không ghi nhận vẫn trả lời dù gợi ý
  `missing_data` đúng (mô hình gốc bỏ qua gợi ý), `debt_aging(..., bins=[7, 15])` lỗi kiểu (toolkit chỉ nhận cặp).
- **D3:** 15k chương trình × 2 câu / thầy, lọc vòng lặp ngược bằng 9B. Tự viết (9B): giữ 17,882 / 34,505 (51.8 %);
  Qwen3.6-35B-A3B: giữ 19,607 / 30,445 (64.4 %) — tỉ lệ giữ cao hơn ở mọi kiểu đáp án. Loại thêm chương trình suy biến (cùng
  phần tử hai lần trong một list, vd. `increasing([month(2026, 6), month(2026, 6), ...])`: câu không thể nói đúng; boolean
  vòng lặp ngược khớp ngẫu nhiên 50 %): còn 17,670 / 19,345 bản ghi. Qwen3.6-27B, Gemma 4 31B, gpt-oss-120b đang chạy.
- **WP6 vòng 1 huấn luyện** (`synchan5704/zc-armsr1a-1`, Unsloth 1 GPU / nhánh, 4M token ≈ 1.87k bản ghi ≈ 30 bước batch 64,
  ~80 phút, 876–915 tok/s, VRAM 18.9–19.8 GB): A1 = D1, A2 = D1 + D2 + D2N, A2T = A2 + D4T, A2T64 = A2T r 64 / α 128;
  loss tham chiếu (token đáp án) 0.22 → 0.0003–0.0014. Đánh giá trong kernel hỏng (4 bản gộp 19 GB làm /tmp chỉ đọc) → đánh giá
  lại bằng vLLM `--lora` (`nguyenhoa99/zc-evalr1-1`: thư mục dataset tìm sai một cấp → sửa `adapters_root`, thử trên cây
  /kaggle/input giả; chạy lại `zc-evalr1b-1`) + mốc gốc cùng cấu hình (`quynhtran35/zc-evalbase-1`, nhánh BASE, `--eval-dp 4`).

## Entry 38 — 2026-11-08: WP6 vòng 1 đánh giá → QĐ7 (A1 = D1 thắng); WP5 xong; vòng 2 khởi chạy

Đánh giá LoRA qua vLLM (`nguyenhoa99/zc-evalr1b-1`, 1 GPU / nhánh, code `dbec142`, `configs/e0.json` compact) và mốc gốc cùng
code (`quynhtran35/zc-evalbase-1`, DP 4). quick400 chứa 150 câu pub_full, 150 pub_k0, 100 tr_ho:

| Nhánh (4M token, r 32 trừ ghi chú) | quick400 | pub_full (150) | pub_k0 (150) | **pub_ho (1373)** | tr_ho (100) | sai / bỏ (quick400) |
|---|---|---|---|---|---|---|
| BASE (Qwen3.5-9B gốc) | 0.700 | 0.767 | 0.473 | 0.516 | 0.94 | 40 / 40 |
| A1 = D1 | **0.7775** | 0.840 | **0.600** | 0.531 | 0.95 | 29 / 31 |
| A2 = D1 + D2 + D2N | 0.7275 | 0.740 | 0.560 | 0.532 | 0.96 | 46 / 17 |
| A2T = A2 + D4T | 0.7075 | 0.753 | 0.493 | 0.516 | 0.96 | 52 / 13 |
| A2T64 = A2T, r 64 / α 128 | 0.750 | 0.793 | 0.567 | **0.540** | 0.96 | 40 / 20 |

- **QĐ7:** A1 hơn gốc 7.75 điểm quick400, không thua tr_ho / k=0 → **hỗn hợp thắng = D1**. Với ngân sách cố định (≈ 1.87k bản ghi),
  thêm D2 / D2N / D4T làm loãng D1 (A2 chỉ còn ~350 bản ghi D1) và tăng câu trả lời sai (mô hình bớt bỏ, sai nhiều hơn). r 64
  tốt hơn r 32 trên cùng dữ liệu (+4.25 quick400, +2.5 pub_ho) → thử ở vòng 2 / 3.
- Held-out pub_ho (paraphrase k = 1, 2) chỉ tăng 1.5–2.4 điểm: fine-tune vòng 1 chủ yếu sửa câu cùng cách hỏi với train; bền
  với cách hỏi mới là điểm yếu chính (mốc v34 held-out 0.602).
- Đánh giá 1 GPU / nhánh (LoRA): 1,773 câu trong 1,850–2,250 s; mốc gốc DP 4: 1,225 s.
- **WP5 xong** — bản ghi D3 sau lọc: qwen27 21,019 (giữ 73.6 %), qwen35a3b 18,146 (64.4 %), gemma31 17,043 (69.5 %; câu theo khuôn,
  nhiều bản trùng bị chặn ở 3), self 16,688 (51.8 %), gptoss 10,079 (31.9 %; ~1.5k câu rác "unanswerable" / "questions" / code được
  vòng lặp ngược giữ — đã lọc).
- **Vòng 2 (khởi chạy 21:10 UTC, worktree `dbec142` = code đánh giá vòng 1):** D1 (2,119) + 2,119 bản ghi D3 của một thầy (50 %
  theo bản ghi, `training/data/mix.py`), 4M token, quick400 + pub_ho. `synchan5704/zc-armsr2a-1`: B3S (self), B3Q35, B3OSS, B3G31;
  `quynhle07/zc-armsr2b-1`: B3Q27, B3MIX (5 thầy, ~424 mỗi thầy), A1R64 (D1, r 64), B3MIX64 (r 64). So với A1 ở cùng ngân sách: D3
  phải có giá trị ít nhất ngang D1 trên mỗi bản ghi.
- **QĐ4 làm lại cho đúng:** Gemma 4 12B mới chỉ chạy prompt compact. Chạy Gemma prompt full (quick400 + pub_ho), Gemma compact
  pub_ho và Qwen full pub_ho trên đúng code WP3 (`bc7c1a3`, worktree `wt_v35`) sau preflight → so gốc với gốc cùng prompt tốt nhất.

## Entry 39 — 2026-11-08: WP9 vòng 2 (log A1, dev) + đọc lại (M11 rút gọn); kiểm gợi ý trên paraphrase

- Lỗi A1 trên dev (pub_full + pub_k0 trong quick400, 300 câu): đúng 81.3 %, **đồng thuận sai 10 %** (nhóm lớn nhất), bất đồng 6.3 %.
  Một phần pub_k0 là paraphrase đổi nghĩa (vd. "tháng này" → "hôm nay" nhưng nhãn giữ của câu gốc): trần pub_k0 < 1, không sửa.
  Lỗi tổng quát lặp lại: từ đơn vị (đã sửa gợi ý), câu chủ quan / không ghi nhận vẫn trả lời dù gợi ý `missing_data` đúng.
- Độ chính xác gợi ý `missing_data`: dev 32 / 32 đúng (0 / 1,115 câu trả lời được bị gắn nhầm), train 20 / 20 (0 / 2,930); độ phủ
  train 20 / 70 — phần còn lại toolkit tự `raise Unanswerable` (kỳ ngoài dữ liệu, không có nhập hàng / công nợ / giá vốn /
  phương thức thanh toán / kênh), không cần gợi ý thêm.
- **M11 rút gọn (`--review 1`, mặc định tắt đến khi đo A/B):** mẫu có kết quả đáng ngờ (trả lời dù gợi ý nói dữ liệu không ghi
  nhận, hoặc cờ postcheck) được đọc lại đúng một lượt; mô hình tự quyết (viết lại chương trình hoặc `raise Unanswerable`); đáp án
  cũ giữ nếu lượt đọc lại lỗi. Không có luật trả lời.
- Tổng hợp pub_ho theo khuôn (chỉ số tổng, không xem câu): yếu nhất `unit_qty` (PA −0.60), `subjective` (−0.61), `top_item`
  (A1 0.11), `share` (0.31), `n_debtors`; `revenue` (294 câu) chỉ 0.67 → độ bền với cách hỏi mới là chỗ mất điểm lớn nhất.
- Gợi ý kỳ trên D2 (paraphrase câu train, nhãn đã kiểm): 3,412 khớp / 25 lệch (phần lớn tương đương: `between` cả tháng = `month`);
  lỗi thật: năm đứng trước quý / tháng không có chữ "năm" ("Năm 2026, quý 2", "2025 tháng 8") → `quarter_noyear` / `month_noyear`
  → sửa `qparse.parse_period`.

## Entry 40 — 2026-11-08: WP7 chạy thử đầu-cuối (`nguyenhoa99/zc-fullsmoke-1`)

- `training/full_job.py` (FSDP4 trên 4×L4, LoRA r 64): 142 bản ghi D1 (300k token, 3 bước) trong 333 s, **1,889 tok/s**, VRAM đỉnh
  17.7 GB, 200 module LoRA (không chạm vision / MTP); lưu adapter dưới FSDP (FULL_STATE_DICT) OK (642 MB); `merge.py merge` 200
  tensor trong 28 s; đánh giá mô hình gộp qua luồng hybrid (FP8, DP 4): 20 câu, 17 đúng / 1 sai / 2 bỏ; tổng 1,015 s.
- Ngân sách bản cuối (QĐ16: ≤ 4.5 h / phiên): ≈ 30M token / phiên ở tốc độ này (F1 1 epoch ~25M token ≈ 3.7 h).
- Lỗi preflight QĐ4 (job chỉ có `pub_ho`: smoke CPU không có văn bản paraphrase → 0 câu → ZeroDivisionError) → sửa
  `eval_flow` (`$ZC_PARA_DIR`, thoát rõ khi 0 câu), `select_subset`; job QĐ4 chạy `quick400,pub_ho` (lặp quick400 để đo nhiễu).

## Entry 41 — 2026-11-08: WP6 vòng 2 → QĐ6 (D3 không giúp ở ngân sách cố định); nhiễu đo lớn; QĐ4 nửa đầu; lỗi chia GPU

Vòng 2 (4M token ≈ 1.87k bản ghi, D1 + 50 % D3 một thầy, r 32 trừ ghi chú; code / cấu hình đánh giá như vòng 1):

| Nhánh | quick400 | pub_full (150) | pub_k0 (150) | **pub_ho (1373)** | tr_ho (100) |
|---|---|---|---|---|---|
| A1 (vòng 1, D1) | **0.7775** | 0.840 | **0.600** | 0.531 | 0.95 |
| B3S (self) | 0.755 | 0.820 | 0.567 | 0.528 | 0.94 |
| B3Q35 | 0.7225 | 0.780 | 0.553 | 0.4975 | 0.89 |
| B3OSS | 0.7525 | 0.800 | 0.580 | 0.515 | 0.94 |
| B3G31 | 0.7475 | 0.833 | 0.527 | 0.527 | 0.95 |
| B3Q27 | 0.720 | 0.760 | 0.547 | 0.530 | 0.92 |
| B3MIX (5 thầy) | 0.745 | 0.773 | 0.580 | 0.538 | 0.95 |
| A1R64 (D1, r 64) | 0.7175 | 0.787 | 0.520 | 0.530 | 0.91 |
| B3MIX64 | 0.700 | 0.713 | 0.533 | 0.493 | 0.93 |
| *Qwen gốc, prompt full (QĐ4 job)* | 0.740 | 0.833 | 0.533 | **0.541** | 0.91 |

- **QĐ6:** không nhánh D3 nào hơn A1 (D1 thuần) → thay một nửa D1 bằng câu tổng hợp làm giảm 2–5 điểm ở cùng ngân sách; trong các
  thầy, self / OSS / mix tốt nhất trên quick400, mix tốt nhất trên pub_ho. D3 không dùng ở tỉ lệ 50 %.
- **Nhiễu:** A1R64 (≈ A1, chỉ khác r) thấp hơn A1 6 điểm quick400; cùng mô hình gốc đo hai lần lệch ~1.5–2 điểm (0.755 / 0.740;
  0.720 / 0.700). Trên pub_ho (1373 câu, ổn định hơn) mọi nhánh fine-tune 0.49–0.54, **ngang mô hình gốc với prompt full 0.541**:
  fine-tune 4M token trên prompt compact chưa cho lợi ích held-out chắc chắn; điểm quick400 0.7775 của A1 một phần là may.
- Prompt full cho mô hình gốc: pub_ho 0.541 vs compact 0.516 (+2.5) → vòng 3 thử fine-tune trên prompt full.
- **Vòng 3 (`synchan5704/zc-armsr3-1`, code chính + dữ liệu v2):** C1 / C1S (D1 v2, hai seed: đo phương sai huấn luyện), F1 (D1 v2
  prompt full, max_len 4608, mb 2, đánh giá `configs/e0_full.json`), C1E2 (2 epoch). **QĐ9:** súp tuyến tính top-4 / top-8 nhánh
  theo quick400 (`training/soup_job.py`, `nguyenhoa99/zc-soupr2b-1`).
- **Lỗi tìm thấy (sửa `fd2b876`):** kernel nhiều job `--dp 4` chạy song song, mỗi job một GPU, và shard DP đặt
  `CUDA_VISIBLE_DEVICES=0` tuyệt đối → job 2 cũng chạy trên GPU 0; hai súp dùng chung `/tmp/soup` (job sau xoá mô hình gộp của
  job trước). Job đầu của `qd4gemma` vẫn hợp lệ (chạy 1 GPU).
- Thêm biến thể `think` cho vòng 2 (`--variants direct,think`): mẫu vòng 2 (câu khó, ~15 %) suy nghĩ trước; chưa đo.
- **QĐ4 (xong):** Gemma 4 12B prompt full (`quynhle07/zc-qd4gemma-1`, job 1 chạy trên 1 GPU do lỗi chia GPU, kết quả vẫn hợp lệ):
  quick400 0.7675, pub_k0 0.547, **pub_ho 0.521** (999 / 284 / 90) vs Qwen full quick400 0.740 / 0.755, **pub_ho 0.541** (971 / 228 /
  174). Hơn quick400 +2.0 so với trung bình hai lần đo Qwen = đúng ngưỡng nhưng trong nhiễu đo (±2); held-out lớn hơn 3.4× nghiêng
  về Qwen 2 điểm (Gemma trả lời nhiều hơn, sai nhiều hơn); Gemma bf16 24 GB không vừa 1 L4 để LoRA → **giữ Qwen3.5-9B**. Job 2
  (Gemma compact pub_ho) không ra kết quả (lỗi chia GPU) — không cần cho quyết định.

## Entry 42 — 2026-11-09: QĐ12 (DuckDB) — không giữ; súp / vòng 3 / A/B suy luận đang chạy

- `yuan1103/zc-e1sql-1` (mô hình gốc, prompt full, code v2, `pub_noind` ẩn hàm ngành, `--escape python,sql`): PA 0.036 (28 / 24 / 60)
  vs −0.0625 trước (prompt compact, chỉ `<python>`). Mô hình **không viết khối `<sql>` nào** (0 / 674 mẫu) và chỉ 2 khối `<python>`
  → phần tăng đến từ prompt full + gợi ý v2, không từ SQL. **QĐ12: DuckDB không giữ** (mô hình không dùng nhánh thoát; code giữ, tắt
  mặc định). Điểm yếu thật: khi thiếu hàm ngành, mô hình ép hàm toolkit chung thay vì thoát sang pandas.
- **A/B suy luận trên mô hình gốc (code v2 `4f1df6c`, prompt full, quick400 + pub_ho, DP 4):**

| Cấu hình | quick400 | pub_full (150) | pub_k0 (150) | pub_ho (1373) | tr_ho | thời gian kernel |
|---|---|---|---|---|---|---|
| code cũ `bc7c1a3`, full (QĐ4 job) | 0.740 | 0.833 | 0.533 | **0.541** | 0.91 | 2,056 s |
| **E1 code v2, full** (`quynhtran35/zc-e1full-1`) | **0.775** | **0.893** | **0.567** | 0.535 | 0.91 | 1,963 s |
| E1 + `--review 1` (`quynhle07/zc-e1review-1`) | 0.750 | 0.847 | 0.540 | 0.523 | 0.92 | 2,133 s |

  Code v2 (gợi ý đơn vị, kỳ năm-trước, `debt_aging`) +2.75 quick400 / +6 pub_full / +3.4 pub_k0, held-out ngang (−0.6, trong nhiễu).
  **M11 đọc lại: không giữ** — 23 % mẫu bị đọc lại (chủ yếu do cờ postcheck "zero result" / lệch kiểu), 310 đáp án đổi, dev câu không
  có gợi ý thiếu dữ liệu 214 → 203 điểm ròng; câu có gợi ý thiếu dữ liệu chỉ 6 (đã đúng 5 / 6 với prompt full).
| E1 + biến thể `think` ở vòng 2 (`nguyenhoa99/zc-e1think-1`, thay `rephrase`) | 0.7575 | 0.880 | 0.527 | **0.549** | 0.92 | 2,984 s |

  Think vòng 2: 20 % mẫu, trung vị 553 token; held-out +1.4 (cao nhất đến nay) nhưng quick400 −1.75, pub_k0 −4, chậm hơn 52 % →
  chưa quyết định (≈ 1 SE); giữ làm ứng viên cho so sánh cấu hình cuối (WP10).
- **QĐ9 súp (`nguyenhoa99/zc-soupr2b-1`, gộp tuyến tính đều vào trọng số gốc, code `fd2b876`, prompt compact):** top-4 theo quick400
  (A1, B3S, B3OSS, A2T64): quick400 0.7575, pub_full 0.807, pub_k0 **0.607**, **pub_ho 0.5637** (931 / **157** / 285), tr_ho 0.91;
  top-8: quick400 0.7425, pub_k0 0.587, pub_ho 0.554 (936 / 175 / 262). Súp top-4 = **held-out cao nhất đến nay** (+2.9 so với gốc
  prompt full code v2, +3.3 so với A1): sai ít hơn hẳn (157 vs ~230), bỏ nhiều hơn (mẫu bất đồng → vòng 2 483 câu). Theo đúng QĐ9
  (+0.5 quick400 so với nhánh đơn tốt nhất A1 0.7775) súp chưa đạt, nhưng 0.7775 của A1 là đo may (entry 41) → súp là **ứng viên
  cho so sánh held-out cuối (QĐ15)**; ủng hộ hướng WP7 "thêm seed để gộp". Gộp 8 adapter: 367–419 s; đánh giá DP 4 ~2,700–3,000 s
  (vòng 2 nhiều hơn).
- **QĐ14 hiệu chỉnh trên log ứng viên** (`tests/calibrate.py`, θ chọn trên dev 300 câu; held-out chỉ đọc tổng): súp top-4 pub_ho
  0.564 (vote ≥ 0.5) → **0.596 ở θ dev 0.386** (0.603 nếu nâng θ có giới hạn) ≈ mốc v34 held-out 0.602; gốc + think 0.548 → 0.545;
  gốc code v2 0.534 → 0.530; A1 giữ vote. Kiểm conformal (sai held-out ≤ dev + 2 điểm) **không đạt được ở mọi θ hợp lý**: paraphrase
  khó hơn ở mọi ngưỡng (dịch phân phối, không phải lệch hiệu chỉnh); nâng θ không giới hạn làm PA sập (gốc: θ 0.96 → PA 0.04) →
  `calibrate.py` chỉ nâng θ khi PA dev mất ≤ 1 điểm, còn lại báo "SHIFT" và giữ θ dev.

## Entry 43 — 2026-11-09: WP6 vòng 3 — mọi fine-tune đơn lẻ thua mô hình gốc trên held-out; WiSE-FT

`synchan5704/zc-armsr3-1` (dữ liệu v2, code chính, mọi bản ghi D1, r 32):

| Nhánh | quick400 | pub_full | pub_k0 | **pub_ho** | tr_ho | ghi chú |
|---|---|---|---|---|---|---|
| C1 (compact, seed 20261108) | 0.7175 | 0.780 | 0.513 | 0.510 | 0.93 | 34 bước, 4.5M token |
| C1S (compact, seed 7) | 0.715 | 0.800 | 0.487 | **0.485** | 0.93 | phương sai seed: −2.5 pub_ho |
| F1 (prompt full) | **0.760** | 0.847 | 0.540 | 0.503 | 0.96 | 7.4M token, đánh giá prompt full |
| C1E2 (2 epoch) | 0.715 | 0.753 | 0.527 | **0.479** | 0.94 | huấn luyện thêm → held-out kém hơn |
| *gốc prompt full, code v2 (E1)* | 0.775 | 0.893 | 0.567 | **0.535** | 0.91 | |

- Mọi fine-tune đơn lẻ **dưới mô hình gốc trên held-out** (0.48–0.51 vs 0.535): 2k câu train làm mô hình chuyên biệt hoá theo cách hỏi
  của train; thêm epoch làm xấu hơn; prompt full giúp quick400 (+4.25) nhưng không giúp held-out. Chỉ **súp** (trung bình các
  fine-tune khác nhau) vượt gốc (0.564; 0.596 với θ hiệu chỉnh).
- Đúng mẫu "bền dưới dịch phân phối" của **WiSE-FT** (Wortsman et al., CVPR 2022: nội suy trọng số fine-tune về trọng số gốc tăng độ
  bền ngoài phân phối mà không mất trong phân phối) → thử súp và A1 ở α = 0.5 (`phatle1578/zc-wisep1-1`, `-wisep2-1`), đánh giá thêm
  **tr_ho đầy đủ (1,826 câu)** làm tập held-out thứ hai cho súp α = 1 và mô hình gốc (tránh chọn quá khớp pub_ho).
- Thành viên súp WP7 (`w7c` compact, `w7f` full: D1 seed 11 / 12, D1 + D2, D1 + D2N) đang huấn luyện; ý tưởng trên súp top-4
  (think vòng 2, 12 mẫu, lặp seed để đo nhiễu) đang chạy (`nguyenhoa99/zc-ideasa-1`, `-ideasb-1`).
- **Nhiễu đánh giá (đo trực tiếp):** súp top-4 chạy lại với seed lấy mẫu dịch (`nguyenhoa99/zc-ideasb-1`, `seed_base 1`): quick400 0.750
  (lần 1: 0.7575), pub_k0 0.560 (0.607), **pub_ho 0.555** (0.564), hiệu chỉnh θ dev → 0.577 (0.596); 143 / 1373 câu pub_ho đổi kết quả
  giữa hai lần. → nhiễu pub_ho ≈ ±1 điểm, pub_k0 / quick400 ±2–5; súp vẫn hơn gốc (0.535) và mọi fine-tune đơn ở cả hai lần.
- **Ý tưởng trên súp top-4** (`nguyenhoa99/zc-ideasa-1`): + think vòng 2: quick400 0.770, pub_k0 0.580, **pub_ho 0.558** (θ dev 0.584),
  3,799 s; + 12 mẫu: quick400 0.725, pub_ho 0.548 (θ dev 0.573), 3,829 s (nhiều mẫu hơn → conf = phiếu / mẫu thấp hơn → bỏ nhiều hơn).
  Cả hai trong nhiễu so với súp thường (0.564 / 0.555; θ dev 0.596 / 0.577) mà chậm hơn 30 % → **không dùng**.
- **WiSE-FT xác nhận** (`phatle1578/zc-wisep2-1`): A1 × 0.5 (Δ LoRA giảm nửa, gộp vào gốc): quick400 0.775 (A1 0.7775), pub_k0 **0.620**,
  **pub_ho 0.559** (A1 0.531, +2.8; sai 154 vs 228), **tr_ho đầy đủ 0.851** (1,826 câu: 1622 / 68 / 136); mô hình gốc prompt full trên
  tr_ho đầy đủ 0.8467 (1652 / 106 / 68). → Nội suy về gốc giữ trong phân phối, tăng held-out, giảm mạnh câu sai tự tin.
- **Súp × WiSE** (`phatle1578/zc-wisep1-1`): súp top-4 α 0.5 (mỗi thành viên 0.125): quick400 0.7475, pub_ho 0.559, **tr_ho 0.870**
  (1656 / 68 / 102); súp α 1 trên tr_ho đầy đủ 0.859. Held-out gộp (pub_ho + tr_ho, 3,199 câu; QĐ15): gốc full 0.713, A1 × 0.5 0.726,
  súp α 1 0.730, **súp α 0.5 0.736 → 0.749 với θ hiệu chỉnh trên dev (θ 0.504; kiểm conformal đạt)**. Quét α 0.35 / 0.7 đang chạy.
- **Súp thành viên WP7 compact** (`mucnokieu/zc-soupwc4-1`; WC1 / WC2 = D1 v2 seed 11 / 12, WC3 = D1 + D2, WC4 = D1 + D2N; α 1):
  quick400 0.7575, pub_k0 0.580, **pub_ho 0.579** (cao nhất), tr_ho 0.867 → held-out gộp **0.743**, θ hiệu chỉnh dev **0.761** (cao nhất).
  Thành viên huấn luyện trên dữ liệu v2 (khớp gợi ý suy luận) + đủ D1 → súp mạnh hơn súp cũ kể cả chưa WiSE. Quét α 0.5 / 0.7 trên súp
  này: `huyenco/zc-wisewc05-1`, `-wisewc07-1`.
- Súp 8 (top-4 cũ dữ liệu v1 + 4 thành viên WP7 v2, `mucnokieu/zc-soup8m-1`): quick400 0.7575, pub_ho 0.567, tr_ho 0.869 → gộp 0.739
  (θ dev 0.750) < súp 4 thành viên v2 (0.743 / 0.761): thành viên v1 không thêm gì.
- Súp top-4 α 0.7 (`nguyenhoa99/zc-wisea07-1`): quick400 0.750, pub_ho 0.576, tr_ho 0.865 → gộp 0.741 (vote); θ hiệu chỉnh trên 300 câu dev
  dao động mạnh giữa các lần (0.37–0.71) → hiệu chỉnh cuối (WP10) phải dùng dev lớn: **pub_full + pub_k0 đầy đủ (~1,700 câu, không
  huấn luyện trên chúng)**; câu train không dùng làm dev cho mô hình fine-tune (trong mẫu). Đường α của súp cũ phẳng trong nhiễu
  (α 0.5 0.736, α 0.7 0.741, α 1 ~0.730).
- Súp top-4 α 0.35 (`nguyenhoa99/zc-wisea035-1`): **quick400 0.785** (cao nhất, chỉ 18 sai), pub_full 0.860, pub_k0 0.607, pub_ho 0.570,
  tr_ho 0.870 → gộp 0.741. Đường α súp cũ (held-out gộp): 0.35 0.741 / 0.5 0.736 / 0.7 0.741 / 1 ~0.730 — phẳng với α ≤ 0.7, α nhỏ tốt
  nhất trong phân phối (giữ nhiều trọng số gốc hơn không mất gì).
- **Súp WC1–4 × α 0.5** (`huyenco/zc-wisewc05-1`): quick400 **0.785**, pub_full (150) 0.887, pub_k0 0.580, **pub_ho 0.589**, **tr_ho 0.875**
  (1660 / 62 / 104) → held-out gộp **0.752** (α 1: 0.743; gốc: 0.713) = **tốt nhất mọi tập**; sai pub_ho 162.

## Entry 44 — 2026-11-09: WP9 vòng 3 — cờ postcheck sai làm bỏ câu đúng

- Phân tích lỗi dev của súp WC α 0.5: 81 % đúng, lỗi rải rác (không nhóm nào > 5 câu); 17 câu "đúng nhưng bỏ". Ví dụ `debt_aging`: **cả 7
  mẫu cùng đáp án đúng nhưng bỏ** vì mỗi mẫu bị cờ "type list but the question asks number" (gợi ý kiểu đáp án đọc "từng nhóm bao
  nhiêu tiền" là number) → trọng số 0.5 → conf 0.375 < 0.5; câu tăng trưởng bị cờ "a count must be a non-negative integer" trên đáp
  án percent.
- Độ chính xác từng loại cờ trên mẫu đã chạy (dev; held-out chỉ xem tổng): mẫu không cờ sai 22 %; "count non-negative integer" dev
  sai **1 %** (206 / 2), "type list but asks number" dev **3 %** (112 / 4), "type number but asks string" 6 % → cờ **ngược thông tin**;
  "zero result" sai 100 % (dev 0 / 29), "empty list" 100 %, "list but asks string" 87 % → cờ tốt. **Sửa (quyết định theo dev):** bỏ
  cờ đếm; cặp kiểu (list cho number, number cho string) không còn là cờ; gợi ý kiểu đáp án: "mỗi / từng nhóm, loại, kênh ... bao
  nhiêu", "chia theo / thành" → list (train: khớp kiểu 2,620 / 2,641, trước 2,619).
- Phát lại phiếu trên log (cùng mẫu, cờ mới): súp WC α 0.5 pub_full 0.887 → **0.907**, pub_k0 0.580 → 0.593, pub_ho 0.589 → **0.597**,
  tr_ho 0.875 → **0.884**; súp WC α 1 và mô hình gốc cũng tăng ở mọi tập (+0.7 → +2 điểm).
- Súp WC α 0.7 (`huyenco/zc-wisewc07-1`): quick400 0.7675, pub_ho 0.587, tr_ho 0.865 → gộp 0.746 (cờ mới, phát lại: 0.756). Đường α súp WC
  (gộp, cờ mới): α 1 0.751 / α 0.7 0.756 / **α 0.5 0.761**; quick400 0.758 / 0.768 / **0.785** → thử α 0.35.
- Súp WF1–4 (thành viên prompt full, đánh giá prompt full; `quynhle07/zc-soupwf4-1`): quick400 0.7575, pub_ho 0.550, tr_ho 0.849 → gộp
  0.720 (cờ mới 0.725) < súp compact WC (0.743 / 0.751): sai nhiều hơn (229 vs 189 trên pub_ho) → **thành viên compact thắng**.
- Súp WF1–4 × α 0.5 (prompt full; `quynhle07/zc-wisewf05-1`): **quick400 0.790** (cao nhất), pub_full (150) 0.880, pub_k0 0.593, pub_ho 0.581,
  tr_ho 0.867 → gộp 0.744 (cờ mới 0.751) ≈ súp WC α 0.5 (0.752 / 0.761) → vào danh sách đánh giá public gốc đủ 1000 câu.
- Súp WC α 0.35 (`huyenco/zc-wisewc035-1`, đã chạy với cờ mới): quick400 0.785, pub_full (150) 0.880, **pub_ho 0.599** (cao nhất), tr_ho 0.870
  → gộp 0.754 (α 0.5 cùng cơ sở: 0.761) → ngang; quyết định bằng public gốc đủ 1000 câu (đang chạy: `pf*` kernel).
- WC5–8 (`synchan5704/zc-w7c2-1`; WC5 D1 seed 21, WC6 D1 + D2 mẫu mới, WC7 D1 + D2N mẫu mới, WC8 D1 + D4T) huấn luyện xong → súp 8 thành viên v2
  α 0.5 / 0.35 trên public gốc đủ + pub_ho + tr_ho (`huyenco/zc-soup8v2a05-1`, `synchan5704/zc-soup8v2a035-1`).

## Entry 45 — 2026-11-09: public gốc đủ 1000 câu (ưu tiên của user) — ứng viên

| Ứng viên (code `183953d`, cờ mới) | **pub_full 1000** | đúng / sai / bỏ | thời gian đánh giá |
|---|---|---|---|
| Gốc, prompt full | 0.906 | 930 / 24 / 46 | 727 s |
| Gốc, prompt full + think vòng 2 | **0.924** | 946 / 22 / 32 | 771 s |
| Súp WF1–4 α 0.5 (prompt full) | **0.923** | 942 / 19 / 39 | 1,139 s (gồm gộp) |
| *mốc v28 (LLM luồng cũ)* | *0.921* | | |

→ thêm súp WF α 0.5 + think (`quynhle07/zc-pfwf05think-1`, + pub_ho / tr_ho).
| Súp WC1–4 α 0.5 (compact) + think vòng 2 (`mucnokieu/zc-pfwc05think-1`) | **0.926** | 945 / 19 / 36 | |
| Súp WC1–4 α 0.5 (compact), không think (`mucnokieu/zc-pfwc05-1`) | 0.913 | 931 / 18 / 51 | |

  Think vòng 2 trên public gốc: gốc +1.8 (0.906 → 0.924), súp WC +1.3 (0.913 → 0.926) — chủ yếu đổi câu bỏ thành câu đúng.
| Súp WC1–4 α 0.35 (compact), không think (`huyenco/zc-pfwc035-1`) | 0.913 | 930 / 17 / 53 | |
| Súp top-4 cũ (v1) α 0.35, không think (`phatle1578/zc-pfs4035-1`) | 0.921 | 936 / 15 / 49 | |
| Súp WC α 0.5 + think **mọi mẫu** (6144 token, k 2 → 4; `mucnokieu/zc-pfwc05allthink-1`) | 0.921 | 951 / **30** / 19 | 2,284 s |

  Think mọi mẫu: đúng nhiều nhất (951) nhưng sai tăng (30 vs 19), gấp đôi thời gian → **không dùng**; think chỉ ở vòng 2 tốt hơn và nhanh hơn.
| **Súp WF1–4 α 0.5 (prompt full) + think vòng 2** (`quynhle07/zc-pfwf05think-1`) | **0.939** | 955 / 16 / 29 | 4,423 s (gồm pub_ho + tr_ho) |

  WF + think: pub_ho 0.588, tr_ho 0.869 → held-out gộp 0.748 (gốc 0.713; súp WC α 0.5 cờ mới 0.761) → rào chắn held-out đạt. Xác nhận:
  lặp seed (`pf_wf05_think_s1`) và α 0.35 + think (`pf_wf035_think`).
| Súp 8 thành viên v2 compact (WC1–8) α 0.35, không think (`synchan5704/zc-soup8v2a035-1`) | 0.913 | 927 / 14 / 59 | pub_ho 0.586, tr_ho 0.874 (gộp 0.750) = súp 4 thành viên → thêm thành viên không giúp |

## Entry 46 — 2026-11-09: WP9 vòng 4 — mẫu "nói lan man" không có chương trình (think rescue)

- Lỗi public gốc của súp WF + think (dev, 1000 câu): 95.5 % đúng, 1.7 % đúng-nhưng-bỏ, 1.6 % bất đồng, 1.2 % đồng thuận sai. Cơ chế lớn
  nhất: mẫu không-think viết lập luận bằng văn xuôi ("toolkit không có hàm đếm mặt hàng có ≥ N đơn ...") tới trần 512 token mà không
  có `<code>` (dù `int((item_sales(p, by="n_orders") >= 128).sum())` hợp lệ); lượt sửa hỏi lại vẫn lan man. Tần suất (mẫu bị cắt không
  chương trình / câu bị ảnh hưởng): súp WF + think 145 / 62 câu, **súp WC + think 217 / 100 câu**, gốc + think 16 / 12 câu → súp fine-tune
  dễ mắc hơn nhiều.
- **Sửa (luồng, không đổi prompt để không lệch dữ liệu huấn luyện):** `--think-rescue 1` — mẫu bị cắt ở trần token mà không có chương
  trình được chạy lại một lần ở chế độ suy nghĩ (cùng prompt, bỏ lượt lan man; `--think-tokens`). Test: mô hình giả "lan man" → 68 / 68
  mẫu được cứu (0 khi tắt); ca `o7` / `o8` trong `test_flow_hybrid`.
- Súp top-4 cũ α 0.35 + think vòng 2 (`phatle1578/zc-pfs4035think-1`): pub_full **0.901** (927 / 26 / 47) < không think 0.921; pub_ho 0.580,
  tr_ho 0.876 → think không giúp ổn định trên mọi mô hình; chờ hai lần lặp seed (WC + think, WF + think) để đo nhiễu public gốc trước khi chọn.
- Súp 8 thành viên v2 α 0.5, không think (`huyenco/zc-soup8v2a05-1`): pub_full 0.899 (924 / 25 / 51), pub_ho 0.581, tr_ho 0.878 (α 0.35: 0.913) →
  không hơn súp 4 thành viên.
- **Súp WF α 0.35 + think** (`quynhle07/zc-pfwf035think-1`): pub_full **0.939** (956 / 17 / 27) = α 0.5; pub_ho 0.592, tr_ho 0.873 → held-out gộp
  **0.752** (α 0.5: 0.748) → hai α độc lập cùng 0.939: súp WF (prompt full) + think vòng 2 ổn định; α 0.35 nhỉnh hơn ở held-out.
- Lặp seed súp WF α 0.5 + think (`quynhle07/zc-pfwf05thinks1-1`): pub_full **0.930** (951 / 21 / 28; lần 1 0.939), pub_ho 0.586, tr_ho 0.868 →
  nhiễu public gốc ≈ ±0.5–1 điểm; trung bình 3 lần súp WF + think ≈ **0.936** (dẫn đầu mọi ứng viên).
- **Lỗi hạ tầng:** `huyenco/zc-pfwc05thinks1-1` chạy **code cũ** (config `configs/e0_think_s1.json` thêm sau lần tải code trước không có trong
  dataset được gắn: kernel được đẩy khi phiên bản code mới chưa sống; `fanout` chỉ chờ trạng thái "ready" — có thể là phiên bản trước).
  Sửa: `make_kernel` ghi mã phiên bản ngẫu nhiên vào `ZC_CODE_MARKER` và vào runner; runner so khớp khi khởi động, lệch → dừng ngay
  "STALE CODE" (không bao giờ chạy code cũ âm thầm). Lần lặp WC + think không đo được (WF dẫn đầu nên không chạy lại).
- **Think rescue trên mô hình thật** (`phatle1578/zc-pfwc05rescue-1`, súp WC α 0.5 + think vòng 2 + rescue): 1,253 mẫu được cứu (1,001 hợp lệ),
  nhưng pub_full **0.913** (940 / **27** / 33) < 0.926 không rescue (945 / 19 / 36); pub_ho 0.583, tr_ho 0.874 → câu trả lời được cứu kém tin
  cậy hơn (sai tăng). Chờ súp WF + rescue trước khi quyết định.
