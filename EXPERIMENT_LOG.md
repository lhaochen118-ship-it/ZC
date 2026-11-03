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
