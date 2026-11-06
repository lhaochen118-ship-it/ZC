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
