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
