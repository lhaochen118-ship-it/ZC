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
