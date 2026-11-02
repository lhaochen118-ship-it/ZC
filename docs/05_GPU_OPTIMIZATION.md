# 05 — Tối ưu tốc độ GPU (playbook, dùng lại cho mọi phiên)

*2026-11-02. User yêu cầu: "tối ưu nhanh nhất có thể" và ghi cách làm để lần sau tận dụng. Tóm tắt nằm trong `CLAUDE.md`;
file này giải thích lý do và ghi số đo.*

## 1. Phần cứng và hệ quả

| Máy | GPU | Hệ quả |
|---|---|---|
| Kaggle phiên L4 | 4 × L4 24 GB, PCIe, không NVLink; 48 CPU | 4 GPU độc lập. Mô hình ≤ ~12 GB (14B AWQ, 9B FP8) → **1 bản / GPU** (data parallel). TP qua PCIe tốn truyền thông mỗi lớp, chỉ dùng khi mô hình không vừa 1 GPU |
| Kaggle phiên T4 | 2 × T4 16 GB | fp16 (T4 không có bf16 nhanh), mô hình int4 |
| Máy chấm | 2 × RTX 5090 32 GB, 512 GB RAM | `--dp 2`; 9B bf16 (~19 GB) vừa 1 GPU, còn ~10 GB KV; FP8 nếu cần thêm KV |

"96 GB VRAM phân tán" **không** phải một GPU 96 GB: KV cache, prefix cache và batch là của từng GPU.

## 2. Mất tốc độ ở đâu (đã thấy trong v1 / v2)

| Nguồn lãng phí | Triệu chứng | Cách xử lý (đã cài) |
|---|---|---|
| Đầu ra dài | v1: chương trình pandas dài + thinking, 25–98 s/câu/L4 | Việc của LLM hẹp: chương trình toolkit ~100 token (`src/dm.py` giữ ngữ nghĩa) |
| Prefix bị tính lại | v1 E2b: gửi mọi DN cùng lúc → prefix đẩy nhau khỏi KV, chậm ~12× | Phần chung trước, phần riêng sau; gom theo DN; ≤ 1–2 DN đang trả lời mỗi GPU (`--active`) |
| Nhóm quá nhỏ | v2: EXAMPLES riêng mỗi câu nằm trước `# QUESTION` → mỗi câu một nhóm, 4 request / lần gọi | `Engine.prefix_of`: cắt ở `# EXAMPLES` / `# QUESTION` / `# TASK` |
| Rào chắn giữa vòng | sinh cả đợt → GPU chờ CPU chạy chương trình → mới sinh vòng sửa; chờ request dài nhất | `--pipeline`: `add_request`/`step`, CPU chạy song song, sửa lỗi vào hàng ngay |
| Chia tải tĩnh | GPU gặp DN khó chạy lâu, GPU khác ngồi chờ | Claim file nguyên tử, DN nhiều câu trước (LPT) |
| Đọc mô hình từ mount mạng | E2b: 4 tiến trình đọc mount mất 18 phút | `run_kaggle.py` chép mô hình sang `/tmp/models` một lần |
| Ingest lặp | mỗi tiến trình GPU tự ingest | `run_dp` ingest mọi DN một lần (mọi CPU) trước khi mở tiến trình GPU |

## 3. Cấu hình vLLM

- `gpu_memory_utilization` 0.92 (`--gpu-util`), `enable_prefix_caching=True`, seed cố định.
- KV thật: dòng log `[engine] ... KV cache N tokens | prefix budget M`. Qwen3-14B: 40 lớp × 8 KV head × 128 × 2 × 2 B
  = 160 KB/token. Qwen3.5-9B: chỉ 8/32 lớp full attention, 4 KV head × 256 → ~32 KB/token (+ trạng thái linear attention
  cố định mỗi chuỗi) → xếp được nhiều prefix / chuỗi hơn ~5×.
- `--max-num-seqs`, `--max-batched-tokens`: để mặc định đến khi có số đo; tăng khi `gpu_summary.json` cho thấy util thấp
  mà KV còn trống.
- `--spec-mtp N`: speculative decoding bằng đầu MTP có sẵn của Qwen3.5 (tự quay về giải mã thường nếu vLLM không hỗ trợ).
  Chỉ giữ khi tok/s tăng và PA không giảm.

## 4. Kaggle

- Mỗi acc: 2 phiên GPU + 5 phiên CPU cùng lúc. Dồn luồng lên một acc trước, hết slot mới sang acc khác
  (`fanout.py --accounts a,b`).
- Thí nghiệm nhỏ trước (vài DN), thí nghiệm quyết định ≥ 300 câu; so cặp trên cùng câu.
- Số đo bắt buộc mỗi run: `gpu_summary.json` (util), log `[pipe] ... tok/s`, `DP DONE ... s`.

## 5. Số đo (điền khi có)

| Run | Cấu hình | Câu | Wall (s) | GPU util TB | tok/s | PA |
|---|---|---|---|---|---|---|
| v1q14-1 | round-based, Qwen3-14B-AWQ, 4×L4 | 3000 train | 4170 | (chưa có monitor) | | 0.725 |
| v1q14-2 | round-based, Qwen3-14B-AWQ, 4×L4 | 1000 public | 3390 (shard 3355 / 2804 / 2146 / 2023: lệch 40 %) | | | |
| v1q35-1 | round-based, Qwen3.5-9B FP8 | 750 + 749 train (2 shard) | 2424 / 2549 | | | 0.900 / 0.877 |
| p1q14b-1 | `--pipeline`, Qwen3-14B-AWQ | 3000 train | đang chạy | | | |
| p2q35-1 | `--pipeline`, Qwen3.5-9B FP8 | 3000 train | đang chạy | | | |

Chi phí cố định mỗi kernel (đo 02/11): `pip install` vLLM từ wheels 398–471 s, chép mô hình 66–158 s, nạp vLLM ~3 phút.
→ môi trường dựng sẵn `lelynhan/zc-env-vllm032` (đang dựng) để bỏ bước pip.
