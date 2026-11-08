# Environment Summary — DataMate (cập nhật 08/11/2026)

*Cấu hình các môi trường dự án dùng. Bài học vận hành ở PDF 2; vận hành Kaggle chi tiết ở `docs/02_KAGGLE_OPS.md`.*

## 1. Container phát triển (Claude Code trên cloud)

| Mục | Giá trị |
|---|---|
| Máy | 4 nhân CPU, 15 GB RAM, **không GPU, không Docker daemon**; ~24 GB đĩa trống (hạn mức riêng) |
| Python | 3.13; pandas 3.x, numpy 2.x, openpyxl, python-calamine, kaggle CLI |
| PDF | `pip install markdown xhtml2pdf` (mất khi đổi container); `automation/md2pdf.py`, font DejaVu |
| vLLM | không có — test CPU dùng mô hình giả (`--fake-responder`) |
| Repo | `/home/user/ZC`, branch `claude/datamate-v3-0dny90` |
| Scratch | thư mục scratchpad của phiên (`runs/`, `cache_train/`, `cache_pub/`, `ana/`) — **mất khi sang khung chat mới** |
| Token Kaggle | `~/.kaggle/tok_<username>` (chmod 600); xin lại user mỗi khung chat |
| Tác vụ nền | trần 2 giờ; container có thể khởi động lại giữa phiên (tiến trình nền mất) |

Dựng lại sau khi mất container:
```bash
kaggle datasets download nhatlong1103/zc-2026 -p data/raw --unzip
python3 src/ingest.py data/raw/train/train <S>/cache_train 4
python3 src/ingest.py data/raw/public_test/public_test <S>/cache_pub 4
```
Đáp án public (`data/raw/public_gt.jsonl`): user gửi (`score_1.jsonl`). Paraphrase: `kaggle datasets download <acc>/zc-para`.

## 2. Kaggle (nơi chạy GPU)

| Mục | Giá trị |
|---|---|
| Máy | `NvidiaL4` → **4×L4** (22.5 GB / GPU, không NVLink), 48 CPU; không internet |
| Điều kiện L4 | `competition_sources: ["gemma-4-developer-agent"]` + acc đã accept rules trên web |
| Mô hình | Kaggle Model `danbth/qwen3-5-9b/transformers/default/1` (bf16, FP8 online `--quant fp8`), kiểm sha256 |
| Thư viện | `nhatlong1103/zc-assets` (wheels vLLM 0.32 + deps); code đẩy theo kernel |
| Dữ liệu | `nhatlong1103/zc-2026`; paraphrase / câu chờ: `<acc>/zc-para`, `<acc>/zc-pend` (private, mỗi acc một bản) |
| Song song | `--dp 4` + `--pipeline` (work stealing theo DN) |
| Đo được | ~200 token sinh / s / GPU; bận 84–94 %; KV ≈ 201k token / GPU (`--gpu-util 0.92`) |
| Thời gian | LLM 1 lần chạy: public 1000 câu ≈ 55–75 phút, train 3000 câu ≈ 65–70 phút; luồng nộp v34 trên 2065 paraphrase: 3.1 h |
| Giới hạn | 2 phiên GPU + 5 CPU đồng thời / acc; **30 giờ GPU / tuần / acc, tính theo GPU**; reset thứ Bảy 00:00 UTC |

### Tài khoản (08/11)

| Acc | Loại | Tình trạng tuần W46 (từ 07/11) |
|---|---|---|
| quynhtran35, phatle1578 | L4 | chưa dùng |
| yuan1103 | L4 | đã dùng ~15 h (v33, v34) |
| yeshenhi, mxuanquy | chỉ T4 (không chạy được Qwen3.5-9B FP8) | — |
| nhatlong1103 | chỉ đăng dataset | — |

Acc dùng chung với dự án khác: chỉ thao tác kernel `zc-*`. `automation/accounts.json`: danh sách acc không có L4.

## 3. Máy chấm (BTC) và Docker nộp

| Mục | Giá trị |
|---|---|
| Phần cứng | 2×RTX 5090 (32 GB), ≤ 6 giờ, `--network none`, `/data` chỉ đọc, kết quả `/result` |
| Image | `docker/Dockerfile`: `vllm/vllm-openai:v0.32.0`, pandas 2.3.3, numpy 2.1.3, openpyxl, python-calamine; trọng số trong image |
| Chạy | `docker/predict.sh` → `src/predict_flow.py` (ngân sách 20400 s, gộp kết quả mỗi 20 s) |
| Hướng dẫn | `docker/SETUP_DOCKER.md` (`fetch_assets.sh` → `build.sh` → `run_public_test.sh` → `export_image.sh`), `docker/README_SOLUTION.md` |
| Chưa làm | build + đo trên GPU thật (user làm); luồng sẽ đổi sang LLM-first trước khi build bản cuối |
