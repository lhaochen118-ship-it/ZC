# Environment Summary — DataMate v3 (cập nhật 03/11/2026 14:30 UTC)

*Cấu hình các môi trường dự án dùng. Bài học vận hành ở PDF 2; vận hành Kaggle chi tiết ở `docs/02_KAGGLE_OPS.md`.*

## 1. Container phát triển (Claude Code trên cloud)

| Mục | Giá trị |
|---|---|
| Máy | 4 nhân CPU, 15 GB RAM, **không GPU**; hạn mức đĩa riêng (df không phản ánh đúng) |
| Python | 3.13; pandas 3.0.5, numpy 2.5.3, openpyxl 3.1.5, python-calamine 0.9.0, kaggle CLI 2.2.5 |
| PDF | xhtml2pdf 0.2.28, reportlab 5.0.1, pypdf 6.17 (`automation/md2pdf.py`, font DejaVu cho tiếng Việt) |
| vLLM | **không có** (chỉ chạy test CPU với mô hình giả `--fake-responder`) |
| Repo | `/home/user/ZC`, branch `claude/datamate-v3-0dny90` |
| Scratch | thư mục scratchpad của phiên (runs/, cache_train/, cache_pub/) — **mất khi sang khung chat mới** |
| Token Kaggle | `~/.kaggle/tok_<username>` (chmod 600), xin lại user mỗi khung chat |
| Mạng | qua proxy của môi trường; Kaggle API dùng được |
| Chỉ dùng cho | sửa code, test CPU (`tests/run_all.sh` ~10 phút), preflight (~10–12 phút), phân tích log, bỏ phiếu offline |

Dựng lại scratch sau khi mất container (cấu trúc cần có: `data/raw/train/train/`, `data/raw/public_test/public_test/`):
```bash
kaggle datasets download nhatlong1103/zc-2026 -p data/raw --unzip
python3 src/ingest.py data/raw/train/train <S>/cache_train 4
python3 src/ingest.py data/raw/public_test/public_test <S>/cache_pub 4
```

## 2. Kaggle (nơi chạy GPU)

| Mục | Giá trị |
|---|---|
| Máy | `machine_shape: NvidiaL4` → **4×L4** (22.5 GB mỗi GPU, không NVLink), 48 CPU; không internet |
| Điều kiện L4 | `competition_sources: ["gemma-4-developer-agent"]` + acc đã accept rules trên web |
| Mô hình | `lelynhan/zc-m-qwen35-9b` (Qwen3.5-9B bf16, lượng tử FP8 online `--quant fp8`) |
| Thư viện | `nhatlong1103/zc-assets` (wheels vLLM 0.32 + deps); code: dataset `<acc>/zc-code` (mỗi lần đẩy một version) |
| Dữ liệu | `nhatlong1103/zc-2026` |
| Song song | `--dp 4` (mỗi GPU một bản mô hình) + `--pipeline` (work stealing theo DN) |
| Đo được | ~200 token sinh / s / GPU; bận 84–94 %; VRAM đỉnh 21.6 / 22.5 GB; KV ≈ 201k token / GPU (`--gpu-util 0.92`) |
| Thời gian | train 3000 câu ≈ 3800–4200 s; public 1000 câu ≈ 4700–7100 s (pha ánh xạ LLM chiếm phần lớn) |
| Giới hạn | 2 phiên GPU đồng thời / acc; **30 giờ GPU / tuần / acc** |

### Tài khoản (03/11)

| Trạng thái | Acc |
|---|---|
| L4, đang chạy v21 / v22 | huyenco, mucnokieu (v21); thnliu, duylnng (v22) |
| L4, rảnh | linhmai24, phuongnehe, hai1108, nguynduylm, lelynhan |
| Hết quota tuần | caotienminh1 (03/11 14:02) |
| Chỉ T4 / cần accept rules | mxuanquy (T4), nguyenmanh0404 (chưa accept `gemma-4-developer-agent`) |
| Chỉ đăng dataset | nhatlong1103 (`~/.kaggle/access_token`; không dùng chạy kernel) |

Acc dùng chung với dự án khác: chỉ thao tác kernel `zc-*`.

## 3. Máy chấm (BTC) và Docker nộp

| Mục | Giá trị |
|---|---|
| Phần cứng | 2×RTX 5090 (32 GB), ≤ 6 giờ, `--network none`, `/data` chỉ đọc, kết quả `/result` |
| Image | `docker/Dockerfile`: `FROM vllm/vllm-openai:v0.32.0`, pandas 2.3.3, numpy 2.1.3, openpyxl, python-calamine |
| Chạy | `docker/predict.sh`: tối đa 3 lần chạy (seed 0/1/2) trong ngân sách thời gian, rồi `src/vote.py` |
| Hướng dẫn | `docker/SETUP_DOCKER.md` (build, chạy thử, xuất `.tar.gz` + checksum), `docker/README_SOLUTION.md` |
| Chưa làm | build + đo trên máy GPU thật (2×RTX 5090), kiểm |ΔPA| giữa hai lần chạy image |
