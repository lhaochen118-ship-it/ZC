# Docker_PG — hướng dẫn

Lời giải DataMate (Zalo AI Challenge 2026) đóng gói thành Docker image. Điểm public: **0.880**.

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `Dockerfile` | 2 stage: gộp LoRA vào Qwen3.5-9B, rồi image chạy (vLLM 0.32.0, CUDA ≥ 12.8) |
| `app/src/` | mã nguồn: `predict_flow.py` (điểm vào), `ingest.py` / `automap.py` / `canon.py` (đọc dữ liệu), `dm.py` / `industry.py` (bộ hàm), `lrl_gen.py` / `units.py` (thẻ luật, hàm cửa hàng), `hybrid.py` / `pipeline.py` (sinh chương trình), `judge.py` (phán xử), `refs/` (ví dụ, từ điển) |
| `app/configs/solution.json` | cấu hình chạy |
| `app/training/merge.py` | gộp LoRA vào trọng số |
| `docker/predict.sh` | entrypoint: đọc `/data`, ghi `/result/submission.jsonl` + `/result/logs/` |
| `docker/merge_assets.py` | bước gộp lúc build |
| `scripts/` | `fetch_assets.sh` (tải trọng số), `build.sh`, `run_public_test.sh`, `export_image.sh` |
| `hashes/` | sha256 của trọng số gốc và adapter |
| `submission.jsonl` | kết quả trên public test |
| `README_SOLUTION.md` | mô tả mô hình và phương pháp cho BTC |

## 1. Môi trường

- Linux (hoặc Windows + Docker Desktop + WSL2), Docker ≥ 24, ổ trống ≥ 80 GB.
- Chạy thử cần GPU NVIDIA ≥ 24 GB (máy chấm 2 × RTX 5090), driver ≥ 570, NVIDIA Container Toolkit. Kiểm tra:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

- Python 3.10+ và token Kaggle có quyền đọc dataset adapter (`hashes/adapter.json`, `hashes/judge.json`: trường `dataset`) ở `~/.kaggle/kaggle.json`.

## 2. Tải trọng số (~20 GB) và build

Chạy trong thư mục này:

```bash
scripts/fetch_assets.sh
```

```bash
scripts/build.sh datamate:latest
```

Cuối build phải có dòng `import OK`.

## 3. Chạy thử trên public test (giống BTC: không mạng, `/data` chỉ đọc)

```bash
scripts/run_public_test.sh ~/zc_data/public_test ./result_public
```

Thư mục truyền vào phải chứa `questions.jsonl`, `rulebook.md`, `calendar.csv`, `data/`. Kết quả: `result_public/submission.jsonl`,
log ở `result_public/logs/`. Thời gian tham chiếu ~70–110 phút cho 1000 câu trên 4 × L4; giới hạn mặc định 5 h 40 phút (`BUDGET`).

## 4. Xuất file nộp

```bash
scripts/export_image.sh datamate:latest datamate_solution.tar.gz
```

Upload `datamate_solution.tar.gz`, gửi link cùng **checksum MD5** (nội dung file `datamate_solution.tar.gz.md5`, tạo sẵn bởi script) theo hướng dẫn nộp của BTC. Tự kiểm lại: `md5sum datamate_solution.tar.gz`.
