# Hướng dẫn dựng Docker cho bài nộp DataMate (trên máy của bạn)

Mục tiêu: build image chứa code + trọng số mô hình, chạy thử đúng như BTC chạy lại (`--network none`, `/data` chỉ
đọc, kết quả ở `/result`), rồi xuất file `.tar.gz` + checksum để nộp qua form "Solution Submission" (link Google
Drive public). Mọi lệnh chạy từ **thư mục gốc repo ZC** (branch `datamate`).

## 0. Kiểm tra máy (một lần)

```bash
docker --version
```

```bash
nvidia-smi
```

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

- Lệnh thứ 3 phải in ra GPU. Nếu báo lỗi `could not select device driver ... gpu`, cài **NVIDIA Container Toolkit**
  (Ubuntu: `sudo apt-get install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`).
- Driver NVIDIA cần hỗ trợ CUDA ≥ 12.8 (driver ≥ 570). Trên Windows: dùng Docker Desktop + WSL2, chạy các lệnh trong
  terminal Ubuntu của WSL.
- Ổ đĩa trống: khoảng **60 GB** (image vLLM ~20 GB giải nén + mô hình 16 GB + file xuất).
- **Chỉ build và xuất image thì không cần GPU.** Chạy thử cần GPU NVIDIA: ≥ 24 GB VRAM cho cấu hình nộp
  (Qwen3-14B-FP8, GPU Ampere trở lên); GPU 16 GB thì xem mục 4b.

## 1. Lấy code

```bash
git clone -b datamate https://github.com/lhaochen118-ship-it/ZC.git
```

```bash
cd ZC
```

(Đã có repo thì `git pull origin datamate`.)

## 2. Tải trọng số mô hình (một lần, ~16 GB)

```bash
docker/fetch_assets.sh Qwen/Qwen3-14B-FP8
```

Trọng số nằm ở `models/Qwen3-14B-FP8/` (thư mục này không commit vào git).

## 3. Build image

```bash
docker/build.sh datamate:latest Qwen3-14B-FP8
```

Lần đầu tải base image `vllm/vllm-openai:v0.32.0` (~9 GB nén) nên lâu. Cuối bước build có dòng `agent import OK`.

## 4. Chạy thử trên public test (giống hệt BTC chạy lại)

Giải nén dữ liệu public test sao cho có một thư mục chứa `questions.jsonl`, `rulebook.md`, `calendar.csv`, `data/`
(ví dụ `~/zc_data/public_test/public_test`), rồi:

```bash
docker/run_public_test.sh ~/zc_data/public_test/public_test ./result_public
```

- Kết quả: `result_public/submission.jsonl` (đem nộp lên portal để xem điểm public), log từng câu ở
  `result_public/logs/<id>.json`, log tổng ở `result_public/logs/run.log`.
- `submission.jsonl` được ghi dần sau mỗi nhóm doanh nghiệp, nên dừng giữa chừng vẫn có các câu đã xong.

### 4b. GPU nhỏ hơn (16 GB)

Dùng bản AWQ (9.4 GB) và ngữ cảnh ngắn hơn:

```bash
docker/fetch_assets.sh Qwen/Qwen3-14B-AWQ
```

```bash
docker/build.sh datamate:awq Qwen3-14B-AWQ
```

```bash
DOCKER_ARGS="-e MAX_LEN=20480 -e K=2" docker/run_public_test.sh ~/zc_data/public_test/public_test ./result_awq datamate:awq
```

## 5. Xuất file để nộp

```bash
docker/export_image.sh datamate:latest datamate_solution.tar.gz
```

Upload `datamate_solution.tar.gz` lên Google Drive (chia sẻ "Anyone with the link"), dán link + nội dung file
`datamate_solution.tar.gz.sha256` vào form "Solution Submission". Hạn: **16/11 23:59 (GMT+7)**, tối đa 2 lần nộp
(BTC lấy lần cuối).

BTC có thể nạp lại image bằng `gunzip -c datamate_solution.tar.gz | docker load`.

## 6. Khi có Docker guideline của BTC

Nếu guideline chỉ định base image khác, chỉ cần sửa dòng `FROM` trong `docker/Dockerfile` rồi build lại (bước 3).
Cách gọi (`/data` → `/result/submission.jsonl`, log ở `/result/logs/`) đã đúng như mô tả trong tab Data.

## Gửi lại cho Claude

Sau mỗi lần chạy thử, gửi lại (hoặc commit vào repo): dòng cuối của `result_public/logs/run.log`, thời gian chạy
(lệnh `time` in ra), loại GPU (`nvidia-smi -L`) và điểm public nếu đã nộp lên portal.
