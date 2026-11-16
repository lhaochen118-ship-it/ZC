# Docker_submission — hướng dẫn build, chạy thử và nộp

Thư mục này đóng gói **cấu hình nộp cuối = framework mạnh nhất** `configs/flow_p5_final.json` (Qwen3.5-9B + V3A × 0.70, WL1, WJR1,
toolkit có `industry.py`): public **0.952** (D1, `quynhtran35/zc-d1-1`). **`submission.jsonl`** trong thư mục này là file do chính lượt chạy
đó tạo ra (đầu ra phán xử = file container sau khi sửa lỗi gộp; 1000 câu, 973 đúng / 21 sai / 6 bỏ; bản sao `final_submission/p5_container_d1/submission.jsonl`). README cho BTC (khai báo mô hình, phương pháp): `README_SOLUTION.md`. Mọi lệnh chạy từ **thư mục gốc repo**
(branch `claude/datamate-v3-0dny90`); tên thư mục có dấu cách nên luôn đặt trong ngoặc kép.

## 1. Đọc gì trước

| File | Nội dung |
|---|---|
| `submission.jsonl` | file nộp public của cấu hình này (D1, PA 0.952) |
| `README_SOLUTION.md` | README nộp kèm (mô hình ≤ 15B, giấy phép, phương pháp, định dạng log) |
| `Dockerfile` | 2 stage: gộp LoRA V3A × 0.70 vào Qwen3.5-9B → image chạy (vLLM 0.32.0, CUDA ≥ 12.8) |
| `predict.sh` | điểm vào: đọc `/data`, ghi `/result/submission.jsonl` + `/result/logs/` |
| `fetch_assets.sh` | tải trọng số Qwen3.5-9B (Hugging Face) + 3 adapter (Kaggle), kiểm sha256 |
| `assets.env` | tên dataset Kaggle chứa adapter |
| `build.sh`, `run_public_test.sh`, `export_image.sh` | build, chạy thử như BTC, xuất file nộp |

## 2. Chuẩn bị môi trường (một lần)

- Linux (hoặc Windows + Docker Desktop + WSL2, chạy lệnh trong terminal Ubuntu), Docker ≥ 24, ổ trống **≥ 80 GB**.
- Chạy thử cần GPU NVIDIA ≥ 24 GB VRAM (máy chấm: 2 × RTX 5090), driver ≥ 570 (CUDA ≥ 12.8) và NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

  Lệnh trên phải in ra GPU. Nếu lỗi `could not select device driver`: cài `nvidia-container-toolkit`, chạy
  `sudo nvidia-ctk runtime configure --runtime=docker` rồi khởi động lại Docker. Chỉ build / xuất image thì không cần GPU.
- Python 3.10+ với `pip` (để chạy `fetch_assets.sh`), và token Kaggle của **tài khoản có quyền đọc** dataset trong `assets.env`
  (`nhatlong1103/zc-final-adapter`, `nhatlong1103/zc-w-v2`) đặt ở `~/.kaggle/kaggle.json` (quyền 600). Nếu dataset riêng tư: dùng token
  của chính tài khoản nhatlong1103. Không đưa token vào repo.

## 3. Tải trọng số (~20 GB) và build

```bash
git clone -b claude/datamate-v3-0dny90 https://github.com/lhaochen118-ship-it/ZC.git && cd ZC
```

```bash
Docker_submission/fetch_assets.sh Qwen/Qwen3.5-9B
```

Kết quả: `models/Qwen3.5-9B/` (sha256 khớp `src/model_hashes.json`), `models/adapter/` (V3A + `soup.json` trọng số 0.70) và
`models/adapters/{WL1,WJR1}/adapter/` (sha256 khớp `src/adapter_hashes*.json`; script dừng nếu sai).

```bash
Docker_submission/build.sh datamate:latest
```

Lần đầu tải base image `vllm/vllm-openai:v0.32.0` (~9 GB). Cuối build phải có dòng `agent import OK`.

## 4. Chạy thử trên public test (giống hệt BTC: không mạng, `/data` chỉ đọc)

Giải nén public test sao cho một thư mục chứa `questions.jsonl`, `rulebook.md`, `calendar.csv`, `data/`, rồi:

```bash
Docker_submission/run_public_test.sh ~/zc_data/public_test ./result_public
```

- `result_public/submission.jsonl`: một dòng mỗi câu, ghi dần trong lúc chạy (dừng giữa chừng vẫn giữ các dòng đã ghi).
- `result_public/logs/<id>.json`: id, đáp án, thời gian, chương trình mô hình viết, output thực thi, phiếu bầu, phán xử;
  `result_public/logs/run.log`: log tổng (các pha: thẻ luật W1 → hàm cửa hàng → link → giải → phán xử).
- Thời gian tham chiếu: 69–110 phút trên Kaggle 4 × L4 cho 1000 câu; máy chấm 2 × RTX 5090 dự kiến tương đương hoặc nhanh hơn.
  Ngân sách mặc định 5 h 40 phút (`BUDGET=20400`), tự dừng các pha còn lại khi gần hết giờ.
- Điểm public kỳ vọng ~0.94–0.95 (D1 0.952; chạy lại bằng mã hiện tại 0.942). Lệch quá ±0.03 → kiểm `run.log` (dòng `units: rc=[...]`, `links: rc=[...]`,
  `judge: ...`) trước khi nộp.

## 5. Xuất file nộp

```bash
Docker_submission/export_image.sh datamate:latest datamate_solution.tar.gz
```

Upload `datamate_solution.tar.gz` (chia sẻ "Anyone with the link"), dán link + nội dung `datamate_solution.tar.gz.sha256` vào form
"Solution Submission". BTC nạp lại bằng `gunzip -c datamate_solution.tar.gz | docker load`.

## 6. Lưu ý kỹ thuật (đã kiểm)

- **Tên thư mục mô hình trong image là `/models/solver`, không chứa "qwen3".** Template Qwen3.5 tự suy nghĩ khi không truyền
  `enable_thinking`; mã truyền cờ này (tắt suy nghĩ) chỉ khi đường dẫn có "qwen3". Mọi lượt đã chấm phục vụ thư mục merged không có
  chữ đó (suy nghĩ bật). Đặt tên khác sẽ đổi hành vi và làm giảm điểm (đã đo: 0.952 → 0.921 ở framework có industry).
- Nếu guideline của BTC yêu cầu base image khác: chỉ sửa 2 dòng `FROM` trong `Dockerfile`, build lại, chạy lại mục 4.
- Biến môi trường ghi đè được: `MODEL`, `CONFIG`, `BUDGET`, `DP` (xem `predict.sh`); mặc định là cấu hình nộp.
- Thư mục `docker/` cũ giữ để tham khảo lịch sử; **dùng thư mục này để nộp**.
