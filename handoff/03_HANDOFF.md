# PDF 3 — Handoff (DataMate v3, khung chat 03/11/2026 → khung mới)

*Phạm vi: mục tiêu, tiến độ, quyết định đã thống nhất, vị trí mã / dữ liệu / kết quả, các bước tiếp theo. Bằng chứng và
phân tích ở PDF 1; kinh nghiệm kỹ thuật ở PDF 2; cấu hình máy ở `environment_summary.pdf`; hướng dẫn đọc repo ở
`READING_GUIDE.md`. Trạng thái sống (cập nhật liên tục): `handoff/STATUS.md`.*

## 1. Mục tiêu

- Zalo AI Challenge 2026, track **DataMate**: PA cao nhất trên **private test** và nhanh nhất có thể.
- Ràng buộc: ≤ 15B tham số, offline, ≤ 6 giờ trên 2×RTX 5090, |ΔPA| ≤ 0.03 giữa các lần chạy. Nộp Docker trước
  **16/11/2026 23:59 GMT+7** (user nộp portal).
- Mục tiêu user đặt: hướng tới đúng 100 % trên valid và test; luật **tổng quát**, không fix theo DN / câu / dataset.

## 2. Tiến độ (14:30 UTC 03/11)

- Pipeline v3 hoàn chỉnh, chạy trên Kaggle 4×L4 trong ~1.1–2.3 giờ / lần chạy, Docker + bỏ phiếu 3 lần chạy đã có.
- Điểm portal tốt nhất: **v19 = 0.687** (`submissions/v19/submission.jsonl`). v20 = 0.650.
- Train LOO tốt nhất: v20 0.991 (holdout 0.977).
- Code mới nhất **chưa có điểm public**: v21 (8dca4a2), v22 (20cfec7), bỏ phiếu biết ánh xạ (7fa70f2).

### Kernel còn chạy khi đóng khung chat

| Bản | Code | Kernel (acc/slug) | Job | Dự kiến xong |
|---|---|---|---|---|
| v21 | 8dca4a2 | huyenco/zc-v21-1, huyenco/zc-v21-2, mucnokieu/zc-v21-3, mucnokieu/zc-v21-4 | train, pub, pub_s1, pub_s2 | ~15:00 UTC |
| v22 | 20cfec7 | thnliu/zc-v22-1, thnliu/zc-v22-2, duylnng/zc-v22-3, duylnng/zc-v22-4 | train, pub, pub_s1, pub_s2 | ~16:15 UTC |

Kaggle giữ output kernel sau khi xong; tải bằng token của acc tương ứng (bước 1 của §5).

## 3. Quyết định đã thống nhất (không hỏi lại)

- Kiến trúc v3 (`docs/04_DECISION_V3.md`): ánh xạ theo DN → bảng chuẩn → toolkit `dm.Shop` → luật + mẫu LLM → bỏ phiếu.
- Mô hình **Qwen3.5-9B FP8** (vLLM 0.32); cấu hình chuẩn ở `automation/jobs/v22.jobs`.
- Nộp cuối = **bỏ phiếu 3 lần chạy** (seed 0/1/2) trong `docker/predict.sh` (min_share 0.4, tie-break theo ánh xạ).
- Mọi sửa lỗi là quy tắc tổng quát + test; train oracle phải giữ **2967 / 1 / 32**, ánh xạ tham chiếu **0 flagged**.
- Cổng trước khi chạy: review diff → `tests/run_all.sh` → preflight (fanout tự kiểm). Phần chỉ chạy trên Kaggle → chạy nhỏ trước.
- Không gán nhãn tay đáp án public, không train trên test. Chỉ tạo file nộp khi kết quả tốt; **push ngay** + gửi user.
- Token Kaggle chỉ ở `~/.kaggle/tok_<user>` (chmod 600); khung mới phải xin lại. Không đụng kernel không phải `zc-*`.
- Watcher nền thay cho poll; dừng / chờ đúng cách (PDF 2 §2).
- Routine tự chạy `trig_01U26KaZP68Dmg8JL2RGcbVS` **đã tắt** (user dừng khung chat).

## 4. Vị trí mã nguồn, dữ liệu, kết quả

| Thứ | Vị trí |
|---|---|
| Repo / branch | `lhaochen118-ship-it/ZC`, branch **`claude/datamate-v3-0dny90`** |
| Pipeline | `src/agent.py` (CLI, prompt, engine), `src/pipeline.py` (StepLLM, chọn ánh xạ, bỏ phiếu theo câu), `src/canon.py` (chạy ánh xạ, kiểm tra), `src/dm.py` (toolkit), `src/qparse.py` (luật đọc câu), `src/qnorm.py` (làm sạch câu), `src/vote.py` (bỏ phiếu nhiều lần chạy) |
| Ánh xạ tham chiếu / rulebook train / README train | `src/refs/compile/`, `src/refs/train_rulebook.md`, `src/refs/readmes/` |
| Kaggle | `automation/fanout.py`, `make_kernel.py`, `run_kernel.py`, `watch_done.py`, `preflight.py`, `make_submission.py`, `map_penalties.py`; job + công cụ phân tích: `automation/jobs/` |
| Test | `tests/run_all.sh`, `tests/eval_lib.py` (oracle), `tests/test_plausibility.py`, `tests/test_mapping_fixes.py` |
| Dữ liệu BTC | Kaggle dataset `nhatlong1103/zc-2026` → `data/raw/{train,public_test}/` (không commit) |
| Làm sạch câu train | `question_cleaning/` |
| File nộp | `submissions/vNN/submission.jsonl` + README (v11, v13, v15, v18, **v19**, v20) |
| Nhật ký / trạng thái | `EXPERIMENT_LOG.md` (entry 0–13), `handoff/STATUS.md` |
| Docker | `docker/` (`Dockerfile`, `predict.sh`, `README_SOLUTION.md`, `SETUP_DOCKER.md`) |
| Kết quả kernel cũ | thư mục scratch của khung cũ — **đã mất**; tải lại từ Kaggle nếu cần (`kaggle kernels output <acc>/zc-vNN-k`) |

## 5. Các bước tiếp theo (thứ tự)

1. **Xin user token Kaggle**, lưu `~/.kaggle/tok_<user>` (600). Tải dữ liệu: `kaggle datasets download nhatlong1103/zc-2026`
   → `data/raw/`. Ingest public: `python3 src/ingest.py data/raw/public_test/public_test <cache_pub> 4`.
2. **Tải output v21 / v22**: `KAGGLE_API_TOKEN=$(cat ~/.kaggle/tok_<acc>) kaggle kernels output <acc>/<slug> -p <S>/runs/vNN/<slug>_out`.
3. Chấm train: `python3 src/report.py data/raw/train/train <S>/runs/vNN/zc-vNN-1_out/results/train`.
4. File nộp: `python3 automation/map_penalties.py <cache_pub> <run2>/results/pub <run3>/results/pub_s1 <run4>/results/pub_s2`
   rồi `S=<S> bash automation/jobs/mksub.sh vNN` → README → **push** → gửi user. So từng câu với v19 (giống / đổi /
   thêm abstain) trước khi đề nghị nộp; đề nghị bản có ít câu chuyển sang abstain ở DN lớn.
5. **Ổn định ánh xạ giữa các lần chạy** (đòn bẩy public lớn nhất): gộp ứng viên ánh xạ của các lần chạy (hoặc chọn bằng
   kiểm tra + đồng thuận trên toàn bộ ứng viên), để mọi lần chạy dùng cùng ánh xạ tốt nhất.
6. Lỗi chung đã biết chưa sửa xong: nợ chỉ trong ghi chú (dev_008; v22 có cảnh báo + `money_in_text`); định nghĩa ngành
   điện thoại (dòng máy không gồm phụ kiện); tìm thêm đáp án 0 bất thường (từ địa phương, tên viết khác).
7. Docker cuối: build trên máy GPU, đo 3 lần chạy trên 2×RTX 5090 (≤ 6 h), kiểm |ΔPA| giữa 2 lần build.

## 6. Tài nguyên

- Acc Kaggle (L4): duylnng, huyenco, mucnokieu, thnliu, linhmai24, phuongnehe, hai1108, nguynduylm, lelynhan;
  mxuanquy (chỉ T4), nguyenmanh0404 (cần accept rules). **caotienminh1 hết quota tuần** (30 h, đặt lại theo tuần Kaggle).
- Mỗi bản thử đầy đủ ≈ 8 giờ GPU (4 kernel × ~2 h). Cần nhiều bản song song → xin user thêm acc L4 đã accept rules
  `gemma-4-developer-agent`.
