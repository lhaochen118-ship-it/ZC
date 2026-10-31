# 03 — Handoff (DataMate, sang khung chat 3)

*Bàn giao 2026-10-31 01:10 UTC.*
- *Repo `lhaochen118-ship-it/ZC`, branch làm việc **`claude/peaceful-carson-0dny90`** (tiếp nối `datamate`). Không tạo PR.*
- *Kết quả và bằng chứng: PDF 1. Vận hành và lỗi: PDF 2. File này chỉ có mục tiêu, trạng thái, quyết định, vị trí, việc tiếp theo.*

## 1. Mục tiêu

- **Cuộc thi:** Zalo AI Challenge 2026, track DataMate (đề: `docs/00_TASK.md`).
- **Mục tiêu:** tối đa Penalized Accuracy trên private test, trong các ràng buộc:
  - offline, ≤ 15 B tham số;
  - ≤ 6 giờ trên 2 × RTX 5090;
  - tất định (|ΔPA| ≤ 0.03).
- **Hạn:**
  - nộp Docker **16/11 23:59 GMT+7**;
  - private test 20–23/11.
- **Public LB:** ≤ 5 lần nộp/ngày, user nộp tay.

## 2. Quyết định user đã chốt (giữ nguyên)

- **Tự chủ:** tự động hoá tối đa, không hỏi lại quyết định nghiên cứu thông thường. Vòng lặp: hiểu → đề xuất → chạy → chấm
  → cải tiến.
- **Nộp bài:** chỉ tạo file nộp khi kết quả tốt; user quyết định nộp.
- **Tính tổng quát:** bộ làm sạch / helper phải tổng quát, không vá cứng theo DN của dataset.
- **Kaggle:**
  - dùng mọi acc; mỗi acc đủ 2 phiên GPU; L4 trước, T4 sau;
  - QUEUED > 5 phút (L4) / 10 phút (T4) → huỷ và đẩy lại, tối đa 4 lần;
  - định kỳ dọn tiến trình treo; không tự ngắt được thì báo user.
- **Bảo mật token:** token chỉ ở `~/.kaggle`; `nhatlong1103` chỉ để đăng dataset (không chạy kernel).
- **Theo dõi:** bằng watcher nền, không tự kiểm tra định kỳ.
- **Tìm thông tin:** chỉ đọc văn bản / paper, không đọc code repo của người khác.

## 3. Trạng thái

| Hạng mục | Trạng thái |
|---|---|
| Làm sạch (`ingest.py`) + lớp ngữ nghĩa (`semantic.py`) | Xong, đã kiểm chứng |
| Agent | Cấu hình tốt nhất (mục 5) đạt **PA train dev +0.47 / holdout +0.44** (3000 câu, leave-business-out) |
| Thư viện shots | v4: 4228 chương trình (`src/shots/train_shots.jsonl`) |
| Public test | Run `p1` (cấu hình tốt nhất với shots v2) **đang chạy**; chưa có file nộp |
| Thinking | **Xác nhận có lợi:** e10l (L4, 316 câu) PA dev +0.69 vs +0.59 (không think), holdout +0.90 vs +0.70; chậm ≈ 2.5×. Chờ `e10b` (T4, holdout 40 câu) |
| Sửa chưa đo trên GPU | khoảng ngày dữ liệu → unanswerable; NOTES compile không gọi bảng T là thiếu; `SafePool`; gợi ý `time=` trong compile |
| Docker | `docker/predict.sh` đã dùng cấu hình tốt nhất; chưa build / chưa chạy thử; chưa có guideline BTC |
| FP8 cho 5090, tất định, thời gian thật | Chưa làm |

**Kernel còn chạy:** `handoff/RUNNING_KERNELS.md`. Kéo kết quả:

```bash
python3 automation/pull_kernels.py <scratch>/pulled
```

## 4. Vị trí

| Thứ | Ở đâu |
|---|---|
| Dữ liệu BTC | Kaggle `nhatlong1103/zc-2026` → `data/raw/` (không commit) |
| Mã | `src/` (agent, ingest, semantic, vnnorm, score, compare, analyze, build_shots, harvest, train_sft, run_kaggle); `automation/` (Kaggle); `tests/`; `docker/` |
| Assets offline | output kernel `nhatlong1103/zc-assets` (wheels vLLM 0.32 + Qwen3-14B-AWQ, `--ks`); FP8: dataset `nhatlong1103/zc-m-qwen3-14b-fp8` |
| Log mọi run khung chat 2 | `results/chat2_runs/<run>__<kernel>/*.jsonl.gz` (README trong đó) |
| Nhật ký thí nghiệm | `EXPERIMENT_LOG.md` entry 3–5 |
| Token | `~/.kaggle/tok_<acc>` (container cũ; khung chat mới **xin lại user**) |

## 5. Cấu hình tốt nhất (tái lập)

```bash
python3 automation/fanout.py <runs>/<run> zc-<run> "run_kaggle.py <train|public_test> OUT --model Qwen/Qwen3-14B-AWQ --dp 4 --cache /tmp/zc_cache --max-len 24576 --chunk 2 --helpers --norm-q --link --compile --compile-scope sales --k 4 --kc 4 --shots shots/train_shots.jsonl --n-shots 2 [--biz '<regex>']" 1 --accounts <acc> --l4 --ks nhatlong1103/zc-assets
```

τ = 0.5 (mặc định). **Ứng viên tốt hơn:** thêm `--think --max-tokens 8192` (e10l), nếu thời gian trên 2×5090 cho phép (bước 4 mục 6).

## 6. Việc tiếp theo (theo thứ tự)

1. **Khởi động:**
   - xin token, kiểm bằng `kaggle competitions list`;
   - tải dữ liệu, chạy 2 test không GPU;
   - kéo kết quả `p1`, `e10`, `e10b`, `e10l`, `e3h-4`, `e3b-2` (`pull_kernels.py`).
2. **File nộp public:**
   - khi `p1` xong: kiểm đủ 1000 dòng, đúng định dạng;
   - chép `submission.jsonl` vào `results/p1_submission.jsonl`;
   - báo user kèm ước lượng PA train (+0.44 holdout); user quyết định nộp.
3. **Đo các sửa chưa đo:** chạy cấu hình mục 5 (code mới nhất + shots v4) trên 4 DN của e7 (114, 116, 117, 125) và 4 DN của
   e6a (105, 108, 115, 118); so cặp bằng `src/compare.py` với log trong `results/chat2_runs/`.
4. **Thinking:** từ e10 / e10b / e10l, so think + shots với e7 / e8 trên cùng câu. Nếu lợi > ~0.05 PA:
   - đo thời gian;
   - cân nhắc chỉ bật think cho câu đồng thuận thấp (leo thang: chạy lần 2 với `--think` cho câu conf < 0.75).
5. **Giảm lỗi còn lại** (PDF 1 §6):
   - unanswerable;
   - `qty_base` / xếp hạng;
   - tuần / ca;
   - câu kho đơn vị lớn.

   Mỗi sửa: tìm trong log → sửa ở `semantic.py` / thông báo lỗi trước, prompt sau.
6. **Mở rộng shots:** sau mỗi run train, `build_shots.py` gộp mọi log (gồm `results/chat2_runs`) → commit `src/shots/`.
7. **Chuẩn bị nộp cuối:**
   - Qwen3-14B-FP8 (5090, Blackwell): chạy thử TP=2 trên L4 hoặc mượn GPU lớn;
   - kiểm tất định: chạy 2 lần cùng cấu hình, |ΔPA| ≤ 0.03;
   - ước lượng 6 giờ (`--budget`);
   - Docker build theo `docker/SETUP_DOCKER.md` (user build trên máy mình).

**Quota:** phần lớn acc hết quota tuần (PDF 2 §2.1). Quota reset theo tuần; probe lại bằng kernel nhỏ trước khi lên lịch.
