# PDF 3 — Handoff (DataMate, 14/11/2026 15:45 UTC)

*Mục tiêu, tiến độ, quyết định, vị trí, việc tiếp theo. Kết quả khoa học và bằng chứng: PDF 1. Kinh nghiệm kỹ thuật, lỗi và cách tránh: PDF 2.
Môi trường: `environment_summary.pdf`. Trạng thái sống: `handoff/STATUS.md`.*

## 1. Mục tiêu

Zalo AI Challenge 2026, track DataMate: trả lời câu hỏi tiếng Việt trên dữ liệu bảng bẩn của cửa hàng nhỏ. Điểm PA = (đúng − sai) / N. Mô hình ≤ 15B
tham số, offline, ≤ 6 h trên 2×RTX 5090, nộp Docker. Mục tiêu hiện tại của user: **public càng gần 1.0 càng tốt** bằng hệ **không luật viết tay**
lúc suy luận. Hạn: đóng băng container **16/11 04:00 UTC**, nộp 16/11 23:59 GMT+7.

## 2. Quyết định còn hiệu lực

| Ngày | Quyết định |
|---|---|
| 08/11 | Cấm hard code đáp án; không train trên public / private; phân tích lỗi chỉ trên public gốc / train |
| 13/11 | Ngừng luật viết tay; luật nếu có phải do mô hình viết; bản nộp = hệ không luật viết tay |
| 14/11 02:30 | Chỉ tối ưu public (không held-out); GPU: 2 phiên × 4 L4 / acc, FSDP 4 GPU hoặc mỗi GPU một biến thể; không tiết kiệm acc |
| 14/11 06:30 | Kế hoạch 5: S đóng băng (v3a × 0.70) + LoRA-W huấn luyện riêng (W1 biên dịch rulebook, W2 nối luật, W3 phán xử); bộ khớp thuật ngữ của P **chỉ** để gán nhãn câu train; image = 1 mô hình gốc + adapter LoRA (gộp theo pha); dùng acc hợp lý |
| 14/11 15:30 | Kế hoạch 5 xong → **DỪNG**; user lập chiến lược mới ở khung chat Fable, gửi kế hoạch dạng **docx** để Agent chạy |

## 3. Tiến độ

| Hệ | Public | Ở đâu |
|---|---|---|
| P (luật viết tay; chỉ dự phòng) | 0.994 | `final_submission/submission.jsonl`, `configs/flow_p_final.json` |
| Kế hoạch 4 (S + W unit) | 0.938 / container 0.922 | `configs/flow_final.json` (Docker hiện build cấu hình này) |
| Kế hoạch 5 container thật (phán xử WJB1) | 0.943 / 0.941 | `final_submission/swv/`, `Process/runs/f5a`, `f5b` |
| **Cấu hình cuối kế hoạch 5** (phán xử WJR1, tổ hợp 3 biến thể) | **TB 0.952** (offline trên 4 log S) | `configs/flow_p5_final.json` — **chưa** chạy container, **chưa** có trong Docker |

## 4. Vị trí

| Thứ | Ở đâu |
|---|---|
| Luồng suy luận | `src/predict_flow.py` (pha: `prepare_lrl` W1 bind → `prepare_units` → `prepare_link` W2 → agent S → `judge_phase` W3; `w_model` gộp adapter W theo pha) |
| Thành phần W mới | `src/link.py` (W2), `src/judge.py` (W3: ứng viên, prompt, phân tích, chính sách, `ensemble`), `src/lrl_gen.py --mode bind` (W1) |
| Dữ liệu W | `training/data/w_judge.py`, `w_link.py`, `p_rows.py`; bản đã dựng: `Process/train_data/p5/` |
| Job / kế hoạch Kaggle | `automation/jobs/*.jobs`, `Process/kaggle/plans/plan_*.json`, sổ kernel `Process/kaggle/kernels.csv` |
| Đánh giá offline | `tests/replay_judge.py`, `Process/tools/plan5/{pol,ens2,err}.py`; phiếu: `Process/runs/p5_judge/` |
| Điểm | `Process/scores_p5.csv` (kế hoạch 5), `scores_p3.csv`, `scores.csv` |
| Adapter | S: `<acc>/zc-v3a-adapter`, `nhatlong1103/zc-final-adapter`; W (WL1, WJR1, WJB1): **`Process/adapters/` (Git LFS)** + Kaggle `nhatlong1103/zc-w-final` |
| Nhật ký chi tiết | `docs/18_PLAN5_TWO_LORA.md` §7 |

## 5. Việc tiếp theo

1. **Chờ kế hoạch mới của user (docx).** Đọc `READING_GUIDE.md`, `handoff/STATUS.md`, PDF 1 §8, PDF 2 §9 trước khi chạy.
2. **Bắt buộc trước 16/11 04:00 UTC (nếu kế hoạch mới không thay):** D1 Docker cho `configs/flow_p5_final.json`:
   - tải `nhatlong1103/zc-w-final` vào `models/` (`docker/fetch_assets.sh`), image chép `training/merge.py` + `WL1/`, `WJR1/` (adapter);
   - `predict.sh` dùng `CONFIG=/app/configs/flow_p5_final.json`; adapter tìm theo tên (`ZC_ADAPTER_ROOT` hoặc `/models`);
   - chạy `docker/run_public_test.sh` trên máy có GPU (hoặc một phiên Kaggle `soup_job --runner predict` với `--ds nhatlong1103/zc-w-final`) để có
     điểm container của đúng cấu hình WJR1; thời gian 1000 câu trên 4×L4 hiện ~60–75 phút.
3. Token Kaggle: xin lại user (mất khi đổi container); quota làm mới 21/11 00:00 UTC; nhiều acc còn ~14–25 h.
