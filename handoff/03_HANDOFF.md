# PDF 3 — Handoff (DataMate, 16/11/2026 09:00 UTC)

*Mục tiêu, tiến độ, quyết định, vị trí, việc tiếp theo. Kết quả khoa học + bằng chứng: PDF 1 (§10 = P5). Lỗi / kỹ thuật / vận hành: PDF 2
(§11 = P5). Trạng thái sống: `handoff/STATUS.md`. Môi trường: `environment_summary.pdf`.*

## 1. Mục tiêu nghiên cứu

Zalo AI Challenge 2026, track DataMate: hỏi đáp tiếng Việt trên dữ liệu bảng bẩn của cửa hàng nhỏ; PA = (đúng − sai) / N. Ràng buộc: tổng
tham số ≤ 15B (tính mọi expert), offline, ≤ 6 h trên 2 × RTX 5090, nộp Docker; nộp **16/11 23:59 GMT+7**. Mục tiêu: điểm cao nhất bằng hệ
**chế độ H** (không hàm ngành viết tay), mọi cải tiến là năng lực mô hình tổng quát, không luật nghiệp vụ viết tay.

## 2. Quyết định đã thống nhất (còn hiệu lực, mới nhất trước)

| Ngày (UTC) | Quyết định |
|---|---|
| 16/11 05:30 | **Thí nghiệm chỉ thay đúng thành phần nghiên cứu**; baseline fine-tune = `flow_p5_final` (0.952), chế độ H = `flow_p5_final_h`; baseline zero-shot = cùng framework với Qwen3.5-9B gốc; nhiều thay đổi ⇒ ablation riêng |
| 16/11 02:15 | Còn ~12 h: chạy nhanh nhất, không tiết kiệm acc / GPU; chạy mọi biến thể P5 còn trong ngưỡng thời gian |
| 15/11 13:20 | Chỉ tối ưu trên public test, không chia tập khác |
| 14/11 17:00 | **Bản nộp chế độ H**; cấm hard-code / rule-based / heuristic thủ công; sửa nguyên nhân gốc |
| 08/11 | Cấm hard code đáp án; không train trên public / private; đáp án public chỉ để chấm |

## 3. Tiến độ

- **Cấu hình nộp: `configs/flow_v2_final.json` (= `flow_p5_final_h`)**, public 0.863 / 0.863 / 0.858 / 0.857 (TB 0.860). Docker đóng gói
  xong trong `Docker submission/` (chưa build — cần máy có Docker; hướng dẫn trong `Docker submission/README.md`).
- P5 (zero-shot, MoE, ghép component): không tổ hợp nào vượt 0.952 / 0.863; chi tiết PDF 1 §10, bảng `docs/21_COMBO_PLAN.md`.
- **Đang chạy lúc bàn giao** (tự động, theo dõi `Process/runs/p5/watch.out`):

| Việc | Kernel / script | Dự kiến (UTC) | Quyết định khi xong |
|---|---|---|---|
| MS3 (MoE E88 + LoRA cả 88 expert) trên F_H, 2 nửa cửa hàng | `after_ms3b.sh` → `zc-m3*` (gộp) → `zc-e8ms3p0/1` | ~11:30 | ghép 2 file, chấm; chỉ thay bản nộp nếu > 0.863 có ý nghĩa (z ≥ 2 vs H0a, H0a2) **và** cấu hình MoE vừa 6 h |
| e8r (ablation trọng số MS2 × 1.0, F_H) | `caohuyen74/zc-e8r-1` | ~09:30 | chỉ ghi nhận |
| Gộp ML1 (huyenco, `zc-hml-1`) | — | ~09:00 | không dùng nữa (rsl đã huỷ: lai 9B + MoE vượt 15B) |

## 4. Vị trí

| Thứ | Ở đâu |
|---|---|
| Luồng nộp + Docker | `src/predict_flow.py`, `configs/flow_v2_final.json`, **`Docker submission/`** (Dockerfile, predict.sh, fetch / build / run / export, README) |
| Kế hoạch / nhật ký P5 | `docs/19_PLAN_P5_ZS.md` (nhật ký), `docs/20_NEW7_PLAN_UPDATED.md`, `docs/21_COMBO_PLAN.md` (tổ hợp) |
| Config thí nghiệm P5 | `configs/flow_p5_zs_*`, `flow_c*`, `flow_e*`, `flow_h_*`, `flow_t02*`, `flow_rs_*`; MoE keep-list `configs/moe/` |
| Job / plan / sổ chạy | `automation/jobs/p5/`, `automation/plans_p5/`, `Process/runs/p5/` (kernel), `Process/runs/p5_results/` (kết quả gzip) |
| Điểm | `Process/scores_p5zs.csv` (`Process/tools/p5/p5_report.py`), ghép cặp `Process/tools/p5/jscore.py` |
| Mã MoE / LoRA | `training/moe_saliency.py`, `src/stage_hf.py`, `training/moe_ft_job.py`, `training/sft.py` (`ZC_LORA_EXPERTS`), `training/merge_fp8.py` |
| Adapter | 9B: `nhatlong1103/zc-final-adapter` (V3A × 0.70), `nhatlong1103/zc-w-v2` (WL1, WJR1), `<acc>/zc-ad3`; MoE: `training/data/out/p5ad/` (MS1–MS3, ML1, MJ1–MJ2; không commit), bản gộp FP8 là output kernel riêng tư (`<acc>/zc-mg*`, `zc-mc*`, `zc-m3*`) |
| Tài liệu bàn giao | `handoff/01`–`03` (.md + .pdf), `handoff/STATUS.md` |

## 5. Việc tiếp theo (theo thứ tự)

1. **Build + chạy thử Docker** trên máy có GPU (`Docker submission/README.md` mục 3–4): điểm public phải ~0.86 (± 0.03); kiểm `run.log`
   (`units: rc`, `links: rc`, `judge:`), thời gian.
2. Khi MS3 xong: chấm 2 nửa (`cat` 2 `submission.jsonl` → `jscore.py`), ghi `docs/19` + `docs/21`. Chỉ đổi bản nộp theo quy tắc §3.
3. Xuất image (`export_image.sh`), upload, nộp form trước 16/11 23:59 GMT+7.
4. Sau nộp: xoá routine `trig_013dELF9kwndZnzmL5KAZ6GS` (và `trig_015hi1s1WHshqorFZC7DWY9j`); hướng mở cho vòng sau: PDF 1 §10.3.

**Token Kaggle** chỉ ở `~/.kaggle/tok_<acc>` (mất khi đổi container, xin lại user); quota làm mới 21/11 00:00 UTC.
