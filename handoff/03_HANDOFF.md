# PDF 3 — Handoff (DataMate, 16/11/2026 10:15 UTC)

*Mục tiêu, tiến độ, quyết định, vị trí, việc tiếp theo. Kết quả khoa học + bằng chứng: PDF 1 (§10 = P5). Lỗi / kỹ thuật / vận hành: PDF 2
(§11 = P5). Trạng thái sống: `handoff/STATUS.md`. Môi trường: `environment_summary.pdf`.*

## 1. Mục tiêu nghiên cứu

Zalo AI Challenge 2026, track DataMate: hỏi đáp tiếng Việt trên dữ liệu bảng bẩn của cửa hàng nhỏ; PA = (đúng − sai) / N. Ràng buộc: tổng
tham số ≤ 15B (tính mọi expert), offline, ≤ 6 h trên 2 × RTX 5090, nộp Docker; nộp **16/11 23:59 GMT+7**. Mục tiêu: điểm cao nhất bằng hệ
**chế độ H** (không hàm ngành viết tay), mọi cải tiến là năng lực mô hình tổng quát, không luật nghiệp vụ viết tay.

## 2. Quyết định đã thống nhất (còn hiệu lực, mới nhất trước)

| Ngày (UTC) | Quyết định |
|---|---|
| 16/11 09:00 | **Bản nộp = framework mạnh nhất `flow_p5_final` (0.952) + `submission.jsonl` của nó** (thay quyết định chế độ H 14/11) |
| 16/11 05:30 | **Thí nghiệm chỉ thay đúng thành phần nghiên cứu**; baseline fine-tune = `flow_p5_final` (0.952), chế độ H = `flow_p5_final_h`; baseline zero-shot = cùng framework với Qwen3.5-9B gốc; nhiều thay đổi ⇒ ablation riêng |
| 16/11 02:15 | Còn ~12 h: chạy nhanh nhất, không tiết kiệm acc / GPU; chạy mọi biến thể P5 còn trong ngưỡng thời gian |
| 15/11 13:20 | Chỉ tối ưu trên public test, không chia tập khác |
| 14/11 17:00 | **Bản nộp chế độ H**; cấm hard-code / rule-based / heuristic thủ công; sửa nguyên nhân gốc |
| 08/11 | Cấm hard code đáp án; không train trên public / private; đáp án public chỉ để chấm |

## 3. Tiến độ

- **Cấu hình nộp: `configs/flow_p5_final.json` (framework 0.952)**; file nộp public `Docker_submission/submission.jsonl` (D1, PA 0.952;
  chạy lại bằng mã hiện tại C0L 0.942). Docker đóng gói trong `Docker_submission/` (mặc định cấu hình này; chưa build — cần máy có
  Docker, hướng dẫn `Docker_submission/README.md`). Chế độ H (`flow_p5_final_h`, 0.860) chỉ còn là tham chiếu.
- Bản P (0.994, luật kiểm tra viết tay) đóng gói riêng `0.994_Docker_submission/` (+ `submission.jsonl` 0.994); kiểm chứng chạy lại bằng mã
  hiện tại: lượt `pver` (duckzuykobip) — **chưa có kết quả** lúc bàn giao.
- P5 (zero-shot, MoE, ghép component): không tổ hợp nào vượt 0.952 / 0.863; chi tiết PDF 1 §10, bảng `docs/21_COMBO_PLAN.md`.
- **Đang chạy lúc bàn giao** (tự động; `Process/runs/p5/after_*.log`, `watch.out`; sau khởi động lại worker: `python3 automation/resume.py
  Process/runs/p5` rồi chạy lại đúng các lệnh `after_*.sh` trong `Process/kaggle/p5_chains.txt` — các script bỏ qua bước đã khởi chạy):

| Việc | Acc (2 nửa cửa hàng) | Ước xong (UTC) | Sau khi xong |
|---|---|---|---|
| A: MoE, solver MS2 × **0.5** (F_H) | bichthuy93 + yuan1103 → `e8rw5p0/1` | ~11:00–11:30 | ghép 2 `submission.jsonl` → `jscore.py`, so e8rw7 0.836 |
| B: MoE, solver MS2 × **0.35** (F_H) | thnliu + huuhien95 → `e8rw35p0/1` | ~11:00–11:30 | như trên |
| MS3 (LoRA cả 88 expert) | loinguyentran + namhunhh → `e8rm3p0/1` | ~11:30–12:00 | như trên |
| MS2C (MS2 + 7M token, lr 1e-4) | huấn luyện caotienminh1 `ftc2` → đánh giá quynhtran35 + phatle1578 (`e8rc2p0/1`) | ~13:00 | như trên |
| MS3C (MS3 + 5.5M token) | huấn luyện mucnokieu `ftc3` → nguyenhoa99 + huyenco (`e8rc3p0/1`) | ~13:00 | như trên |
| pver (P 0.994 chạy lại, mã hiện tại) | duckzuykobip | ~10:45 | |PA − 0.994| ≤ 0.03 ⇒ `0.994_Docker_submission` tái lập được |

Mọi kết quả MoE chỉ để ghi nhận (findings); **không thay bản nộp 0.952** trừ khi user quyết định khác.

## 4. Vị trí

| Thứ | Ở đâu |
|---|---|
| Luồng nộp + Docker | `src/predict_flow.py`, `configs/flow_p5_final.json`, **`Docker_submission/`** (0.952) và **`0.994_Docker_submission/`** (P), mỗi thư mục có `submission.jsonl` (Dockerfile, predict.sh, fetch / build / run / export, README) |
| Kế hoạch / nhật ký P5 | `docs/19_PLAN_P5_ZS.md` (nhật ký), `docs/20_NEW7_PLAN_UPDATED.md`, `docs/21_COMBO_PLAN.md` (tổ hợp) |
| Config thí nghiệm P5 | `configs/flow_p5_zs_*`, `flow_c*`, `flow_e*`, `flow_h_*`, `flow_t02*`, `flow_rs_*`; MoE keep-list `configs/moe/` |
| Job / plan / sổ chạy | `automation/jobs/p5/`, `automation/plans_p5/`, `Process/runs/p5/` (kernel), `Process/runs/p5_results/` (kết quả gzip) |
| Điểm | `Process/scores_p5zs.csv` (`Process/tools/p5/p5_report.py`), ghép cặp `Process/tools/p5/jscore.py` |
| Mã MoE / LoRA | `training/moe_saliency.py`, `src/stage_hf.py`, `training/moe_ft_job.py`, `training/sft.py` (`ZC_LORA_EXPERTS`), `training/merge_fp8.py` |
| Adapter | 9B: `nhatlong1103/zc-final-adapter` (V3A × 0.70), `nhatlong1103/zc-w-v2` (WL1, WJR1), `<acc>/zc-ad3`; MoE: `training/data/out/p5ad/` (MS1–MS3, ML1, MJ1–MJ2; không commit), bản gộp FP8 là output kernel riêng tư (`<acc>/zc-mg*`, `zc-mc*`, `zc-m3*`) |
| Tài liệu bàn giao | `handoff/01`–`03` (.md + .pdf), `handoff/STATUS.md` |

## 5. Việc tiếp theo (theo thứ tự)

1. **Build + chạy thử Docker** trên máy có GPU (`Docker_submission/README.md` mục 3–4): điểm public phải ~0.94–0.95 (± 0.03); kiểm `run.log`
   (`units: rc`, `links: rc`, `judge:`), thời gian.
2. Thu kết quả §3: mỗi biến thể `cat Process/runs/p5/<run>p0/*_out/results/*/*/submission.jsonl ...p1/... > /tmp/x/flow/submission.jsonl`
   → `python3 Process/tools/p5/jscore.py /tmp/x`; ghi `docs/19` + `docs/21` + PDF 1 §10. pver: chấm như trên, ghi vào `0.994_Docker_submission/README.md`.
3. Build / chạy thử `0.994_Docker_submission/` nếu user muốn nộp P (user chọn 1 trong 2 image). Nộp `Docker_submission/submission.jsonl` trên portal (nếu cần); xuất image (`export_image.sh`), upload, nộp form trước 16/11 23:59 GMT+7.
4. Sau nộp: xoá routine `trig_013dELF9kwndZnzmL5KAZ6GS` (và `trig_015hi1s1WHshqorFZC7DWY9j`); hướng mở cho vòng sau: PDF 1 §10.3.

**Token Kaggle** chỉ ở `~/.kaggle/tok_<acc>` (mất khi đổi container, xin lại user); quota làm mới 21/11 00:00 UTC.
