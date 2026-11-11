# PDF 3 — Handoff (khung chat kế tiếp, từ 11/11/2026)

*Phạm vi: mục tiêu, tiến độ, quyết định đã thống nhất, vị trí mã nguồn / dữ liệu / kết quả, các bước tiếp theo. Kết quả khoa học:
PDF 1; kinh nghiệm kỹ thuật / vận hành: PDF 2; cấu hình máy: `environment_summary.pdf`; trạng thái sống: `handoff/STATUS.md`.*

## 1. Mục tiêu

- Zalo AI Challenge 2026, track **DataMate**: hỏi đáp tiếng Việt trên dữ liệu bảng bẩn của hộ kinh doanh; **PA = (đúng − sai) / N**.
- Ràng buộc BTC: tổng tham số ≤ 15B, offline, ≤ 6 h trên 2×RTX 5090, Docker. **Hạn nộp 16/11/2026 23:59 GMT+7**; user build Docker
  ngày 15/11 và tự nộp portal.
- Mục tiêu user: PA cao nhất trên private test, nhanh nhất có thể. **Điều kiện dừng nghiên cứu (`docs/13_TASK_AFTER_PLAN1.md`):
  public gốc 100 % bằng cải tiến tổng quát mà held-out không giảm.**

## 2. Quyết định đã thống nhất (không hỏi lại; bản đầy đủ: `CLAUDE.md`)

| Ngày | Quyết định |
|---|---|
| 08/11 | Hybrid framework: mô hình viết chương trình trả lời mọi câu; luật chỉ chuẩn bị ngữ cảnh (gợi ý, làm sạch, schema, toolkit, ví dụ) và kiểm tra. Cấm luật sinh đáp án |
| 08/11 | Chạy tự động đến khi xong, không hỏi; routine mỗi giờ đánh thức phiên; chỉ xin user token / acc / accept rules / nộp portal |
| 08/11 | Dataset ngoài uy tín được dùng; không bao giờ train trên public / private (kể cả paraphrase câu public) |
| 09/11 | Sau kế hoạch 1: vòng phân tích lỗi → tài liệu → kế hoạch 2 → triển khai, lặp đến public 100 %, báo cả public và held-out |
| 10/11 | Hai hướng song song: R (luật tổng quát) và M (cải tiến mô hình từ M0, không thêm luật) |
| 10/11 | `final_submission/submission.jsonl` luôn là file public PA cao nhất; cập nhật tự động sau mỗi lần chạy |
| 10/11 | Acc < 6 h GPU: kiểm đủ giờ trước khi giao task; phiên Kaggle ≤ 12 h (chia nhỏ / checkpoint) |
| 11/11 | Chạy nhanh nhất mà không ảnh hưởng nhiều độ chính xác (sàng lọc tập con song song → xác nhận đủ tập) |
| bất biến | Không vá theo DN / câu; đáp án public (`data/raw/public_gt.jsonl`) chỉ để chấm; không xem từng câu held-out; không push kernel thiếu kiểm tra (CPU suite / preflight; job soup / merge-eval: dry-run + review + `--no-preflight`) |

## 3. Tiến độ

| Hạng mục | Trạng thái |
|---|---|
| Kế hoạch 1 (`docs/08`, WP0–WP12) | xong |
| Kế hoạch 2 (`docs/12_PLAN2.md`) hướng R | 16 vòng; luật tổng quát (gợi ý, làm sạch, kiểm tra size / unit / len_mask, cờ) — gần bão hoà |
| Kế hoạch 2 hướng M (`docs/12` §6) | M7, M6, M2, M4 xong (PDF 1 §3.2); M1 / M3 / M5 / M8 / M9 chưa làm |
| Mô hình | WF1–4 (súp) → WF3 → **WG1** (D1 + toàn bộ D2) là mô hình nộp; WG5–8 đang huấn luyện (`bichthuy93/zc-wg2r-1`) |
| Bản nộp | O + WG1 × 0.35: public 0.9855 (2 seed), container 0.986, held-out 0.778; `final_submission` 0.991 |
| Docker | sẵn sàng (`docker/`), dataset adapter `nhatlong1103/zc-final-adapter` = WG1 + `soup.json`; user build 15/11 |

## 4. Vị trí

| Thứ | Ở đâu |
|---|---|
| Repo / branch | `lhaochen118-ship-it/ZC`, `claude/datamate-v3-0dny90` |
| Hướng dẫn đọc repo | `READING_GUIDE.md` |
| Trạng thái sống | `handoff/STATUS.md` (lịch sử: `Process/history/`) |
| Luồng nộp | `src/predict_flow.py` → `src/agent.py --flow hybrid` (`src/hybrid.py`, `pipeline.py`, `hints.py`, `qnorm.py`, `precheck.py`, `tools.py`, `dm.py`, `industry.py`); cấu hình `configs/flow_final.json` |
| Mô hình nộp | `soup.json` (WG1 × 0.35) trong Kaggle dataset `nhatlong1103/zc-final-adapter`; sha256: `src/adapter_hashes.json`; đổi: `automation/set_final_model.py` |
| Adapter khác | `<acc>/zc-wg-adapters` (WG1–4), `<acc>/zc-final-adapter` (WF1–4 hoặc WG1), output kernel `loinguyentran/zc-wg-1`, `thnliu/zc-m2w3-1`, `anderson1202/zc-m2g1-1`; danh sách: `Process/kaggle/datasets_by_account.txt` |
| Dữ liệu BTC | `kaggle datasets download nhatlong1103/zc-2026 -p data/raw --unzip`; đáp án public `data/raw/public_gt.jsonl` (user gửi `score_1.jsonl` nếu mất) |
| Paraphrase (đánh giá) | `<acc>/zc-para` (`para_pub.jsonl`, `para_train.jsonl`); tập: `tests/splits/*.txt` (`pub_full`, `pub_ho`, `tr_ho`, `pub_ho400`, `tr_ho400`, `tr_dev_*`, `d3sel`) |
| Dữ liệu huấn luyện | `Process/train_data/` (nén: `wg/` = D1 + D2 + D2N + RFT khó, `m2/`, `m4/`, `w7f_*`); giải nén vào `training/data/out/<tên>/` |
| Log + kết quả lần chạy | `Process/runs/<run>/<result>/` (`logs.jsonl.gz`, `submission.jsonl.gz`, `result.json`); bảng điểm: `Process/scores.csv` |
| Kernel / plan | `Process/kaggle/kernels.csv` (run → kernel id → trạng thái), `Process/kaggle/plans/*.json`, job: `automation/jobs/` |
| Code M0 đóng băng (hướng M) | commit `b093d46` + `Process/m0_worktree/` (bản vá + file thêm) |
| Bộ chọn | `src/refs/selector/selector_m0_inv.json` (M0) |

## 5. Các bước tiếp theo

1. **Tạo routine mỗi giờ** cho phiên mới (`create_trigger`, lời nhắc như `CLAUDE.md`) và xin user token Kaggle nếu container mới.
2. **Gắn lại / tải kết quả WG5–8** (`bichthuy93/zc-wg2r-1`, xong ~14:30 UTC 11/11): `kaggle kernels output` → dataset riêng →
   sàng lọc → xác nhận → đổi mô hình nộp nếu đạt §4 (chi tiết `handoff/STATUS.md` §4).
3. **Quota mới 14/11 00:00 UTC**: đợt lớn cuối (thêm nhánh WG1-style / seed, đánh giá nhiều seed của ứng viên tốt nhất); ý tưởng chưa
   thử ở STATUS §4.3.
4. **Chốt trước 15/11**: đường container với bản nộp cuối, `docker/README_SOLUTION.md`, `final_submission/`, báo user build và
   nộp. Không đổi bản nộp sau khi user bắt đầu build trừ khi user đồng ý.
5. Sau mỗi bước: cập nhật `handoff/STATUS.md`, commit + push.

## 6. Rủi ro cần nhớ

- Nhiễu đánh giá lớn (public ±0.4 một lần chạy): không đổi bản nộp theo một seed.
- Token và scratch mất khi đổi container; dữ liệu `data/raw`, `training/data/out` không nằm trong git (lấy lại theo §4).
- Quota tuần này còn ít (nhiều acc < 6 h) đến 14/11; acc dùng chung dự án khác: chỉ đụng kernel `zc-*`.
- Thời gian trên 2×RTX 5090 chưa đo trực tiếp (4×L4: ~22 phút / 1,000 câu, ngân sách luồng nộp 20,400 s).
