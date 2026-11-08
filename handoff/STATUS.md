# STATUS — trạng thái hiện tại (đọc đầu tiên khi tiếp tục)

*Cập nhật 2026-11-08. Bàn giao: `handoff/03_HANDOFF.pdf` (việc tiếp theo), `01_SCIENTIFIC_FINDINGS.pdf`,
`02_ENGINEERING_OPS_LESSONS.pdf`, `environment_summary.pdf`. Số liệu: `EXPERIMENT_LOG.md` entry 0–27.*

## Quyết định mới nhất (user, 08/11)

**Hybrid framework**: mọi câu đi qua mô hình (hiểu câu → sinh code → chạy → kiểm tra → trả lời / abstain). Cấm luật
sinh thẳng code / đáp án không qua mô hình; **được** dùng luật tổng quát để nhận diện ý định, đọc data, dựng schema
(`automap`), làm sạch câu, toolkit, few-shot, kiểm tra — làm ngữ cảnh / gợi ý cho Agent. Chấp nhận public giảm
(~0.92–0.95) để tổng quát hơn. Chi tiết: `handoff/03_HANDOFF.md` §2.1, §5; thiết kế: `docs/06_HYBRID_FRAMEWORK.md`; **kế hoạch + đặc tả triển khai: `docs/08_MASTER_PLAN.md`** (lý do: `docs/09_OPTIONS_DEEP_DIVE.md`).

## Hiện có

| Thành phần | Trạng thái |
|---|---|
| Luồng nộp hiện tại `src/predict_flow.py` | rule-first (luật → stage 1.5 → LLM) — **sẽ thay** |
| Đường LLM `src/agent.py` + `pipeline.py` + `consensus.py` | v28: public 0.921, train 0.9897 |
| Toolkit `src/dm.py`, `src/industry.py`; kiểm tra `src/canon.py` | giữ (phần "tính toán" + "kiểm tra" của khung mới) |
| Luật `src/rules_first.py`, `qparse.py`, `automap.py`, `qnorm.py` | train 3000 / 3000, public 1000 / 1000 trong mẫu; paraphrase held-out yếu → không trả lời thẳng nữa; `automap`, `qnorm`, phần nhận diện ý định của `qparse` giữ làm ngữ cảnh / gợi ý cho Agent |

| Đo | Kết quả |
|---|---|
| Public gốc | v30 (luật) 1000 / 1000; v28 (LLM) 0.921 |
| Paraphrase public held-out (k=1,2), toàn luồng | v34 0.602 (v32 0.584) |
| Paraphrase train held-out | ~0.801 (ước lượng, entry 26) |
| Thời gian | luật ~0.14 s / câu; LLM ~9 s / câu / lần chạy (4×L4); v34 2065 câu: 3.1 h |

File nộp: `submissions/v30/` (luật), `submissions/v28/` (LLM). Không có routine tự chạy.

## Kaggle

L4: quynhtran35, phatle1578 (chưa dùng tuần W46), yuan1103 (~15 h đã dùng). T4: yeshenhi, mxuanquy. Token phải xin lại.
**Cần thêm 4 acc L4 mới (lý tưởng 5) trước sáng 11/11** — kế hoạch ≈ 36.3 giờ phiên 4×L4 tuần W46 (≈ 189 GPU-h kể cả 30 % dự
phòng) so với 75.6 GPU-h còn lại; bảng việc / phân bổ theo đợt: `docs/08_MASTER_PLAN.md` §9.

## Việc tiếp theo (user đã duyệt 08/11): triển khai code đúng `docs/08_MASTER_PLAN.md` §8 (WP0 → WP12), đặc tả §5, quy tắc §10

1. WP0 chuẩn bị → WP1 chạy thử GPU (yuan1103) → QĐ1–QĐ3, QĐ12.
2. WP2 (P0) luồng hybrid v1: `api_spec`, `grammar` (xgrammar), `hints`, `tools` (find / period / check / help), `precheck`,
   `prompting`, `predict_flow` mới; test + preflight. Song song WP4 nhà máy dữ liệu.
3. WP3 E0 + chọn mô hình chính + A/B → WP5 dò thầy → WP6 nhánh fine-tune → WP7 huấn luyện đầy đủ; song song WP8 (P1: data
   card, đọc kết quả, bộ kiểm 0.8B, truy hồi embedding 0.6B, tin cậy), WP12 (DuckDB), WP9 sửa lỗi tới public = 1.0.
4. WP10 đánh giá cuối (14/11) → WP11 Docker; **user** build ngày 15/11 và nộp trước **16/11/2026 23:59 GMT+7**.
