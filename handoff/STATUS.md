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

## Chế độ chạy tự động (user, 08/11)

Chạy tự động đến khi xong toàn bộ `docs/08_MASTER_PLAN.md`; không dừng, không hỏi. Hết hạn mức Claude (phiên 5 giờ) → tạm dừng,
reset thì tự chạy tiếp (routine mỗi giờ tạo đầu khung chat mới). Tiếp tục từ bước chưa xong đầu tiên theo bảng dưới. Hết quota GPU
Kaggle → làm phần code / CPU, chạy GPU tiếp khi reset.

## Tiến độ WP (cập nhật sau mỗi bước — nguồn duy nhất để biết làm tiếp từ đâu)

| WP | Trạng thái | Kết quả / commit / kernel |
|---|---|---|
| WP0 chuẩn bị | **xong** 08/11 | token 9 acc (8 có 4×L4: nhatlong1103, quynhle07, quynhtran35, nguyenhoa99, yuan1103, synchan5704, anderson1202, loinguyentran; yeshenhi chỉ T4) — probe `zc-probe`; `score_1.jsonl` của user → `data/raw/public_gt.jsonl` (v30 1000/1000, v28 921 đúng / PA 0.852); cache ingest; `tests/splits/` (pub_full 1000, pub_k0 692, pub_ho 1373, tr_ho 1826, quick400 400, pub_noind 112, pub_noacc 1000); `eval_all.sh quick` train 3000/3000, public 1000/1000; wheels huấn luyện = output kernel public `yeshenhi/zc-train-wheels` (`wheels_train/`: transformers 5.17.0, trl 1.15.0; ảnh GPU Kaggle + vLLM 0.32 đã có torch 2.13.0, peft 0.21.2, accelerate 1.15.0, datasets 4.8.5, xgrammar 0.2.8, duckdb 1.3.2, python 3.13; `wheels_unsloth/` tuỳ chọn) |
| WP1 chạy thử GPU (G1) | **đang** | vòng 1 `yuan1103/zc-smoke35-1`: Unsloth 1 GPU mb 4 đỉnh 18.84 GB, 844 tok/s/L4 (QĐ1 → 4 nhánh/phiên); HF/FSDP hỏng vì torchao 0.10 (đã sửa); vòng 2 `zc-smoke35b-1` + suy luận `zc-smoke35i-1` (ngữ pháp, thầy) đang chạy — kết quả: `<scratch>/runs/smoke35*/…_out/results/smoke*/results.json` |
| WP2 luồng hybrid v1 | **đang** | module + test CPU xong (commit 804a43c và sau): api_spec, grammar, hints, prompting, retrieval, shopctx, tools, precheck, hybrid (`agent.py --flow hybrid`), predict_flow mới, eval_flow, score_sets; preflight 7 job WP3 đang chạy |
| WP3 E0 / S0 / A/B (G2–G5) | chưa | |
| WP4 nhà máy dữ liệu (+ G6, G7) | chưa | |
| WP5 D3 theo thầy (G8–G11) | chưa | |
| WP6 nhánh fine-tune (G12–G19) | chưa | |
| WP7 huấn luyện đầy đủ (G20–G21) | chưa | |
| WP8 tin cậy / bộ kiểm / embedding (G22–G24) | chưa | |
| WP9 sửa lỗi tới public = 1.0 (G25) | chưa | |
| WP10 đánh giá cuối (G27–G29) | chưa | |
| WP11 Docker + nộp | chưa | |
| WP12 DuckDB | chưa | |

## Kaggle

08/11: user cấp 9 token (`~/.kaggle/tok_<acc>`); probe 4×L4 OK trên 8 acc: nhatlong1103, quynhle07, quynhtran35, nguyenhoa99,
yuan1103 (~15 h W46 đã dùng), synchan5704, anderson1202, loinguyentran (quota W46 của các acc mới chưa rõ: acc dùng chung dự án
khác). yeshenhi: chỉ T4 (dùng cho kernel wheels / T4). Đủ 8 acc L4 mà kế hoạch §9 cần (7, lý tưởng 8).

## Việc tiếp theo (user đã duyệt 08/11): triển khai code đúng `docs/08_MASTER_PLAN.md` §8 (WP0 → WP12), đặc tả §5, quy tắc §10

1. WP0 chuẩn bị → WP1 chạy thử GPU (yuan1103) → QĐ1–QĐ3, QĐ12.
2. WP2 (P0) luồng hybrid v1: `api_spec`, `grammar` (xgrammar), `hints`, `tools` (find / period / check / help), `precheck`,
   `prompting`, `predict_flow` mới; test + preflight. Song song WP4 nhà máy dữ liệu.
3. WP3 E0 + chọn mô hình chính + A/B → WP5 dò thầy → WP6 nhánh fine-tune → WP7 huấn luyện đầy đủ; song song WP8 (P1: data
   card, đọc kết quả, bộ kiểm 0.8B, truy hồi embedding 0.6B, tin cậy), WP12 (DuckDB), WP9 sửa lỗi tới public = 1.0.
4. WP10 đánh giá cuối (14/11) → WP11 Docker; **user** build ngày 15/11 và nộp trước **16/11/2026 23:59 GMT+7**.
