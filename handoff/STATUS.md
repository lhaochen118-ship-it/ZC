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
| WP1 chạy thử GPU (G1) | **xong** 08/11 | QĐ1 Unsloth 1 GPU (mb 4, 18.84 GB, ~0.8k tok/s) → 4 nhánh/phiên; QĐ2 thầy Qwen3.6-27B-FP8 / Gemma 4 31B fp8 chạy TP2; QĐ3 gộp + FP8 (LoRA chậm 18 %); QĐ12 tắt ngữ pháp (chậm 32×); QĐ16 nhánh 4M token; FSDP4 mb 8 2,023 tok/s cho bản cuối — entry 34, 36 |
| WP2 luồng hybrid v1 | **xong** 08/11 | module + test CPU; preflight OK cho 7 job WP3 (code chốt `bc7c1a3`, worktree `<scratch>/wt_v35`) |
| WP3 E0 / S0 / A/B (G2–G5) | **xong** 08/11 (QĐ4 chờ WP6) | E0 quick400 0.720 / pub_full 0.864; full 0.755; Gemma 4 12B compact 0.7675; think 0.765; pub_noacc 0.824; pub_noind −0.06; QĐ5 full cho mô hình gốc, QĐ11 n-gram, QĐ12 tắt ngữ pháp — entry 37. Mốc gốc cùng code arms (`quynhtran35/zc-evalbase-1`): quick400 0.700, **pub_ho 0.516** |
| WP4 nhà máy dữ liệu (+ G6, G7) | **xong** (DV chưa: cắt được) | D1 2119, D2 3816, D2N 5418, D4T 2000; bản ghi D3 (`training/data/d3_records.py`, lọc chương trình suy biến + câu rác + ≤ 3 bản / câu): self 16,688, qwen35a3b 18,146, gptoss 10,079, gemma31 17,043, qwen27 (đang dựng); thư mục nhánh `training/data/out/r2/` (symlink); `training/data/mix.py` trộn cố định |
| WP5 D3 theo thầy (G8–G11) | **xong** 08/11 | giữ sau vòng lặp ngược: self 51.8 %, qwen35a3b 64.4 %, gptoss 31.9 % (nhiều câu rác), gemma31 69.5 %, **qwen27 73.6 %** |
| WP6 nhánh fine-tune (G12–G19) | **đang** | vòng 1 đã huấn luyện (`synchan5704/zc-armsr1a-1`, adapter → dataset `nguyenhoa99/zc-adapters-r1`); đánh giá `nguyenhoa99/zc-evalr1b-1` (LoRA, quick400 + pub_ho) đang chạy → QĐ7 → vòng 2 (thắng + D3 từng thầy, 50 % bản ghi) từ worktree `<scratch>/wt_r2` (`dbec142`, cùng code đánh giá vòng 1) |
| WP7 huấn luyện đầy đủ (G20–G21) | chưa | |
| WP8 tin cậy / bộ kiểm / embedding (G22–G24) | **đang** | `tests/calibrate.py` (QĐ14): log E0 dev PA trả lời hết 0.825, vote ≥ 0.5 0.814, calibrator CV 0.839 → hiệu chỉnh lại trên log mô hình cuối; embedding: QĐ11 không dùng |
| WP9 sửa lỗi tới public = 1.0 (G25) | **đang** | vòng 1 (log E0 dev): gợi ý từ đơn vị (`base_unit`), `debt_aging` nhận mốc phẳng; `tests/test_wp9_fixes.py` — dữ liệu SFT v2 (gợi ý mới) dựng lại trước WP7 |
| WP10 đánh giá cuối (G27–G29) | chưa | |
| WP11 Docker + nộp | **đang** | `docker/predict.sh`, `Dockerfile` đã chuyển sang luồng hybrid (`configs/flow_final.json` chưa có) |
| WP12 DuckDB | **đang** | `src/sqlesc.py` + nối luồng (`--escape python,sql`), `test_sqlesc`, ca `<sql>` trong `test_flow_hybrid` PASS; còn đo A/B `pub_noind` (QĐ12) |

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
