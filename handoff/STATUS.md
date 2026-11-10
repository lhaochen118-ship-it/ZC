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
| WP3 E0 / S0 / A/B (G2–G5) | **xong** 09/11 | E0 quick400 0.720 / pub_full 0.864; full 0.755; think 0.765; pub_noacc 0.824; pub_noind −0.06; QĐ5 full cho mô hình gốc (pub_ho 0.541 vs compact 0.516), QĐ11 n-gram, QĐ12 tắt ngữ pháp; **QĐ4: giữ Qwen** (Gemma 12B full: quick400 0.7675, pub_ho 0.521) — entry 37, 41 |
| WP4 nhà máy dữ liệu (+ G6, G7) | **xong** (DV chưa: cắt được) | D1 2119, D2 3816, D2N 5418, D4T 2000; bản ghi D3 (`training/data/d3_records.py`, lọc chương trình suy biến + câu rác + ≤ 3 bản / câu): self 16,688, qwen35a3b 18,146, gptoss 10,079, gemma31 17,043, qwen27 (đang dựng); thư mục nhánh `training/data/out/r2/` (symlink); `training/data/mix.py` trộn cố định |
| WP5 D3 theo thầy (G8–G11) | **xong** 08/11 | giữ sau vòng lặp ngược: self 51.8 %, qwen35a3b 64.4 %, gptoss 31.9 % (nhiều câu rác), gemma31 69.5 %, **qwen27 73.6 %** |
| WP6 nhánh fine-tune (G12–G19) | **xong** | vòng 1–3 + QĐ9 (entry 38–43): fine-tune đơn < gốc trên held-out; **súp + WiSE-FT** thắng: súp top-4 α 0.35–0.7 held-out gộp (pub_ho + tr_ho) ~0.74 vs gốc 0.713; nhiễu pub_ho ±1 |
| WP7 huấn luyện đầy đủ (G20–G21) | **xong** | **cấu hình cuối: súp WF1–4 (thành viên prompt full, dữ liệu v2, adapter `quynhle07/zc-adapters-wf`, `phatle1578/zc-adapters-w8`) α 0.35 + prompt full + think vòng 2, không rescue** — public gốc 0.939 (3 lần WF + think: 0.939 / 0.930 / 0.939), held-out gộp 0.752 (entry 45–46) |
| WP8 tin cậy / bộ kiểm / embedding (G22–G24) | **đang** | `tests/calibrate.py` + `--calibrator` nối vào luồng (QĐ14; test o6); M11 đọc lại: **không giữ** (A/B −2.5 quick400); biến thể `think` vòng 2: pub_ho +1.4 / quick400 −1.75 (ứng viên); QĐ11 n-gram |
| WP9 sửa lỗi tới public = 1.0 (G25) | **đang** | vòng 1–3: gợi ý đơn vị / kỳ năm-trước / `debt_aging` / nhiều nhóm → list; cờ postcheck ngược thông tin bỏ (phát lại log +0.7–2 điểm mọi tập) — entry 39, 44; `tests/test_wp9_fixes.py` |
| WP10 đánh giá cuối (G27–G29) | **đang** | job `automation/jobs/final_{a,b,stress}.jobs`: public gốc × 2 (+ pub_k0 làm dev hiệu chỉnh θ), pub_noind (ẩn hàm ngành), pub_noacc; sau đó `tests/calibrate.py` trên pub_full + pub_k0 → `configs/flow_final.json` |
| WP11 Docker + nộp | **đang** | `docker/predict.sh`, `Dockerfile` đã chuyển sang luồng hybrid (`configs/flow_final.json` chưa có) |
| WP12 DuckDB | **xong** | QĐ12: **không giữ** (mô hình không viết `<sql>` trên pub_noind; code giữ, tắt) — entry 42 |

## Tiến độ P2 (sau kế hoạch 1 — `docs/13_TASK_AFTER_PLAN1.md`; kế hoạch: **`docs/12_PLAN2.md`**; dừng khi public gốc = 100 % và held-out không giảm)

| Bước / gói | Trạng thái | Public gốc | Held-out (pub_ho / tr_ho) | Ghi chú |
|---|---|---|---|---|
| B1 hoàn tất kế hoạch 1 + liệt kê câu sai | xong phần public (WP10 final_a/b/stress đang chạy) | 0.939 (956/17/27); đường container `final_predict` 0.938 (955/17/28, 1423 s / 1000 câu, 4×L4) | 0.592 / ~0.87 | `tests/list_wrong.py`: 44 câu sai lần tốt nhất, 63 hợp 3 lần, 29 ổn định |
| B2 phân tích lỗi → `docs/10_ERROR_ANALYSIS.md` | xong (public); train dev chờ tr_dev500 | | | 7 nhóm: bị cắt / không chạy 27 %, ý định 23 %, nghiệp vụ 16 %, chọn 14 %, định dạng 9 %, câu bẩn 9 %, linking 2 %; nguyên nhân gốc F1–F8 |
| B3 tài liệu ngoài → `docs/11_LITERATURE.md` | xong | | | |
| B4 kế hoạch 2 → `docs/12_PLAN2.md` | xong | | | WP-P2-01…10, lưới thí nghiệm, quy tắc giữ / bỏ |
| WP-P2-01 gợi ý + làm sạch câu | CPU xong | phát lại +0.9–1.1 | | kiểu gợi ý đúng 0.967 → 0.999 |
| WP-P2-02 trần 1536 token | chờ GPU (`p2_b`) | | | |
| WP-P2-03 định dạng + cờ kỳ lệch | CPU xong | | | |
| WP-P2-04 review thiếu dữ liệu | chờ GPU (`p2_c`) | | | |
| Vòng 1: **p2_c** (= b + review thiếu dữ liệu) | **xong — tốt nhất** | **0.972** (981 / 9 / 10) | **0.601 / 0.874** (gộp **0.757**) | giữ |
| Vòng 2: **p2_d** (= c + gợi ý `counted` / `thresholds`) | xong | **0.975** (985 / 10 / 5) | 0.600 / 0.868 (gộp 0.753 vs c 0.757: −0.38, trong nhiễu) | chưa quyết: sửa 7 câu của c ("mấy bữa", cùng kỳ năm trước…), thêm 3; lặp seed ở vòng 3 |
| Vòng 2: **p2_f** (= b + gợi ý `counted`) | xong | 0.967 (980 / 13 / 7) = b | 0.605 / 0.869 (gộp 0.756 vs b 0.753) | gợi ý đếm ~trung tính / hơi tốt (d vs c: public +0.3, held-out −0.4; f vs b: public 0, held-out +0.2) → lặp seed vòng 3 |
| Vòng 3 GPU (`p2_g`, `p2_g_s1`, `p2_d` code mới, `p2_c_s1`) | **đang chạy** (09/11 20:45–20:53 UTC) | | | `quynhtran35/zc-p2r3g-1`, `quynhtran35/zc-p2r3gs1-1`, `nhatlong1103/zc-p2r3d-1`, `nhatlong1103/zc-p2r3cs1-1`; **huyenco hết 30 h GPU W46**; `quynhtran35/zc-final-adapter` tạo |
| Vòng 3: `p2_c_s1` (c lặp seed 1, code vòng 3) | xong | 0.973 (981 / 8 / 11) | 0.605 / 0.864 (gộp 0.753) | nhiễu c: public ±0.1, gộp 0.753–0.757 → d (0.753) nằm trong nhiễu của c |
| Vòng 3: `p2_d` code vòng 3 (+ thực thể việc / size, lời = lãi) | xong | 0.968 (978 / 10 / 12) | 0.597 / 0.864 (gộp 0.749) | thấp hơn d (0.975 / 0.753) và c → nhóm gợi ý thêm trung tính / hơi âm; vòng 5 dựng trên **c** |
| Vòng 3: `p2_g_s1` (d code vòng 3 + cờ named-periods, seed 1) | xong | 0.972 (982 / 10 / 8) | 0.609 / 0.868 (gộp **0.757**) | ngang c |
| Vòng 3: **`p2_g`** (d code vòng 3 + cờ named-periods) | xong | **0.980** (986 / 6 / 8); tb 2 seed **0.976** | 0.601 / 0.871 (gộp 0.755); tb 2 seed **0.756** | **dẫn đầu** (c tb: 0.9725 / 0.755); 22 câu sai ở ≥ 1 seed, 10 ổn định: đếm đơn vị, ≥ N đơn, hạn dùng N ngày, size → các kiểm tra vòng 4–5 |
| Vòng 5: `p2_i` (c + precheck lỗi size / unit) | xong | 0.972 (981 / 9 / 10) | 0.603 / 0.870 (gộp 0.756) | kiểm tra đúng mục tiêu: 10 / 10 lượt sửa sau lỗi size / unit ra đúng, 5 / 5 câu đếm đơn vị đúng (4 câu c sai); câu mới sai là phiếu tách (nhiễu) |
| Vòng 5: `p2_i_s1` | xong | **0.977** (986 / 9 / 5); tb i 0.9745 | 0.611 / 0.875 (gộp **0.762**); tb i **0.759** | i > c cả hai (tb +0.2 public, +0.4 held-out) → **giữ size / unit** |
| Vòng 5: **`p2_j`** (g + ngưỡng + size / unit), 2 seed | xong | 0.977 (982 / **5** / 13), 0.974 (982 / 8 / 10); tb **0.9755** | 0.605 / 0.868, 0.613 / 0.874; tb **0.758** | bình nguyên: c 0.9725 / 0.755, g 0.976 / 0.756, h 0.9755 / 0.758, i 0.9745 / 0.759, j 0.9755 / 0.758; j ít sai nhất, bỏ nhiều nhất. θ 0.375 phát lại: public j +0.7 / +0.3 nhưng pub_ho −1.0 / −0.5 → **giữ θ 0.5** (cấm tăng public bằng giảm held-out); câu bỏ cần ứng viên tốt hơn (vòng 6: 12 mẫu, review 3) |
| **Nộp**: `configs/flow_final.json` = **j** (10/11) | | 0.9755 (tb 2 seed) | 0.758 | README_SOLUTION cập nhật; adapter giữ nguyên |
| **Đường container** với `flow_final.json` (= j), `quynhtran35/zc-p2r6pred-1` | xong | **0.978** (985 / 7 / 8) | — | `predict_flow.py` (lối vào Docker) tái lập số eval; 1,262 s / 1,000 câu kể cả nạp mô hình (4×L4) |
| Vòng 6: `p2_l` (j + review 3) | xong | 0.972 (979 / 7 / 14) | 0.615 / 0.868 (gộp 0.759) | trong nhiễu của j; review 3 chủ yếu thêm câu bỏ → chưa giữ (chờ k) |
| Vòng 7 GPU (`p2_j` trên code `next_n_days`, 2 seed) | **đang chạy** (10/11 02:51 UTC, CPU suite ALL PASS) | | | `phatle1578/zc-p2r7j-1`, `phatle1578/zc-p2r7js1-1` |
| **Hướng M** (cải tiến mô hình từ M0 = 0.939, không luật; `docs/12_PLAN2.md` §6, WP-M0…M9) | chưa bắt đầu (kế hoạch xong 10/11) | M0 0.939 (tb 0.934) | 0.592 / 0.873 (0.752) | thứ tự: M7 nhiều bộ sinh → M6 bộ chọn → M1 + M2 fine-tune lại / tự chưng cất → M3 thầy → M4 DPO → M5 GRPO, M8 dataset ngoài, M9 học sinh khác |
| Vòng 6 GPU (`p2_l` = j + review 3; `p2_k` = l + 12 mẫu, `p2_k_s1`; `p2r6pred` = đường container với flow_final) | **đang chạy** (10/11 01:40 UTC, code `0b154ac`/`ca0e183`, CPU suite ALL PASS sau khi container khởi động lại) | | | `nhatlong1103/zc-p2r6k-1`, `nhatlong1103/zc-p2r6ks1-1`, `quynhtran35/zc-p2r6l-1`, `quynhtran35/zc-p2r6pred-1` |
| Vòng 5 GPU (`p2_i` = c + precheck lỗi size / unit; `p2_i_s1`) | **đang chạy** (09/11 22:45 UTC, code `89243a8`, CPU suite ALL PASS) | | | `nhatlong1103/zc-p2r5i-1`, `nhatlong1103/zc-p2r5is1-1`; hai luật chỉ chạm mẫu sai trên mọi log (size 54 + 114, unit 71 + 104) |
| Vòng 4: `p2_h` (d + cờ ngưỡng, code vòng 4) | xong | 0.971 (981 / 10 / 9) | 0.611 / 0.864 (gộp 0.755) | ngang c / g ở held-out, thấp hơn g ở public |
| Vòng 4: `p2_h_s1` | xong | **0.980** (986 / 6 / 8); tb h 0.9755 | 0.615 / 0.869 (gộp **0.760**); tb h **0.758** | h ≈ g (tb 0.976 / 0.756) trong nhiễu |
| Vòng 4 GPU (`p2_h` = d + cờ ngưỡng, code: hạn dùng N ngày; `p2_h_s1`) | **đang chạy** (09/11 21:40 UTC, code `6bf3b19`/`e6dd20d`, CPU suite ALL PASS) | | | `phatle1578/zc-p2r4h-1`, `phatle1578/zc-p2r4hs1-1` |
| Vòng 2: **p2_e** (d nhưng prefill `<plan>`, trần 512) | xong — **loại** | 0.898 (939 / 41 / 20) | 0.575 / 0.874 | ép bắt đầu bằng `<plan>` bỏ phần lập luận tự do của súp → sai tăng; giữ cách sửa bằng trần 1536 |
| Vòng 1: p2_a (sửa gợi ý / làm sạch / định dạng / cờ kỳ, trần 512) | xong | 0.965 (976 / 11 / 13) | 0.603 / 0.871 (gộp 0.756) | phần lớn mức tăng đến từ sửa gợi ý; train dev `tr_dev500` 0.992 (498 / 2 / 0: "lời gộp" đọc thành lãi ròng, tăng trưởng cùng kỳ năm trước so với tháng trước → sửa vòng 3) |
| Vòng 1 GPU (`p2_a/b/c`) | **xong cả 3** | **p2_b 0.967** (979 / 12 / 9; mốc 0.934 tb 5 lần, tốt nhất 0.939) | **0.603 / 0.866** (mốc 0.592 / 0.873; gộp 0.753 vs 0.752: không giảm → **giữ**) | p2_b = sửa gợi ý / làm sạch / định dạng / cờ kỳ + trần 1536 token; 1,000 + 3,199 câu trong 4,488 s (merge 792 s), GPU 70–78 %; 21 câu còn sai (15 cũ, 6 mới) |
| Vòng 2 GPU (`p2_d/e/f`) | **đang chạy** (09/11 18:50 UTC, code `c49da84`/`d9b30fb`, CPU suite ALL PASS) | | | `huyenco/zc-p2r2d-1`, `huyenco/zc-p2r2e-1`, `phatle1578/zc-p2r2f-1`; gợi ý `counted` / `thresholds`, prefill `<plan>`, đồng nghĩa %; kiểu gợi ý = đáp án 100 % câu có gợi ý (train + public) |
| WP10 stress (pub_noind, pub_noacc) | không đo được (kernel dừng "STALE CODE" sau restart hàng đợi) → đo trên cấu hình P2 cuối | | | |

## Quyết định của user (10/11)

- Trước mắt **tiếp tục thêm luật tổng quát** (hướng R). Kế hoạch 2 có thêm **hướng M** (`docs/12_PLAN2.md` §6): xuất phát từ bản **không có
  luật P2** có public cao nhất (M0 = code `b093d46` + flow_final cũ: 0.939), thử song song nhiều cách cải tiến **mô hình** (mô hình phụ,
  fine-tune lại, dataset ngoài, chưng cất, DPO / RL…) đến khi public gốc = 1.0; không thêm luật ở hướng M.

## Quyết định của user (09/11)

- **Ưu tiên điểm public gốc** (đề private giống phong cách public): tiêu chí chọn chính = `pub_full` đủ 1000 câu (chạy 2 lần vì nhiễu); held-out (pub_ho + tr_ho) là rào chắn chống quá khớp (không được giảm quá nhiễu); vẫn chỉ sửa tổng quát.
- yeshenhi không thể có L4 (bỏ qua); cần thêm acc L4 thì báo user.
- Ngày 15/11 user tự build Docker, chạy `run_public_test.sh`, nộp portal.

## Kaggle

08/11: user cấp 9 token (`~/.kaggle/tok_<acc>`); probe 4×L4 OK trên 8 acc: nhatlong1103, quynhle07, quynhtran35, nguyenhoa99,
yuan1103 (~15 h W46 đã dùng), synchan5704, anderson1202, loinguyentran (quota W46 của các acc mới chưa rõ: acc dùng chung dự án
khác). yeshenhi: chỉ T4 (dùng cho kernel wheels / T4). Đủ 8 acc L4 mà kế hoạch §9 cần (7, lý tưởng 8).
09/11: user cấp thêm **phatle1578** (probe 4×L4 OK, 48 CPU; đã có `phatle1578/zc-para` trùng sha256 với bản đang dùng) và token mới cho **yeshenhi** (probe: vẫn không có GPU L4 → cần user accept rules `gemma-4-developer-agent` trên web bằng acc này); thêm **mucnokieu**, **huyenco** (probe 4×L4 OK, 48 CPU; tạo `<acc>/zc-para` riêng tư, cùng tệp). → **11 acc 4×L4**. 09/11 18:50: **quynhle07 hết 30 h GPU tuần W46**; `huyenco/zc-final-adapter` (bản riêng tư của adapter cuối) tạo để chạy vòng 2.

## Việc tiếp theo (user đã duyệt 08/11): triển khai code đúng `docs/08_MASTER_PLAN.md` §8 (WP0 → WP12), đặc tả §5, quy tắc §10

1. WP0 chuẩn bị → WP1 chạy thử GPU (yuan1103) → QĐ1–QĐ3, QĐ12.
2. WP2 (P0) luồng hybrid v1: `api_spec`, `grammar` (xgrammar), `hints`, `tools` (find / period / check / help), `precheck`,
   `prompting`, `predict_flow` mới; test + preflight. Song song WP4 nhà máy dữ liệu.
3. WP3 E0 + chọn mô hình chính + A/B → WP5 dò thầy → WP6 nhánh fine-tune → WP7 huấn luyện đầy đủ; song song WP8 (P1: data
   card, đọc kết quả, bộ kiểm 0.8B, truy hồi embedding 0.6B, tin cậy), WP12 (DuckDB), WP9 sửa lỗi tới public = 1.0.
4. WP10 đánh giá cuối (14/11) → WP11 Docker; **user** build ngày 15/11 và nộp trước **16/11/2026 23:59 GMT+7**.
