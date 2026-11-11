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
| Vòng 6: `p2_k` (l + 12 mẫu) | xong | 0.978 (984 / 6 / 10) | 0.607 / 0.871 (gộp 0.758) | ngang j ở held-out, +0.25 public; chậm hơn ~35 % / câu; chờ k seed 1 |
| Vòng 6: `p2_k_s1` | xong | 0.975 (984 / 9 / 7); tb k **0.9765** | 0.610 / 0.875 (0.761); tb k **0.760** | k vs j: +0.1 public (< ngưỡng +0.3), held-out +0.2, chậm 35 % → **không đổi bản nộp** (j giữ); review 3 + 12 mẫu ghi nhận là trung tính |
| Vòng 7 GPU (`p2_j` trên code `next_n_days`, 2 seed) | **đang chạy** (10/11 02:51 UTC, CPU suite ALL PASS) | | | `phatle1578/zc-p2r7j-1`, `phatle1578/zc-p2r7js1-1` |
| Vòng 7: `p2_j` code `next_n_days` (seed 0) | xong | **0.978** (986 / 8 / 6) | 0.606 / 0.869 (0.756) | 2 / 3 câu "hết hạn trong N ngày tới" nay đúng (sai mọi lần trước); chờ seed 1 |
| Vòng 7: `p2_j` code `next_n_days` (seed 1) | xong | **0.980** (987 / 7 / 6); tb **0.979** | 0.618 / 0.869 (0.761); tb **0.758** | vs j code cũ (0.9755 / 0.758): +0.35 public, held-out bằng → **giữ**; **bản nộp = code hiện tại + flow_final (j): 0.979 / 0.758** |
| Vòng 8 GPU (`p2_j` trên code làm sạch "the" → thế / thẻ, 2 seed) | **đang chạy** (10/11 05:29 UTC, CPU suite ALL PASS) | | | `mucnokieu/zc-p2r8j-1`, `mucnokieu/zc-p2r8js1-1`. **Quota W46 hết: quynhle07, huyenco, nhatlong1103, quynhtran35, phatle1578**; còn mucnokieu, nguyenhoa99, yuan1103 (~15 h đã dùng), synchan5704, loinguyentran, anderson1202 (chưa có bản adapter) |
| Vòng 8: `p2_j` code "the" → thế / thẻ (seed 0) | xong | **0.982** (988 / 6 / 6) | 0.615 / 0.876 (gộp **0.764**) | tốt nhất một lần chạy; chờ seed 1 |
| Vòng 8: seed 1 | xong | 0.981 (987 / 6 / 7); tb **0.9815** | 0.608 / 0.871 (0.758); tb **0.761** | vs vòng 7 (0.979 / 0.758): +0.25 / +0.25, sửa lỗi làm sạch → **giữ**; **bản nộp = code hiện tại + flow_final (j): 0.9815 / 0.761** |
| Vòng 9 GPU (`p2_n` = j + kiểm tra lỗi len(mặt nạ), 2 seed) | **đang chạy** (10/11 08:12 UTC, CPU suite ALL PASS) | | | `mucnokieu/zc-p2r9n-1`, `mucnokieu/zc-p2r9ns1-1`; luật chỉ chạm mẫu sai trên mọi log (24 + 14) |
| Vòng 9: `p2_n` seed 0 | xong | 0.982 (987 / 5 / 8) | 0.608 / 0.866 (0.756) | kiểm tra len(mặt nạ) hoạt động (1 lần sửa) nhưng public không có mẫu như vậy lần này; câu "≥ N đơn" bỏ vì phiếu tách; chờ seed 1 |
| Vòng 9: `p2_n` seed 1 | xong | 0.980 (986 / 6 / 8); tb n 0.981 | 0.617 / 0.873 (0.763); tb n 0.759 | vs vòng 8 (0.9815 / 0.761): trung tính → **không đổi bản nộp**. Hướng R gần bão hoà: 11 câu sai ổn định là phiếu tách / lựa chọn của mô hình (≥ N đơn, size bỉm, "uy tín", tỉ trọng máy theo số lượng, nhập theo viên…) → chuyển trọng tâm sang **hướng M** (bộ chọn M6, nhiều bộ sinh M7) |
| **Hướng M** (cải tiến mô hình từ M0 = 0.939, không luật; `docs/12_PLAN2.md` §6, WP-M0…M9) | **bắt đầu 10/11**: M6 (bộ chọn học được) — dữ liệu huấn luyện = log M0 trên câu train dev (`tr_dev_all` 2369) + paraphrase train dev (`tr_dev_para` 4494); kernel `nguyenhoa99/zc-m0dev-1`, `nguyenhoa99/zc-m0para-1` (code đóng băng `b093d46`, worktree `wt_m0`); đánh giá bộ chọn bằng phát lại log M0 có sẵn (public + held-out), không tốn GPU. m0dev xong: PA 0.990 (2353 / 7 / 9) → trong mẫu, quá ít ví dụ sai → thêm `d3sel` (3000 câu thầy D3 của DN train, ngoài mẫu với súp; `nguyenhoa99/zc-m0d3-1` đang chạy). m0para: PA 0.788 (3939 / 396 / 159) → dữ liệu thật. **Bộ chọn M6 lần 1** (train = m0dev + m0para, θ 0.6 chọn trên train), phát lại log M0: public 0.939 → 0.931 / 0.934 → 0.934 / 0.930 → 0.933, pub_ho **0.592 → 0.614**, tr_ho **0.873 → 0.885** (chủ yếu bỏ nhiều hơn); chế độ rerank: public ±0.1, held-out +0.4–0.5 → chưa giữ (public chưa tăng); chờ thêm dữ liệu d3sel. m0d3: PA 0.919 (2743 / 62 / 112; gần độ khó public). **Bộ chọn M6 lần 2** (train = m0dev + m0para + m0d3, θ 0.6, `src/refs/selector/selector_m0.json`), phát lại log M0: public **0.939 → 0.939, 0.934 → 0.942, 0.930 → 0.942** (tb 0.934 → **0.941**), pub_ho **0.592 → 0.610**, tr_ho **0.873 → 0.884** (gộp 0.752 → **0.767**) → **giữ cho hướng M** (M0 + M6). Áp lên log hướng R (j): public −0.5…−0.7, held-out +0.5…+1.8 → không dùng cho R (cần dữ liệu train của cấu hình j). Bước sau: nối bộ chọn vào luồng (M), kiểm GPU. Dữ liệu bộ chọn cho hướng R (j trên tr_dev_all + d3sel, tr_dev_para): `nguynduylm/zc-jseldev-1`, `nguynduylm/zc-jselpara-1` đang chạy. **mucnokieu cũng hết quota W46** (6 / 11 acc); `nguyenhoa99/zc-final-adapter` tạo | M0 0.939 (tb 0.934) | 0.592 / 0.873 (0.752) | thứ tự: M7 nhiều bộ sinh → M6 bộ chọn → M1 + M2 fine-tune lại / tự chưng cất → M3 thầy → M4 DPO → M5 GRPO, M8 dataset ngoài, M9 học sinh khác |
| Hướng M — **M7 lần 1: WF3, WF4 làm bộ sinh riêng** (`phuongnehe/zc-m7b-1`, code M0, mỗi thành viên × 0.35) | xong | WF3 **0.951** (967 / 16 / 17), WF4 0.940; bộ chọn bất biến: 0.954 / 0.947 | WF3 0.597 / 0.882 (gộp **0.759**), WF4 0.605 / 0.875 (0.759); bộ chọn: WF3 0.617 / 0.894 (**0.775**), WF4 0.626 / 0.887 (0.775) | **một thành viên tốt hơn súp 4 thành viên** (M0 tb 0.934 / 0.752; + bộ chọn 0.944 / 0.767): WF3 (D1 + paraphrase) và WF4 (D1 + câu nhiễu) × 0.35 đều hơn ở held-out, WF3 hơn cả public (+1.7). Gộp mẫu WF3 + WF4 + bộ chọn: public 0.951, held-out **0.781** (cao nhất), gấp đôi thời gian sinh. 1 seed → chờ WF1 / WF2 (`zc-m7a-1`); hướng R thử J + WF3 × 0.35 (2 seed, vòng 10) |
| **Vòng 12 R: J + WF3 × 0.35 + RFT trên WF3 (RS1–4 × 0.25)**, 2 seed (`thnliu/zc-m2w3-1`) | xong — **ĐỔI MÔ HÌNH NỘP** | 0.981 (989 / 8 / 3), 0.980 (988 / 8 / 4); tb **0.9805** | 0.627 / 0.886, 0.631 / 0.889; gộp tb **0.776** | cùng seed với J + WF3: public +0.05, held-out +0.3; vs mô hình nộp cũ (súp WF1–4, vòng 8: 0.9815 / 0.761): public −0.1 (trong nhiễu), held-out **+1.5** (pub_ho +1.7, tr_ho +1.3) → nộp: `soup.json` = WF3 × 0.35 + RS1–4 × 0.25 (`automation/set_final_model.py`, dataset `nhatlong1103/zc-final-adapter` phiên bản mới, `src/adapter_hashes.json`, README_SOLUTION); kiểm đường container: `thnliu/zc-p2r12pred-1` |
| Vòng 16 R: **RFT trên WG1** (RS1–4 × 0.25, J 2 seed; `anderson1202/zc-m2g1-1`) | xong — **loại** | 0.980 (986 / 6 / 8), 0.980 (986 / 6 / 8) | 0.641 / 0.888 (0.782), 0.630 / 0.888 (0.777); tb 0.780 | vs J + WG1 cùng seed (0.986 / 0.7765): public −0.6, held-out +0.3 → loại (public giảm) |
| Đường container (`predict_flow.py`) với **O + WG1** (`loinguyentran/zc-p2r16pred-1`) | xong | **0.986** (992 / 6 / 2) | — | khớp eval (tb 0.9855); 1,000 dòng + 1,000 log; gộp 111 s, trả lời 1,172 s (4×L4) → bản nộp đã kiểm đường Docker |
| **BẢN NỘP (11/11 04:55 UTC): O + WG1 × 0.35** | **chốt** | tb 2 seed **0.9855** (0.986, 0.985) | tb **0.778** (0.644 / 0.882, 0.636 / 0.881) | `configs/flow_final.json` = O (J + mask_check); `soup.json` = WG1 × 0.35 (dataset `nhatlong1103/zc-final-adapter` phiên bản mới, `src/adapter_hashes.json`), README_SOLUTION cập nhật; mô hình nộp trước (WF3 + RFT) 0.9805 / 0.776; súp đầu tiên 0.9815 / 0.761. Kiểm đường container: `loinguyentran/zc-p2r16pred-1`. `final_submission` = 0.991 (file public tốt nhất, seed 0 của J + WG1) |
| Vòng 15 R: **O = J + mask_check** (len() của mặt nạ → lỗi → lượt sửa; bắt cả list() / .tolist() / .values) + WG1, 2 seed (`yuan1103/zc-p2r15o-1`, `zc-p2r15os1-1`) | xong — **giữ** | 0.986 (990 / 4 / 6), 0.985 (991 / 6 / 3); tb **0.9855** | 0.644 / 0.882 (0.780), 0.636 / 0.881 (0.776); tb **0.778** | vs J + WG1 cùng seed (0.991 / 0.981; 0.7765): public bằng (dao động nhỏ hơn), held-out +0.1; câu '≥ 152 đơn' nay đúng (lượt sửa sau lỗi len_mask). Sửa lỗi hiển nhiên, không kém → giữ (§4). Ứng viên nộp: **O + WG1** (chờ RFT trên WG1) |
| Vòng 14 R: súp **WG1 + WG3** / **WG1 + WG2** × 0.175 (`yuan1103/zc-p2r14g13-1`, `zc-p2r14g12-1`) | xong | 0.979 (986 / 7 / 7) / 0.977 (986 / 9 / 5) | 0.631 / 0.881 (0.773) / 0.639 / 0.882 (0.778) | đều dưới WG1 một mình (tb 0.985 / 0.776) ở public → súp lại không hơn thành viên tốt nhất (như WF3 vs súp WF3 + WF4) |
| Vòng 13 R: **WG1 × 0.35, seed 1 + 2** (`duylnng/zc-p2r13wg1s1-1`, `zc-p2r13wg1s2-1`) | xong | 0.981 (990 / 9 / 1), 0.984 (989 / 5 / 6); **3 seed tb 0.985** | 0.640 / 0.876 (0.775), 0.635 / 0.880 (0.775); 3 seed tb **0.776** | vs mô hình nộp hiện tại (WF3 + RFT: 0.9805 / 0.776): public **+0.5**, held-out bằng → đạt §4; 0.991 của seed 0 phần lớn là may của lần chạy. Chờ RFT trên WG1 (`anderson1202/zc-m2g1-1`) và súp WG1 + WG3 / WG1 + WG2 để chọn một lần |
| **Vòng 13 R: thành viên mới × 0.35 với J** (seed 0) | xong | WG1 (D1 + toàn bộ D2) **0.991** (994 / 3 / 3 — cao nhất, `final_submission` = 0.991); WG2 (D1 + D2 + D2N) 0.979 (986 / 7 / 7); WG3 (= WG1, seed khác) 0.983 (989 / 6 / 5); WG4 (D1 + D2 + 710 câu khó RFT) 0.981 (989 / 8 / 3) | WG1 0.641 / 0.882 (gộp 0.778); WG2 **0.644** / 0.886 (**0.782**); WG3 0.625 / 0.876 (0.768); WG4 0.602 / 0.869 (0.754) | vs J + WF3 (tb 0.979 / 0.773): D1 + toàn bộ D2 cao hơn ở public (WG1 / WG3 tb 0.987); WG2 cao nhất ở held-out; trộn RFT vào SFT từ đầu (WG4) hại held-out (RFT chỉ giúp khi học **trên** mô hình đã fine-tune). Đang chạy: WG1 seed 1, 2 (`duylnng`), súp WG1 + WG3 và WG1 + WG2 × 0.175 (`yuan1103/zc-p2r14g13-1`, `zc-p2r14g12-1`), RFT trên WG1 + J 2 seed (`anderson1202/zc-m2g1-1`) |
| Đường container (`predict_flow.py`) với mô hình nộp mới (`thnliu/zc-p2r12pred-1`) | xong | **0.977** (985 / 8 / 7) | — | chạy đúng đường Docker: 1,000 dòng, 1,000 log từng câu; gộp 601 s, trả lời 1,336 s (4×L4); trong nhiễu của eval (0.981 / 0.980). Mô hình cũ: container 0.978 vs eval tb 0.9755 |
| **Thành viên mới WG1–4** (`loinguyentran/zc-wg-1`) | huấn luyện xong | | | 66 bước / nhánh, 14.7M token, 17,287–18,745 s, đỉnh 18.8 GB; đánh giá × 0.35 với J: `duylnng/zc-p2r13wg1-1`, `zc-p2r13wg2-1`, `linhmai24/zc-p2r13wg3-1`, `zc-p2r13wg4-1` (đang chạy). Ổ đĩa đầy lúc tải output (đã dọn: test nháp, adapter nhánh cũ / bị loại) |
| Vòng 11 R: J + súp WF3 + WF4, **seed 3** (`duylnng/zc-p2r11s34s2-1`) | xong | 0.971 (980 / 9 / 11) | 0.610 / 0.881 (gộp 0.765) | 3 seed: tb 0.977 / 0.769 < J + WF3 (0.979 / 0.773) → **WF3 thắng súp WF3 + WF4**; còn chờ J + WF3 + RFT |
| Vòng 10 R: J + WF3 × 0.35, **seed 3** (`duylnng/zc-p2r10w3s2-1`) | xong | 0.978 (986 / 8 / 6) | 0.631 / 0.878 (gộp 0.772) | 3 seed: public tb **0.979**, held-out tb **0.773** (vòng 8 súp: 0.9815 / 0.761) |
| Vòng 11 R: **J + súp WF3 + WF4 × 0.175**, 2 seed (`linhmai24/zc-p2r11s34-1`, `zc-p2r11s34s1-1`) | xong | 0.977 (985 / 8 / 7), **0.983** (989 / 6 / 5 — cao nhất, `final_submission` cập nhật); tb 0.980 | 0.632 / 0.881, 0.624 / 0.876; gộp tb 0.771 | vs J + WF3 (tb 0.980 / 0.774): public bằng, held-out −0.3; vs J + súp WF1–4 (vòng 8: 0.9815 / 0.761): held-out +1.0. Chọn mô hình nộp sau khi có J + WF3 + RFT (`thnliu/zc-m2w3-1`) |
| **Vòng 10 R: J + WF3 × 0.35** (thay súp WF1–4), 2 seed (`linhmai24/zc-p2r10w3-1`, `zc-p2r10w3s1-1`) | xong — **ứng viên bản nộp** | 0.981 (986 / 5 / 9), 0.979 (987 / 8 / 5); tb **0.980** | 0.622 / 0.887, 0.629 / 0.883; tb 0.626 / 0.885 (gộp **0.774**) | vs vòng 8 (j + súp, tb 0.9815 / 0.761): public −0.15 (trong nhiễu ±0.4), held-out **+1.3** (pub_ho +1.4, tr_ho +1.1 — tr_ho = DN chưa học, gần private nhất). Mục tiêu cuối = PA private → đổi mô hình nộp sang WF3 trừ khi súp WF3 + WF4 tốt hơn (vòng 11 đang chạy: `linhmai24/zc-p2r11s34-1`, `zc-p2r11s34s1-1`); `final_submission` giữ 0.982 (luật: chỉ thay khi public cao hơn) |
| Hướng M — **M4 DPO trên súp M0** (`duylnng/zc-m4dpo-1`) | xong — **loại** | × 1.0: 0.935 (953 / 18 / 29); × 0.5: 0.935 | × 1.0: 0.589 / 0.870 (gộp 0.749); × 0.5: 0.747 | vs M0 0.934 / 0.752: public +0.1, held-out −0.3 / −0.5 → loại. Huấn luyện chạy đúng (FSDP 4 GPU, 799 cặp, 25 bước, 4,309 s, đỉnh 12.8 GB, margin 9.2, đúng cặp 0.91): khớp cặp nhưng không tổng quát (nhiều cặp loại = văn bản bị cắt; β 0.1 + NLL 0.2) |
| Hướng M — **M2 RFT trên súp M0** (`thnliu/zc-m2rft-1`) | xong — **giữ (M)** | súp + tb RS1–4 × 1.0: **0.939** (959 / 20 / 21; bộ chọn 0.949); × 0.5: 0.930 | × 1.0: 0.610 / 0.886 (gộp **0.768**; bộ chọn 0.774); × 0.5: 0.757 | vs M0 0.934 / 0.752: public +0.5, held-out **+1.6** → giữ theo §4. Huấn luyện: 19 bước / nhánh, ~5,400 s, loss tham chiếu giảm 0.10–0.16 → 0.04–0.11. Vẫn dưới WF3 × 0.35 ở public (0.951) → bước sau: RFT **trên WF3** + đánh giá J 2 seed (`thnliu/zc-m2w3-1`) |
| Hướng M — M7: **súp WF3 + WF4 × 0.175** (`phuongnehe/zc-m7c-1`, code M0) | xong | 0.943 (957 / 14 / 29; bộ chọn 0.949) | 0.608 / 0.881 (gộp 0.764; bộ chọn 0.775) | vs WF3 × 0.35 (0.951 / 0.759; bộ chọn 0.954 / 0.775): public −0.8, held-out +0.5 → WF3 một mình ≥ súp WF3 + WF4 |
| **Huấn luyện thành viên mới WG1–4** (`loinguyentran/zc-wg-1`, 17:02 UTC, ~5.5 h) | **đang chạy** | | | kiểu WF3 (thắng ở M7): mỗi nhánh 14.7M token như WF3; WG1 / WG3 = D1 + toàn bộ D2 (3,818 paraphrase; 2 seed), WG2 = D1 + D2 + D2N, WG4 = D1 + D2 + 710 câu khó RFT; sau đó đánh giá từng nhánh × 0.35 (J và M0) |
| Hướng M — M7: **WF3 × 0.5** (`phuongnehe/zc-m7d-1`, code M0) | xong | 0.936 (955 / 19 / 26; bộ chọn 0.940) | 0.612 / 0.884 (gộp 0.767; bộ chọn 0.780) | vs WF3 × 0.35 (0.951 / 0.759): public −1.5, held-out +0.8 → giữ α 0.35 (cùng hướng với đường α của súp ở kế hoạch 1) |
| Hướng M — **M7 lần 2: WF1, WF2** (`phuongnehe/zc-m7a-1`) + tổng kết M7 | xong | WF1 0.924, WF2 0.931 (bộ chọn 0.940 / 0.947) | WF1 0.574 / 0.859 (0.737), WF2 0.581 / 0.861 (0.741) | thành viên chỉ học D1 (WF1, WF2) yếu hơn súp; thành viên có paraphrase / câu nhiễu (WF3, WF4) mạnh hơn → súp 4 thành viên bị kéo xuống. Gộp mẫu: WF3 + WF4 + bộ chọn 0.951 / **0.781**; gộp cả 5 không hơn. Bước sau: súp WF3 + WF4 × 0.175 (`phuongnehe/zc-m7c-1`), WF3 × 0.5 (`zc-m7d-1`); hướng R: J + WF3 × 0.35 (`linhmai24/zc-p2r10w3-1`, `zc-p2r10w3s1-1`) |
| Hướng R — bộ chọn huấn luyện trên log J (10/11 14:20 UTC) | xong — **loại** | | | dữ liệu: J trên `tr_dev_all` 0.997, `d3sel` 0.925, `tr_dev_para` 0.786 (M0: 0.990 / 0.919 / 0.788 — luật R không giúp paraphrase train). Bộ chọn J (đặc trưng bất biến, θ 0.65; hoặc đủ đặc trưng) phát lại 6 lần chạy J (vòng 7–9): public **−0.1…−0.8** mọi lần (0.982 → 0.975), held-out +0.1…+1.7 → không dùng cho R (luật public không được giảm); chế độ rerank: public −0.1 / 0, held-out +0.3 → cũng không. Với R, phiếu θ 0.5 đã gần tối ưu ở public |
| Hướng M — **M0 trần 1536** (2 seed: `linhmai24/zc-m0mt-1`, `zc-m0mts1-1`) | xong — **loại** | 0.941 / 0.938 (tb 0.9395; bộ chọn 0.949 / 0.955) | 0.586 / 0.864, 0.583 / 0.858 (gộp tb **0.742**; bộ chọn 0.758) | vs M0 0.934 / 0.752 (M0 + bộ chọn 0.944 / 0.767): public +0.55 nhưng held-out −1.0 (bộ chọn: −0.9) → loại theo §4; trả lời dài hơn làm ít bỏ hơn nhưng sai nhiều hơn ở held-out (khác hướng R, nơi 1536 không giảm held-out) |
| **Hướng M — M4 DPO + M0 trần 1536** (10/11 13:00 UTC) | **đang chạy** | | | **M4** (DPO theo phản hồi thực thi, `training/dpo.py` + `training/data/dpo_from_logs.py`): 871 cặp (721 câu train dev / paraphrase train / D3): chọn = đáp án đúng của chính súp (A / B) hoặc chương trình đã kiểm của câu (C; paraphrase: chương trình câu gốc khi kỳ không đổi, như D2), loại = văn bản sai của chính súp (phiếu sai nhiều nhất trước) hoặc văn bản bị cắt ở 512 token; LoRA DPO (β 0.1, + 0.2 × NLL câu chọn, lr 3e-5) trên súp M0, FSDP 4 GPU (`full_job.py --trainer dpo --base-adapters`); ứng viên súp + DPO × 1.0 / × 0.5 trên pub_full + pub_ho + tr_ho (`duylnng/zc-m4dpo-1`; đẩy lên lelynhan trước rồi huỷ: acc chỉ còn 0.37 h GPU — quy định quota mới); kiểm CPU trên mô hình tí hon (DPO, DPO + SFT, gộp). **M0 trần 1536** (đổi cách lấy mẫu, không luật; `configs/m0/m0_mt1536*.json`, 2 seed: `linhmai24/zc-m0mt-1`, `linhmai24/zc-m0mts1-1`) |
| **Hướng M — M7 + M2** (10/11 12:30 UTC) | **đang chạy** | | | **M7** (nhiều bộ sinh): từng thành viên WF1…WF4 × 0.35 làm bộ sinh riêng trên code M0 (`phuongnehe/zc-m7a-1`: WF1, WF2; `phuongnehe/zc-m7b-1`: WF3, WF4; pub_full + pub_ho + tr_ho) → gom mẫu với log M0 offline, phiếu / bộ chọn M6. **M2** (tự chưng cất có kiểm thực thi, `training/data/rft_from_logs.py`): mẫu đúng của chính súp M0 trên câu train dev / paraphrase train / D3 (prompt dựng lại bằng code M0 + rulebook public như lúc suy luận: gợi ý và ví dụ khớp 100 % log), 3,204 bản ghi (710 từ câu khó: phiếu tách / sai / bỏ; chương trình đúng chỉ có nhờ think / sửa / dài → viết lại dạng trả lời thẳng); 4 nhánh LoRA **trên súp M0** (`arms_job.py --base-adapters`), mỗi nhánh 710 câu khó + 490 câu dễ / D1; đánh giá súp + tb 4 nhánh × 1.0 và × 0.5 (`thnliu/zc-m2rft-1`). Bản sao dataset riêng: phuongnehe, thnliu. **Bộ chọn M6 lần 3 (đặc trưng bất biến, `selector_m0_inv.json`)**: M0 public tb 0.934 → **0.944**, held-out gộp 0.752 → **0.767**; chịu được gộp nhiều bộ sinh (`tests/pool_gens.py`) |
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

### Acc mới (user 11/11, có L4; token trong `~/.kaggle/tok_<username>`)

dinhthinh (còn 10.3 h), hieuho669 (14.4 h), huuhien95 (16.6 h), bichthuy93 (30 h), caohuyen74 (16.1 h) — acc dùng chung (đã có giờ dùng): chỉ đụng kernel `zc-*`; chưa có bản dataset riêng (zc-para, zc-final-adapter, zc-wg-adapters) → tạo khi giao việc.

### Quota GPU (`automation/quota.py`, 10/11 13:30 UTC; làm mới 14/11 00:00 UTC)

Quy định user 10/11: acc còn < 6 h → chỉ giao task mới khi task vừa số giờ còn lại (4×L4 ≈ 4 h quota / giờ tường).
Còn nhiều: duylnng 30 h (mới), mxuanquy 30 h (chưa có L4: cần accept rules), yeshenhi 29.9 h (chỉ T4), thnliu 28.3 h, linhmai24 27.3 h,
phuongnehe 25.9 h, loinguyentran 20.3 h, yuan1103 15.8 h, anderson1202 15.1 h. Thấp: synchan5704 4.7 h, lelynhan 0.4 h. Hết (0 h):
huyenco, mucnokieu, nguyenhoa99, nguynduylm, nhatlong1103, phatle1578, quynhle07, quynhtran35 (phiên đã chạy không bị dừng khi vượt 30 h).

## Quyết định của user (10/11)

- **`final_submission/submission.jsonl`** = file nộp public có PA cao nhất (hiện **0.982**: 988 đúng / 6 sai / 6 bỏ, lần chạy vòng 8 `p2r8j`);
  user nộp thử lên portal để đối chiếu; tự cập nhật bằng `automation/update_final_submission.py` sau mỗi lần chạy có PA cao hơn.

- Trước mắt **tiếp tục thêm luật tổng quát** (hướng R). Kế hoạch 2 có thêm **hướng M** (`docs/12_PLAN2.md` §6): xuất phát từ bản **không có
  luật P2** có public cao nhất (M0 = code `b093d46` + flow_final cũ: 0.939), thử song song nhiều cách cải tiến **mô hình** (mô hình phụ,
  fine-tune lại, dataset ngoài, chưng cất, DPO / RL…) đến khi public gốc = 1.0; không thêm luật ở hướng M.

## Quyết định của user (09/11)

- **Ưu tiên điểm public gốc** (đề private giống phong cách public): tiêu chí chọn chính = `pub_full` đủ 1000 câu (chạy 2 lần vì nhiễu); held-out (pub_ho + tr_ho) là rào chắn chống quá khớp (không được giảm quá nhiễu); vẫn chỉ sửa tổng quát.
- yeshenhi không thể có L4 (bỏ qua); cần thêm acc L4 thì báo user.
- Ngày 15/11 user tự build Docker, chạy `run_public_test.sh`, nộp portal.

## Kaggle

10/11: user cấp thêm 6 acc: **nguynduylm, phuongnehe, thnliu, linhmai24, lelynhan** (probe 4×L4 OK), **mxuanquy** (probe: không có GPU →
cần user accept rules `gemma-4-developer-agent` trên web bằng acc này). Hết quota W46: quynhle07, huyenco, nhatlong1103, quynhtran35,
phatle1578, mucnokieu, nguyenhoa99 (một phiên 4×L4 tính ~4× thời gian → ~5 lần eval / acc / tuần). `nguynduylm/zc-para`,
`nguynduylm/zc-final-adapter` tạo (bản riêng tư). Bài học: ổ đĩa đầy làm file viết dở thành rỗng (jobs jsel) → dọn scratch định kỳ.

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
