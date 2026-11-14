# Run plan — triển khai Proposed plan v2 trong 34 giờ (14 acc Kaggle)

*Lập 14/11/2026 17:00 UTC. Thiết kế: `handoff/04_PROPOSED_PLAN.md` (v2). **T0 = lúc user duyệt; lịch tính bảo thủ với T0 = 18:30 UTC 14/11** (duyệt sớm hơn thì phần dự trữ cuối dài thêm); đóng băng container
**T+33.5 = 16/11 04:00 UTC**. Giờ dưới đây tính theo T0 = 18:30; T0 lệch thì mọi mốc dịch theo.*

## 1. Tài nguyên

### 1.1 Acc và quota

Quota đo lúc 16:45 UTC 14/11 bằng `automation/quota.py`; làm mới 21/11. 4×L4 tốn ~1.8–2.2 giờ quota mỗi giờ tường (lập kế hoạch: 2.2).

| Acc | Quota còn | Giờ tường 4×L4 tối đa | L4 | Vai chính (2 phiên song song) | Giờ tường dự kiến |
|---|---|---|---|---|---|
| nhatlong1103 | 30.0 h | 13.6 | có | huấn luyện WT-b ∥ WT-s; WT v2; container cuối | 7.7 |
| nguynduylm | 30.0 | 13.6 | **probe** | S′ seed 1 ∥ S′ seed 2 (cả acc) | 11.0 |
| phatle1578 | 30.0 | 13.6 | **probe** | sinh dữ liệu thầy TG1 (2 phiên); lượt S của lưới ghép | 8.8 |
| mucnokieu | 30.0 | 13.6 | **probe** | TG1 (2 phiên); phán xử lại WJR2; biến thể phán xử | 8.0 |
| huyenco | 30.0 | 13.6 | **probe** | TG2 tương phản (2 phiên); sinh WT v2; S′ đánh giá | 7.8 |
| loinguyentran | 30.0 | 13.6 | có | huấn luyện W2′ ∥ WJR2; S′ đánh giá; xác nhận | 7.6 |
| phuongnehe | 27.85 | 12.7 | **probe** | đo WT (độ chính xác hàm; 614 câu WT-b); container cuối; thử Docker | 5.0 |
| anderson1202 | 20.03 | 9.1 | có | đo WT (WT-s, Qwen gốc có suy nghĩ); xác nhận | 4.1 |
| synchan5704 | 18.11 | 8.2 | có | đo WT (mốc W = S); đo B; S′ đánh giá | 4.4 |
| quynhle07 | 15.80 | 7.2 | có | H0a ∥ H0b | 3.0 |
| yuan1103 | 14.70 | 6.7 | có | F1 ∥ F2; xác nhận | 4.1 |
| quynhtran35 | 14.12 | 6.4 | có | H0c ∥ D1 | 3.0 |
| nguyenhoa99 | 13.59 | 6.2 | có | TG0 ∥ A1 | 3.0 |
| yeshenhi | 30.0 | (T4) | không (T4) | smoke 2×T4 cho thay đổi phần chỉ chạy trên Kaggle (`sft.py`); phiên CPU dự phòng | 0.5 |

- Tổng ~78 giờ tường phiên ≈ 172 giờ quota trên 334 giờ hiện có. Phần dư (~160 giờ) dùng cho chạy lại và các nhánh thêm ở §4.
- Đỉnh đồng thời: 8–10 phiên trên 4–5 acc.
- **Acc "probe"** chạy kernel 1 dòng ở T+0 (`zc-probe`, `--l4`). Acc nào không có L4 thì thêm vào `automation/accounts.json` (`no_l4`), và vai của nó
  chuyển sang acc kế trong danh sách thay thế: yuan1103 → anderson1202 → synchan5704. Không có 4 acc 30 h nào có L4 → S′ chỉ chạy 1 seed.

### 1.2 Cách dùng GPU (bắt buộc cho mọi job)

| Loại việc | Cách chia trong một phiên 4×L4 | Thiết lập |
|---|---|---|
| **Huấn luyện LoRA** (WT, W2′, WJR2, S′) | **FSDP FULL_SHARD trên cả 4 GPU** (`accelerate --config_file training/fsdp4.yaml`, `training/full_job.py`) | bf16, gradient checkpointing, LoRA r 32 / α 64 (S′: r 64 / α 128); `--max-len` = p99 độ dài đo trước (PDF 2 §9 #4); **`--group-by-length` (mới)**: giảm token đệm ~25–35 %; **dò micro-batch**: 3 bước trên các bản ghi dài nhất với mb 8 → 6 → 4, chọn mb lớn nhất có đỉnh ≤ 20.5 GB / GPU (~91 % của 22.5 GB, ≈ 82 / 96 GB cả phiên); batch hiệu dụng 64 bằng tích luỹ gradient; lr 2e-4 cosine, warmup 3 %; checkpoint mỗi 20 bước + tại hạn (`ZC_DEADLINE` = 12 h − 40 phút); `ddp_timeout` 600 |
| **Sinh dữ liệu thầy** (Qwen gốc có suy nghĩ) | **DP 4**: một vLLM / GPU, hàng đợi động theo DN (claim file, DN nhiều dòng trước) | fp8, `gpu_util` 0.92 (KV gần hết phần còn lại của 24 GB), `max_num_seqs` 256, `max_batched_tokens` 32768, prefix cache (phần chung: API + thẻ bảng DN đặt trước), suy nghĩ ≤ 10k token, 6 mẫu / dòng; **kiểm thực thi chạy song song trên CPU** của phiên (48 lõi, pool tiến trình) trong lúc GPU vẫn sinh (`--pipeline`) |
| **Đánh giá một cấu hình** | DP 4, hàng đợi động theo DN | như luồng nộp: fp8, `gpu_util` 0.92, `max_num_seqs` 128, `max_batched_tokens` 16384, pipeline; pha W (units / links / judge) chia shard cho 4 GPU |
| **Nhiều biến thể nhỏ** (phán xử lại log, độ chính xác hàm WT) | **mỗi GPU một biến thể** (`training/soup_job.py --parallel-gpus 4`): một acc 2 phiên = 8 biến thể | adapter gộp một lần / biến thể; mỗi biến thể cache riêng |
| **Không có L4** | 2×T4 (16 GB, không fp8): 9B bf16 cần TP 2 → chỉ smoke và job CPU; không giao việc chính | — |

**Giới hạn 12 giờ / phiên.** Mọi job dự kiến ≤ 6 giờ. Runner của `make_kernel.py` dừng job trước hạn và lưu output. S′ có checkpoint; nếu cần thì
nối sang acc khác bằng `automation/chain_next.py ... --resume-from auto`.

### 1.3 Quy tắc vận hành (PDF 2 §9)

- Mỗi acc **chỉ một lần đẩy** cho tới khi kernel trước báo RUNNING.
- `launch_many.py` với `est_h` cho mọi mục để kiểm quota và trần 11 giờ.
- Không gửi `.gz` vào dataset.
- Kiểm tỉ lệ parse sau lần chạy đầu của mọi định dạng mới.
- Watcher: `Process/tools/plan5/watch_loop.sh` chạy nền (timeout 7200000 ms). Container khởi động lại → `automation/resume.py`.
- Trước mọi lần đẩy: `tests/run_all.sh` ALL PASS + `automation/preflight.py` OK. Với job `training/*.py`: smoke trực tiếp
  `soup_job.py --fake-responder --subset` rồi `--no-preflight`.
- Sau **mỗi** lần chạy GPU có public gốc: chạy `automation/update_final_submission.py` (giữ file PA cao nhất). File của hệ H để ở `final_submission/h/`.
- Bảng tiến độ ghi ở `handoff/STATUS.md` ("Tiến độ v2") sau mỗi bước.

## 2. Lịch từng bước

Ký hiệu: `acc/s1`, `acc/s2` = phiên 1, 2 của acc. **CPU** = Claude viết code trong container (tuần tự). Mỗi bước GPU ghi: lệnh chính, đầu vào →
đầu ra, cổng.

### Đợt 0 — chuẩn bị và đo trung thực (T+0 → T+2.5; 18:30–21:00 14/11)

| # | Giờ | Ai | Việc | Đầu ra / cổng |
|---|---|---|---|---|
| 0.1 | T+0 → 0.5 | CPU | tải `nhatlong1103/zc-2026` → `data/raw`; `git lfs pull` (adapter W); `rebuild_public_gt.py`; `tests/run_all.sh`; tạo `configs/flow_p5_final_h.json` (= cuối + `"hide_api": "industry"`) | ALL PASS |
| 0.2 | T+0.1 | 5 acc probe | `zc-probe` 1 dòng `--l4` trên phatle1578, mucnokieu, huyenco, nguynduylm, phuongnehe | danh sách L4 thật → sửa §1.1 nếu cần |
| 0.3 | T+0.5 → 2 | quynhle07/s1 | **H0a**: `soup_job --runner predict` `flow_p5_final_h` trên public (đường container: S + W1 + units + W2 WL1 + W3 WJR1) | **`pub_full_H`** + `solver_logs` + phiếu |
| 0.4 | T+0.6 → 2.2 | quynhle07/s2 | **H0b**: `soup_job --runner eval` `flow_p5_final_h` trên `pub_ho400c,tr_ho400c,pub_hoind,tr_newterm` (1,122 câu) → log S; tiếp theo `--runner judge` WJR1 trên log đó | mốc held-out H |
| 0.5 | T+0.5 → 2.1 | quynhtran35/s1 | **H0c**: như H0b với `flow_p5_final` (còn `industry.py`) | mốc held-out thường (đo phần `industry.py` thổi lên) |
| 0.6 | T+0.6 → 2 | quynhtran35/s2 | **D1**: container `flow_p5_final` (WJR1) trên public | bản dự phòng; điểm container của đúng cấu hình cuối kế hoạch 5 |
| 0.7 | T+0.5 → 2 | nguyenhoa99/s1 | **TG0**: `lrl_gen.py --mode newterm` (Qwen gốc, DP 4, không suy nghĩ) trên mọi câu train có chương trình đã kiểm, **bỏ 6 DN held-out** (đầu vào: `Process/train_data/p3/newterm_in`) | ~2k (thuật ngữ, định nghĩa, câu′) cho nguồn (1) của WT và cho C |
| G0 | T+2.5 | CPU | chấm H0a–c, D1 (`Process/tools/score_sets.py`) | **mốc chính**: `pub_full_H`, held-out H / thường; ghi STATUS |

### Đợt 1 — A, F, mã sinh dữ liệu thầy (T+0.5 → T+5; 19:00–23:30)

| # | Giờ | Ai | Việc | Đầu ra / cổng |
|---|---|---|---|---|
| 1.1 | T+0.5 → 2 | CPU | **A**: `src/judge.py` `list` / `point` có ràng buộc (vLLM structured output `{"choice": int\|"none"\|"unanswerable"}`; tuỳ chọn `--choice-logprobs`: xác suất từ một lượt sinh 1 token); `tests/replay_judge.py` đọc dạng mới; test CPU mô hình giả | ALL PASS |
| 1.2 | T+2 → 3 | nguyenhoa99/s2 | **A1**: phán xử lại 4 log public (`f5a`, `f5b`, `pva3k16`, `g1v3apf2`) bằng WJR1, **mỗi GPU một biến thể**: list-ràng buộc, point-ràng buộc, list-xác suất, point-xác suất; tiếp đó cùng phiên chạy trên log H0a, H0b | tỉ lệ đọc được, PA mỗi log |
| G-A | T+3.2 | CPU | chính sách tổ hợp mới phát lại trên CPU (`ens2.py`) | nhận A nếu: đọc được ≥ 95 %, TB 4 log tăng, không log giảm, held-out H0b không giảm > 0.005 |
| 1.3 | T+2 → 2.75 | CPU | **F**: khối `# API` sinh từ `api_spec` (ẩn tên trong `hide_api`); cấu hình `flow_p5_final_hF.json` | ALL PASS, preflight OK |
| 1.4 | T+2.75 → 4 | yuan1103/s1 | **F1**: `pub_full_H` với F (đường eval + phán xử) | |
| 1.5 | T+2.8 → 4.2 | yuan1103/s2 | **F2**: held-out (4 tập) với F | G-F (T+4.3): nhận nếu ghép cặp không âm ở `pub_full_H` và mọi held-out |
| 1.6 | T+2.75 → 5 | CPU | **mã thầy TG1** (`src/lrl_gen.py`, chế độ mới `rows`): dòng nguồn (2) rulebook train × DN, (3) đối chứng: Qwen gốc sinh 3 biến thể định nghĩa / dòng (≥ 1 biến thể ngoài toolkit), (4) mục ngành tổng hợp 10–25 dòng / DN; mỗi dòng → 6 mẫu hàm (suy nghĩ ≤ 10k) → `units.verify` 13 kỳ → chọn khi ≥ 3 / 6 cùng chữ ký; với (3) đòi kết quả khác hàm gốc ≥ 1 kỳ; gắn cờ DN held-out; test mô hình giả | preflight OK |

### Đợt 2 — sinh dữ liệu thầy (T+5 → T+9; 23:30 14/11 – 03:30 15/11)

| # | Giờ | Ai | Việc | Đầu ra / cổng |
|---|---|---|---|---|
| 2.1 | T+5 → 8 | phatle1578/s1, s2 + mucnokieu/s1, s2 | **TG1**: 4 phiên × 4 GPU = 16 shard, hàng đợi động theo DN; ước ~7k dòng × 6 mẫu × ~6k token ≈ 250M token (~30M token / giờ / phiên) | `units_teacher.jsonl` (dòng, hàm được chọn, tỉ lệ đồng thuận, cờ held-out); thống kê tỉ lệ giữ theo nguồn |
| 2.2 | T+5 → 6.5 | CPU | **mã E** (`training/data/contrast.py`, mới): rút khe từ AST chương trình đã kiểm của câu train; prompt "đổi đúng một khe" cho Qwen gốc; kiểm (chạy được, đáp án khác); job kiểm ngược bằng S (4 mẫu) + phân xử Qwen gốc có suy nghĩ khi mâu thuẫn; mô hình tự liệt kê cách viết kỳ / viết tắt từ câu train | preflight OK |
| 2.3 | T+6.5 → 9 | huyenco/s1, s2 | **TG2**: sinh + kiểm cặp tương phản (s1: sinh; s2: kiểm ngược S + phân xử; nối qua dataset trung gian, hoặc cả hai trong một job pipeline nếu kịp) | `contrast.jsonl` (~3–5k cặp giữ lại) |
| 2.4 | T+6.5 → 8 | CPU | **mã WT**: `training/data/wt_build.py` (nguồn 1–4 → bản ghi SFT đúng prompt `unit_items`; tách DN held-out làm tập đo); `predict_flow` / `lrl_gen`: `units_model: "lora:WT@base"` hoặc `"lora:WT@solver"` (gộp WT vào Qwen gốc / vào S); `sft.py --group-by-length`, `--mb-probe`; job đo độ chính xác hàm (`tests/wt_accuracy.py`, mới) | ALL PASS; smoke huấn luyện 20 bước trên **yeshenhi 2×T4** (T+7.5 → 8) |
| 2.5 | T+8 → 8.5 | CPU | dựng dữ liệu WT từ TG0 + TG1 (đo p99 độ dài → `--max-len`), đăng dataset lên nhatlong1103 | ~5–7k bản ghi, ~15M token |

### Đợt 3 — huấn luyện các LoRA độc lập (T+8.5 → T+13; 03:00–07:30 15/11)

| # | Giờ | Ai | Việc | Đầu ra / cổng |
|---|---|---|---|---|
| 3.1 | T+8.5 → 11 | nhatlong1103/s1 | **T-WTb**: `full_job.py` FSDP 4 GPU, mô hình = Qwen gốc (không `--base-adapters`), 1 epoch, dò mb | adapter WT-b |
| 3.2 | T+8.6 → 11.1 | nhatlong1103/s2 | **T-WTs**: như trên, `--base-adapters V3A --base-weights 0.70` | adapter WT-s |
| 3.3 | T+8.5 → 10.5 | CPU | **mã B + dữ liệu W2′**: rulebook đối chứng (i) cho câu train (xoá / đổi / đổi tên / nhiễu; nhãn (a′) lấy từ TG1); `link.py` in thêm trường `status`; `w_link.py --status`; định tuyến trong `judge_phase` (điều kiện (i) / (ii), lượt chạy lại S cho (ii)) | ALL PASS |
| 3.4 | T+10.5 → 11 | CPU | **dữ liệu WJR2** = dữ liệu WJR1 + `contrast.jsonl` (dạng ứng viên phán xử) + câu đồng thuận bị định tuyến trên log train | |
| 3.5 | T+11 → 13 | loinguyentran/s1 | **T-W2′**: FSDP, chồng V3A × 0.70 (như WL1), ~12M token | adapter W2′ |
| 3.6 | T+11.1 → 13.6 | loinguyentran/s2 | **T-WJR2**: FSDP, chồng V3A × 0.70, ~15M token | adapter WJR2 |
| 3.7 | T+11 → 12.5 | CPU | **dữ liệu S′**: dựng lại bản ghi v3a theo bố cục mới (lớp ngữ nghĩa + `# SHOP FUNCTIONS` đặt đầu prompt) + đối chứng (a)(a′)(b)(c)(d) + một phần `contrast.jsonl`; `unanswerable` 3–5 %; bỏ 6 DN held-out | ~30M token; p99 độ dài |

### Đợt 4 — đo WT, chọn nhánh (T+11 → T+13; 05:30–07:30 15/11)

| # | Giờ | Ai | Việc | Đầu ra / cổng |
|---|---|---|---|---|
| 4.1 | T+11.2 → 11.9 | phuongnehe/s1 | **E-WT-u**: độ chính xác hàm trên dòng của 6 DN held-out, mỗi GPU một biến thể: WT-b, WT-s, Qwen gốc có suy nghĩ (10k), W = S | tỉ lệ hàm khớp nhãn trên 13 kỳ; tỉ lệ hàm qua kiểm |
| 4.2 | T+11.3 → 12.6 | phuongnehe/s2, anderson1202/s1, s2, synchan5704/s1 | **E-WT-e2e**: 614 câu (`pub_ind_H` + `pub_hoind_H`), DP 4, cùng S; 4 nhánh W: WT-b / WT-s / Qwen gốc có suy nghĩ / W = S (mốc) | PA ghép cặp mỗi nhánh so mốc |
| G-WT | T+12.8 | CPU | chọn nhánh | nhận WT tốt nhất nếu z ≥ 2 so W = S trên 614 câu **và** độ chính xác hàm held-out cao hơn; hoà → chọn nhánh nhanh hơn (không suy nghĩ); không nhánh nào qua → giữ W = S |

### Đợt 5 — S′, B, WJR2, WT v2 (T+12.5 → T+18.5; 07:00–13:00 15/11)

| # | Giờ | Ai | Việc | Đầu ra / cổng |
|---|---|---|---|---|
| 5.1 | T+12.5 → 18 | nguynduylm/s1, s2 | **T-S′** seed 1 ∥ seed 2: FSDP 4 GPU, chồng V3A × 0.70, r 64, ~30M token, checkpoint mỗi 20 bước | adapter S′1, S′2 |
| 5.2 | T+13.7 → 14.7 | mucnokieu/s1 | **E-J**: phán xử lại bằng WJR2 (chính sách A) trên 4 log public + H0a + H0b, mỗi GPU một nhóm log | G-J (T+15): nhận WJR2 theo quy tắc log cố định (`04` §5.1) |
| 5.3 | T+13.2 → 14.7 | synchan5704/s2 | **E-B**: W2′ trên câu của 6 DN held-out (precision "thiếu định nghĩa", so nhãn đối chứng); W2′ nối luật trên `pub_full_H` (log H0a) + held-out (log H0b); định tuyến; lượt chạy lại S cho câu (ii); phán xử | G-B (T+15): precision ≥ 0.9; số câu đồng thuận sai giảm; câu hỏng ≤ 1/3 câu sửa; held-out không giảm |
| 5.4 | T+13 → 14.2 | huyenco/s1 | **WT v2 (sinh)**: WT tốt nhất viết hàm cho dòng (3)(4) của DN train, 6 mẫu; giữ dòng mà WT bất đồng với nhãn thầy hoặc thầy chưa có nhãn, kiểm 13 kỳ | dữ liệu WT v2 (~1–2k bản ghi khó) |
| 5.5 | T+14.4 → 15.6 | nhatlong1103/s1 | **T-WTv2**: học tiếp từ WT v1 (`--resume-from` adapter, ~4M token) | WT v2; nhận khi độ chính xác hàm held-out tăng (đo lại 4.1 trên mucnokieu/s2 ở T+15.7, 30 phút) |
| 5.6 | T+15 → 16.5 | phatle1578/s1, s2 | **K-base**: một lượt S cho cấu hình ghép (S + F nếu nhận + WT được chọn + W2′ links) trên `pub_full_H` (s1) và held-out (s2) | log S cố định cho lưới phán xử |
| 5.7 | T+16.5 → 17.3 | mucnokieu/s2 | **lưới phán xử** trên log K-base (mỗi GPU một biến thể): WJR1 + A / WJR2 + A / WJR2 + A + B / WJR2 + A + B + ngưỡng thay thế | chọn chính sách phán xử (log cố định, cả held-out) |

### Đợt 6 — đánh giá S′ và chọn cấu hình (T+18 → T+21; 12:30–15:30 15/11)

| # | Giờ | Ai | Việc | Đầu ra / cổng |
|---|---|---|---|---|
| 6.1 | T+18.2 → 19.8 | huyenco/s2, loinguyentran/s1 | S′1, S′2 trên `pub_full_H` (cấu hình K với S′ thay S; phán xử theo 5.7) | |
| 6.2 | T+18.3 → 20 | synchan5704/s1, loinguyentran/s2 | S′1, S′2 trên held-out (4 tập) | G-S′ (T+20.2): nhận S′ nếu ghép cặp z ≥ 2 so K-base, hai seed cùng dấu, mọi held-out không giảm > 1 SE; độ nhạy rulebook ≥ 0.95 trên DN held-out |
| G-final | T+20.5 | CPU | chọn cấu hình nộp theo `04` §5: `pub_full_H` cao nhất trong các cấu hình qua held-out; viết `configs/flow_v2_final.json` | cấu hình cuối |

### Đợt 7 — xác nhận, đường container (T+20.5 → T+24; 15:00–18:30 15/11)

| # | Giờ | Ai | Việc | Đầu ra / cổng |
|---|---|---|---|---|
| 7.1 | T+20.7 → 22.2 | yuan1103/s1, anderson1202/s1 | **đường container × 2** `flow_v2_final` trên public (`soup_job --runner predict`, lưu solver_logs / units / links / phiếu); đo thời gian 1000 câu | hai lần lệch ≤ 0.03; thời gian × (2×RTX 5090 / 4×L4) ≤ 6 h cho kích thước private dự kiến |
| 7.2 | T+20.8 → 22.4 | loinguyentran/s1 | lượt thứ hai (seed khác) trên held-out | xác nhận held-out không giảm |
| 7.3 | T+20.8 → 22.2 | nhatlong1103/s2 | tham chiếu: `flow_v2_final` **còn** `industry.py` trên public | chỉ để báo cáo |
| 7.4 | T+22.5 | CPU | `update_final_submission.py`; `final_submission/h/` = bản H tốt nhất; ghi điểm | |

### Đợt 8 — đóng gói và tài liệu (T+22.5 → T+31; 17:00 15/11 – 01:30 16/11)

| # | Giờ | Ai | Việc | Đầu ra / cổng |
|---|---|---|---|---|
| 8.1 | T+22.5 → 24.5 | CPU | đăng adapter cuối (S′ hoặc V3A, WT, W2′, WJR2) vào `nhatlong1103/zc-w-final` (bản 2) kèm sha256 (`src/adapter_hashes.json`); `docker/`: `fetch_assets.sh`, gộp theo pha cho 4 adapter, `predict.sh` → `flow_v2_final.json`; `README_SOLUTION.md` (kiến trúc, tham số ≤ 15B, không luật viết tay ngành) | |
| 8.2 | T+24.5 → 26 | phuongnehe/s1 | thử bố cục Docker trên Kaggle (`soup_job --runner predict` với đúng thư mục assets / adapter của image) | điểm khớp 7.1 ± 0.01 |
| 8.3 | T+26 → 30 | CPU | `handoff/STATUS.md`, PDF 1 (§9 kết quả v2), PDF 2 (lỗi + kỹ thuật mới), PDF 3, `READING_GUIDE.md`; lưu log / phiếu vào `Process/runs` | bàn giao |
| — | T+30 → 33.5 | — | dự trữ: chạy lại khi phiên chết / hàng đợi; sửa Docker | đóng băng 04:00 UTC 16/11 |

## 3. Cổng quyết định (tóm tắt)

| Cổng | Giờ | Câu hỏi | Nếu không qua |
|---|---|---|---|
| G0 | T+2.5 | `pub_full_H` và held-out thật là bao nhiêu? | — (chỉ đo) |
| G-A | T+3.2 | phán xử có ràng buộc tốt hơn? | giữ tổ hợp WJR1 cũ |
| G-F | T+4.3 | API tự sinh không âm? | chỉ thêm dòng thiếu vào cuối `API_DOC` |
| G-WT | T+12.8 | WT hơn W = S có ý nghĩa? | giữ W = S (hoặc Qwen gốc có suy nghĩ nếu nó qua) |
| G-J / G-B | T+15 | WJR2 / định tuyến có ích, held-out không giảm? | tắt bằng cấu hình |
| G-S′ | T+20.2 | S′ hơn S, 2 seed cùng dấu? | giữ S (V3A × 0.70) |
| G-final | T+20.5 | cấu hình nộp | mọi thứ trượt → `flow_p5_final_h` + A (nếu qua) |

**Cần user quyết (một lần, trước G-final):** bản nộp cuối chạy chế độ H (ẩn `industry.py`) như quyết định 13/11 "không luật viết tay"
(**đề xuất**), hay chế độ thường (public cao hơn, nhưng dựa vào hàm ngành viết tay).

## 4. Nhánh thêm khi còn quota (không nằm trên đường găng)

Theo thứ tự giá trị. Mỗi nhánh ≤ 1 phiên, chỉ chạy khi acc của nó còn ≥ 2.5 × est_h quota.
1. WT-b seed 2: giảm rủi ro chọn nhánh do nhiễu.
2. WJR2 không có `contrast.jsonl`: quy công cho E.
3. D kênh thứ hai trên log K-base.
4. S′ seed 3.

## 5. Thứ tự cắt khi trễ

1. Cắt trước: D → nhánh thêm §4 → WT v2 (5.4–5.5) → S′ seed 2 → toàn bộ S′ (C) → B (giữ WJR2 + A) → E (WJR2 = WJR1 + câu định tuyến).
2. **Không cắt:** H0, A, G (WT), cổng held-out, đường container × 2, Docker.
3. Mỗi cổng trễ quá 2 giờ so lịch → quyết định cắt ngay, ghi lý do vào STATUS.

## 6. Nhật ký chạy và quyết định (UTC)

**14/11 17:00** user duyệt v2, chế độ H cho bản nộp, chạy tự động. Token 14 acc lưu `~/.kaggle`. Probe: phatle1578, mucnokieu, huyenco,
nguynduylm, phuongnehe đều có 4×L4. Dataset `zc-ad2` (V3A + WL1 + WJR1, sha V3A khớp `src/adapter_hashes.json`) đăng lên 13 acc.

**17:05 – rò rỉ chế độ H phát hiện và sửa.** 14 / 125 thẻ W1 public (`compile_public_b.jsonl`) có gợi ý code gọi hàm `industry.py`
(`n_vehicles`, `n_tickets`, `shipping_total`…): nếu chỉ đặt `hide_api` thì hàm viết tay vẫn lọt vào prompt qua thẻ / link / phán xử. Sửa:
`ZC_HIDE_API` xuất cho mọi pha, `lrl.strip_hidden` bỏ gợi ý / check gọi tên bị ẩn (giữ định nghĩa do mô hình viết), W1 không được gắn tên
bị ẩn. `tests/test_mode_h.py`.

**17:10 – đợt 0 + A + TG0 đẩy:** h0a, h0b, h0c, d1, tg0, a1. TG0 xong 17:20: 1,446 / 2,681 thuật ngữ mới.

**17:20 – F:** `API_DOC` chỉ thiếu 3 hàm công khai (next_n_days, lunar_to_solar, purchased_qty_base) → F = nối khối sinh từ `api_spec`
vào cuối (giữ tiền tố đã huấn luyện), không thay toàn bộ. F1 / F2 đẩy.

**17:25 – TG1 (4 phiên), TG1n, TG2 đẩy.** TG1n lần 1 lỗi (thư mục `g<k>` chưa tạo khi bỏ bước rowgen) → sửa + smoke đúng trường hợp, đẩy lại.

**17:40 – quyết định thiết kế:**
- **B rút gọn (B-lite):** bỏ huấn luyện W2′ (nhãn "thiếu định nghĩa" cần phán đoán chủ quan, không kiểm được bằng thực thi). Thay bằng thí
  nghiệm `bl1`: bộ phán xử đã huấn luyện xem lại **mọi** câu (kể cả đồng thuận) ở dạng `point+c` / `point+think`; ngưỡng ghi đè chọn trên log
  held-out (không chọn trên public).
- **C – bỏ loại "deleted":** xem mẫu dựng được: thuật ngữ mới do mô hình đặt phần lớn trong suốt ("doanh thu thuần"…; S đã đạt tr_newterm
  0.85 khi không có định nghĩa) → nhãn Unanswerable khi xoá dòng sẽ dạy S′ từ chối sai. S′ giữ: replay (+ hàm cửa hàng không liên quan),
  term (+ dòng nhiễu), term_fn (lời gọi hàm WT kiểm bằng thực thi), contrast / style.
- **Huấn luyện:** WT-b giữ đường đã chứng minh (mb 4); WT-s bật `--group-by-length` + mb 8 để đo lợi tốc độ (nếu lỗi thì chỉ mất nhánh kiểm tra).

**17:45 – lỗi đo held-out phát hiện và sửa.** H0c xong nhưng chỉ chấm 206 / 1,122 câu: các tập paraphrase (`pub_ho400c`, `pub_hoind`, phần
paraphrase của `tr_ho400c`) cần dataset `<acc>/zc-para`, plan quên mount; `eval_flow` chỉ dừng khi thiếu toàn bộ. Huỷ h0b, f2, ewts, ewbt
(`kaggle kernels delete`), đẩy lại h0b2 / h0c2 / f2b / ewts2 / ewbt2 có `zc-para`; `eval_flow` nay dừng hẳn khi thiếu > 10 % câu.
TG2 lần 1 lỗi: engine vLLM thứ hai trong cùng tiến trình (bước kiểm ngược) không có VRAM → một engine cho cả hai bước, lưu kết quả bước 1
trước; đẩy lại tg2b.

**18:05 – G-A: A loại.** Phán xử lại 4 log public bằng WJR1: biến thể có ràng buộc đọc được 100 % (list+c 470/470, point+c 1375/1375 trên
f5a; không ràng buộc 105/470, 479/1375) nhưng PA từng biến thể không tăng (point_c 0.939 vs point 0.928 trên f5a, nhưng 0.935 vs 0.939 trên
f5b…), tổ hợp: list_c+point_c+point_think m0.5 TB 0.946; list_c+point_think 0.950–0.952 vs mốc chạy lại list+point+point_think m0.5 TB
0.951. **Phát hiện:** đầu ra "không đọc được" của bộ phán xử không suy nghĩ thực chất là phiếu trắng của một bộ phán xử yếu; ép chọn chỉ thêm
nhiễu. Đòn bẩy phán xử nằm ở chất lượng (huấn luyện WJR2, point+think), không ở định dạng. Cấu hình giữ tổ hợp WJR1 cũ.

**18:28–19:05 – Kaggle API 429 (rate limit) cho mọi acc.** 17 tiến trình theo dõi hỏi trạng thái mỗi 30 s từ cùng một IP (proxy dùng chung)
→ từ ~18:25 mọi lệnh `kaggle` trả 429; không thấy kernel nào xong dù đã chạy > 1.5 h. Kernel trên Kaggle không bị ảnh hưởng. Sửa:
`run_kernel.py` hỏi kernel đang chạy mỗi ~150 s (`ZC_POLL_S`, jitter), lùi 4–6 phút khi gặp 429; tạm dừng mọi watcher, một vòng kiểm tra
thưa (3 phút) chờ API mở lại rồi gắn lại (`automation/resume.py`). Bài học (PDF 2): số watcher × tần suất phải dưới ngưỡng của IP chung.

**19:45 – G0 / G-F / nhánh Qwen gốc.** Mốc trung thực chế độ H: H0a 0.863, H0a2 0.863 (chế độ thường D1 0.952). Held-out H (bỏ phiếu):
pub_ho400c 0.581, tr_ho400c 0.901, pub_hoind 0.455, tr_newterm 0.852 (thường: 0.644 / 0.912 / 0.704 / 0.815). **F loại**: public −0.016
(z −1.6 so với trung bình hai seed, thua cả ở cửa hàng không ngành: khối API mới làm lệch tiền tố S đã học), held-out lẫn lộn (pub_hoind
+0.044, pub_ho400c +0.023, train 0). **Qwen gốc có suy nghĩ làm bộ viết hàm (EWBT2)**: 2 hàm qua kiểm, 614 câu kém W = S → không dùng lúc
suy luận; chỉ làm thầy sinh dữ liệu offline (TG1 có 6 mẫu, đồng thuận ≥ 3).

**20:00 – TG1 hỏng vì chế độ suy nghĩ (lặp lại lỗi kế hoạch 4).** tg1p0: Qwen gốc suy nghĩ lặp vòng ("Wait, I need to check…") không đóng
`</think>` trong 7–8k token ở 304 / 447 mẫu → giữ 7 / 243 dòng; bước rowgen chỉ ra 3 / 240 dòng. Đây đúng là G0b của kế hoạch 4 (317 / 389
mẫu ở 4,096 token) — lẽ ra phải biết trước. Huỷ p1–p3 và TG1n. Sửa: thầy **không suy nghĩ** + JSON có ràng buộc (rowgen: mảng
`ROWGEN_SCHEMA`; unit: `UNIT_SCHEMA`), 8 mẫu / dòng, đồng thuận ≥ 3, gộp dòng thuật ngữ mới vào cùng job (TG1b, 4 phiên). Thêm nguồn nhãn
không phụ thuộc thầy: `training/data/wt_from_programs.py` biến chương trình đã kiểm (đúng theo đáp án train) của mỗi thuật ngữ TG0 thành hàm
theo kỳ `p` (thay lời gọi kỳ bằng `p` trên AST), giữ khi gọi lại cho đúng đáp án đã kiểm.
**S′ đẩy không chờ TG1** (S2S1 / S2S2 trên nguynduylm, 23.1M token: replay + thuật ngữ mới trong `# RULES` + 1,005 cặp tương phản / cách viết).
**WJR2D** (quynhle07) = LoRA chồng trên V3A × 0.70 + WJR1, 1 epoch trên 1,738 bản ghi tương phản + 1,000 bản ghi WJR1.
