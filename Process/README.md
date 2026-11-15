# Process/ — mọi thứ cần để tiếp tục mà không nằm sẵn trong code

Khung chat mới chạy trong container mới: scratch (log lần chạy, cache, worktree) và `training/data/out/` (dữ liệu huấn luyện,
`.gitignore`) **không còn**. Thư mục này giữ phần chuyển giao được (nén gzip); phần còn lại dựng lại theo cuối file.

| Đường dẫn | Nội dung | Dùng khi |
|---|---|---|
| `scores.csv` | điểm (PA theo đáp án cuối) của **188 lần chạy** kế hoạch 1–2 (kế hoạch 3–4: `scores_p3.csv`) trên mọi tập: pub_full, pub_ho, tr_ho, pub_ho400, tr_ho400, tr_dev_all, tr_dev_para, d3sel, pub_k0 + held-out gộp | so ứng viên mới với mốc cũ mà không tải lại gì |
| `runs/<run>/<result>/` (lưu bằng `automation/archive_runs.py`; 14/11 chỉ giữ: P (`finala/b`, `p2r17p*`, `p3pbase`), kế hoạch 3 (`rq5_*`, `lrl0*`, `pj`, v3a, `c1a/b`, `q3*`, `plrl*`), kế hoạch 4 (`e0h*`, `g0h*`, `g1*`, `pv*`, `pva3k16` = hệ nộp, huấn luyện v3b / v4 / WR30); log kế hoạch 1–2 đã xoá, còn trong git) | `logs.jsonl.gz` (mọi mẫu: chương trình, đáp án, cờ, phiếu), `submission.jsonl.gz`, `result.json`; `arms.json` / `train_metrics.json` của lần huấn luyện | phát lại phiếu / bộ chọn (`tests/pool_gens.py`), quét tín hiệu sạch cho kiểm tra mới, liệt kê câu sai (`tests/list_wrong.py`), dữ liệu RFT / DPO |
| `train_data/wg/` | dữ liệu huấn luyện kiểu WG: `d1_full` (2,119 câu train → chương trình đã kiểm), `d2_full` (3,818 paraphrase), `d2n_full` (5,419 câu nhiễu), `rft_hard` (710 mẫu đúng của câu khó) — prompt full | huấn luyện nhánh mới (`automation/jobs/wg.jobs`, `wg2r.jobs`: `--train-data wg`) |
| `train_data/p3/` | kế hoạch 3 (13/11): `v3a_mix` (SFT v3a thuần: D1 + RFT + phong cách kỳ / ViLexNorm, nhãn `<spec>`), `v3b_mix` (+ thẻ luật trong prompt, DR, phiên âm), `d1_all` / `d2_all` (30 DN, WR30), đầu vào sinh thẻ (`cases`, `alias_values`, `newterm_in`), thẻ quy nạp ứng viên | `training/data/build_v3b.sh`, `docs/16_PLAN3_LEARNED_RULES.md` |
| `train_data/v4/` | kế hoạch 4 (13/11): `v4_mix` (UD dùng hàm + RFT + DR + replay v3a), `ud`, `rft` — v4 không hơn v3a trên public | tái lập v4 |
| `kaggle/kernels.csv` | lần chạy → kernel id (`<acc>/zc-<slug>`) → trạng thái | tải lại output: `KAGGLE_API_TOKEN=$(cat ~/.kaggle/tok_<acc>) kaggle kernels output <acc>/<slug> -p <dir>` |
| `kaggle/plans/*.json` | plan `launch_many.py` đã dùng (acc, job, est_h, datasets) | mẫu khởi chạy lần sau |
| `kaggle/datasets_by_account.txt` | dataset `zc-*` trên từng acc (zc-para, zc-final-adapter, zc-wg-adapters …) | chọn acc có sẵn dữ liệu / adapter |
| `kaggle/quota_2026-11-13.txt` (cũ: `_11-11`) | quota lúc bàn giao 13/11 (làm mới 14/11 00:00 UTC) | lập kế hoạch (số thật: `python3 automation/quota.py`) |
| `tools/` | công cụ phân tích của phiên (sửa `$S` = scratch): `score_sets.py` (PA theo tập), `cmp_pure_p.py` (lỗi thuần so với P theo kỳ / thực thể / hàm), `period_share.py`, quét tín hiệu sạch `scan_*.py`, `publish.sh` (đăng dataset có `info/`) | phân tích lỗi, đăng adapter / checkpoint |

## Kế hoạch 5 (14/11, `docs/18_PLAN5_TWO_LORA.md`) — đã lưu

| Đường dẫn | Nội dung | Dùng khi |
|---|---|---|
| `scores_p5.csv` | điểm public mọi lần chạy / phán xử của kế hoạch 5 | so mốc |
| `train_data/p5/` | `wj_list*.jsonl.gz` (W3: list, `_aug` có tăng cường từ chối, `_r` có dòng rulebook = dữ liệu WJR1), `wj_point*`, `wj_both*` (WJB1), `p_rows.jsonl.gz` (dòng rulebook câu train theo bộ khớp P), `wl.jsonl.gz` (W2: WL1), `tr_val.logs.jsonl.gz` | huấn luyện lại W: `gunzip` vào `training/data/out/{wj,wl}/` (`--train-data wj` / `wl`) |
| `runs/p5_solver/` | log S của zlk / zlb / zlw, link (zero-shot, WL1), thẻ unit | phán xử lại offline (`src/judge.py --logs`) |
| `runs/p5_judge/<run>/<log>/<variant>/verdicts.jsonl.gz` | phiếu của mọi lần phán xử (ứng viên + lựa chọn từng mẫu) | chọn chính sách không cần GPU: `tests/replay_judge.py`, `tools/plan5/ens2.py`, `pol.py` |
| `runs/f5a`, `runs/f5b` | 2 lần chạy container (solver_logs, links, cards_unit, submission) | mốc container; phán xử lại |
| `runs/p5_train/<W>/` | `train_metrics.json`, `train.log.gz`, `adapter_config.json` của WJB1, WJL1, WJP1, WJA1, WJR1, WL1 | |
| `runs/p5_w1b2/` | W1 bind: tóm tắt + đầu ra thô | (thẻ đã lưu ở `src/refs/lrl/compile_public_b.jsonl`) |
| `tools/plan5/` | `pol.py` (lưới chính sách 1 biến thể), `ens.py` / `ens2.py` (kết hợp biến thể trên nhiều log), `err.py` (phân nhóm lỗi), `pool.py` (gộp log S), `an*.py` (phân tích lỗi), `fake_*.py` (mô hình giả cho smoke), `watch_loop.sh` | sửa đường dẫn `$S` / scratch ở đầu file |
| `kaggle/kernels.csv` (thêm 29 dòng), `kaggle/plans/plan_*.json` | kernel + plan của kế hoạch 5 | tải output: `kaggle kernels output <id>` |

**Adapter W (320 MB / cái): trong repo qua Git LFS — `Process/adapters/{WJR1,WL1,WJB1}/` (`git lfs pull`)**; bản Kaggle: dataset `nhatlong1103/zc-w-final` (WL1, WJR1, WJB1 — mỗi cái `<NAME>/adapter/`), bản sao
`{yuan1103,synchan5704}/zc-w-adapters` (WL1, WJB1); output kernel gốc ở `kaggle/kernels.csv`. Adapter S (V3A) như cũ.

## Khôi phục nhanh

```bash
S=<scratchpad>                                            # thư mục scratch của phiên mới
# log của một lần chạy -> thư mục như cũ (pool_gens / list_wrong / train_selector đọc logs.jsonl)
mkdir -p $S/runs/p2r15o && gunzip -c Process/runs/p2r15o/p2r15o/logs.jsonl.gz > $S/runs/p2r15o/logs.jsonl
# dữ liệu huấn luyện
mkdir -p training/data/out/wg && for f in Process/train_data/wg/*.gz; do gunzip -c $f > training/data/out/wg/$(basename ${f%.gz}); done
mkdir -p training/data/out/v3a && gunzip -c Process/train_data/p3/v3a_mix.jsonl.gz > training/data/out/v3a/v3a_mix.jsonl   # mô hình nộp (v3a)
```

## Không chuyển giao được (dựng lại)

| Thứ | Cách lấy lại |
|---|---|
| Token Kaggle (11 acc dùng 14/11: nhatlong1103, quynhle07, quynhtran35, nguyenhoa99, yuan1103, synchan5704, anderson1202, loinguyentran, yeshenhi (T4), phatle1578, mucnokieu) | xin user → `~/.kaggle/tok_<username>` (600); username: `KAGGLE_API_TOKEN=... kaggle config view` |
| Dữ liệu BTC + đáp án public | `kaggle datasets download nhatlong1103/zc-2026 -p data/raw --unzip`; `data/raw/public_gt.jsonl` do user gửi (`score_1.jsonl`) |
| Cache ingest (W: `training/data/out/_ingest_{train,pub}`) | `python3 training/data/w_link.py` / `p_rows.py` tự dựng; agent: | `python3 src/ingest.py <split> <S>/cache_{train,pub} 4`; `python3 tests/build_canon.py data/raw/train/train <S>/cache_train <S>/canon` |
| Adapter W kế hoạch 5 | đã chuyển giao: `Process/adapters/` (Git LFS, `git lfs pull`); Kaggle `nhatlong1103/zc-w-final` |
| Adapter (LoRA, 320 MB / nhánh) | trên Kaggle: `<acc>/zc-wg-adapters` (WG1–4), `nhatlong1103/zc-final-adapter` (WG1 nộp), output kernel `loinguyentran/zc-wg-1`, `bichthuy93/zc-wg2r-1` (WG5–8) |
| Paraphrase đánh giá | `<acc>/zc-para` (`para_pub.jsonl`, `para_train.jsonl`) |
| Log của lần chạy không có ở `runs/` | `kaggle kernels output` theo `kaggle/kernels.csv` (output kernel còn trên Kaggle) |
| PDF tooling | `pip install markdown xhtml2pdf` |

## Kế hoạch v2 (14/11 17:00 → 15/11, chế độ H) — đã lưu

| Đường dẫn | Nội dung | Dùng khi |
|---|---|---|
| `scores_v2.csv` | pub_full / pub_ind / pub_noind / held-out của **mọi** lần chạy v2; cột `config_or_dir`: `flow_*` = file container, `ensemble` / `list` / `point*` = file phán xử | so mốc (container với container) |
| `runs/v2/<run>/` | lưu bằng `automation/archive_v2.py`: `solver_logs.jsonl.gz` (mọi mẫu S), `submission.jsonl.gz`, `cards_unit.jsonl.gz` (hàm do mô hình viết), `links.jsonl.gz`, `judge/<variant>/verdicts.jsonl.gz`, `eval.log.tail.txt`; huấn luyện: `train_metrics.json`, `train.log.tail.txt`; sinh dữ liệu (tg*, uq): `cards_*`, `raw_*` | phân tích lỗi CPU (`tools/v2/`), phán xử lại offline, dùng lại hàm / link (`units_file` / `links_file`) |
| `train_data/v2/` | `wt2/` (dữ liệu WT v2: 1,036 train + 265 đo), `s3/` (`units_all`, `units_train` 1,012 hàm thầy, `units_unitq` 907, `uq/` 1,336 câu UQ, `uq_index.json`), `s3a/s3a.jsonl` (S″-A: 3,436 bản ghi), `contrast/` (1,005 cặp tương phản), `tg0/` (thuật ngữ mới), `wtprog_bal/` | `gunzip` vào `training/data/out/<dir>/` rồi `--train-data <dir>` |
| `kaggle/kernels_v2.csv` | lần chạy v2 → kernel id → trạng thái cuối | tải output: `kaggle kernels output <id> -p <dir>` (adapter WTB / WTS / S2S1 / S2S2 / S3A nằm trong output kernel huấn luyện) |
| `kaggle/quota_2026-11-15.txt` | quota 19 acc lúc bàn giao | lập lịch |
| `tools/v2/` | `score_run.sh` (chấm 1 lần chạy), `unit_use.py` / `unit_use2.py` / `unit_final.py` (mẫu gọi hàm cửa hàng: tỉ lệ đúng, theo hàm, đáp án cuối), `var_acc*.py` (theo biến thể / vòng), `pool.py` (gộp log), `score_judge.py`, `score_heldout_judge.py`, `watch_loop.sh`, `when_ready.sh`, `after_running.sh`, `fake_*.py` (mô hình giả cho smoke CPU) | sửa biến `S` (scratch) ở đầu file |

Dựng lại dữ liệu không lưu: `training/data/out/s2` (S′) = `python3 training/data/s2_build.py …` (đầu vào `tg0`, `contrast`); `wj2`, `wj2d`, `wt`, `wt0`
(nhánh đã loại) không cần.
