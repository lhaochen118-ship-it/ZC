# STATUS — trạng thái sống (đọc đầu tiên khi tiếp tục)

*Cập nhật 2026-11-13 13:20 UTC (phiên `session_01GwRWJKNnGuKMW6nYYuFVQh`, tiếp quản phiên `session_013HwGSx…` — phiên đó không còn
hoạt động: adapter v3a xong 12:39 mà 22 phút sau chưa được đăng, không commit sau 12:05). Lịch sử: `Process/history/STATUS_full_until_2026-11-13.md`,
`Process/history/STATUS_full_until_2026-11-11.md`; điểm mọi lần chạy: `Process/scores.csv`. Kế hoạch hiện tại: `docs/16_PLAN3_LEARNED_RULES.md`.*

**Phiên này:** token 29 acc (`~/.kaggle/tok_*`); routine mỗi giờ `trig_01N2pVq2uTpkcaBAh9CwpWuu` (:13) + một lần 14/11 00:06 UTC `trig_01UjCMuoEVQpTi4oJLL5pJb1`.
Quota 13/11 13:02 UTC (làm mới 14/11 00:00): phuongnehe 11.7, anderson1202 4.57, linhmai24 4.19, yuan1103 4.07, thnliu 3.37, dinhthinh 2.79,
synchan5704 2.51, huuhien95 2.18, nguyenmanh0404 2.05, bichthuy93 1.99, duckzuykobip 1.97, duylnng 1.90, caohuyen74 1.80, namhunhh 1.45,
caotienminh1 1.30, hieuho669 1.09, hai1108 0.94, loinguyentran 0.32; hết: huyenco, lelynhan, mucnokieu, nguyenhoa99, nguynduylm, nhatlong1103,
phatle1578, quynhle07, quynhtran35; mxuanquy / yeshenhi **không có L4** (thăm dò `automation/probe_l4.py` 13/11: 17 acc chạy được 4×L4 ngay).
**`data/raw/public_gt.jsonl` dựng lại không cần user:** `python3 automation/rebuild_public_gt.py` (từ `submissions/v30` ở commit `30fee95`, đã
chấm 1000 / 1000; tự kiểm: `final_submission` ra đúng 0.994 = 995 / 1 / 4 như `score.json`).

## 1. Bản nộp hiện tại (chốt 12/11)

| Thành phần | Giá trị |
|---|---|
| Luồng | `configs/flow_final.json` = **P** (hybrid + `count_check` / `size_use_check` / `ticket_avg_check`) |
| Mô hình | Qwen3.5-9B + **WG1 × 0.35** (`nhatlong1103/zc-final-adapter`, khớp sha256 `src/adapter_hashes.json` — kiểm 13/11) |
| Public | 0.993 / 0.994 / 0.993 (3 seed), container 0.994; `final_submission/submission.jsonl` 0.994 |
| Held-out | gộp 0.7748; **sạch (đo 13/11, rulebook theo DN): `pub_ho400c` 0.6983, `tr_ho400c` 0.9227** — mốc so sánh kế hoạch 3 |
| Docker | `docker/` sẵn sàng; user build 15/11; hạn nộp **16/11 23:59 GMT+7** |

## 2. Kernel đang chạy (13/11 15:20 UTC; runs ở scratch `<S>/runs`, watcher `automation/watch_done.py`)

| Kernel | Việc | Sau khi xong |
|---|---|---|
| ~~`phuongnehe/zc-v3b-1`~~ | **v3b hỏng 15:25**: đủ 87 / 87 bước (3 h 06) rồi pha reference-loss FSDP hết bộ nhớ ở một số rank → NCCL treo → không lưu adapter. Đã sửa `training/sft.py` (reference-loss chỉ khi 1 GPU) | **huấn luyện lại v3b ở đợt 14/11 00:00** (một acc, `automation/jobs/v3b.jobs`, ~3.5 h) → đánh giá Q / Q2 |
| `phuongnehe/zc-wr30p3b-1` | WR30 phần 3 **chạy lại bằng code đã sửa** (15:33) từ checkpoint-55 / 117 (bản `zc-wr30p3-1` code cũ đã huỷ: nếu train xong sẽ vướng đúng lỗi của v3b) | phần còn lại (nếu dừng theo hạn phiên) ở đợt 00:00: `chain_next.py <S>/runs wr30p3b WR30 <acc> wr30p4 automation/jobs/wr30r.jobs --train-data all30`; xong → đánh giá như P |
| `yuan1103/zc-plrlpf-1`, `linhmai24/zc-plrlho-1` | **P + LRL** (`configs/p3_plrl.json` = `flow_final` + lớp luật; WG1 × 0.35) trên `pub_full` / held-out 4 tập | `python3 Process/tools/score_p3.py plrl <2 output>`; so P (0.994 / 0.8403) |
| `duylnng/zc-lrl1i-1` | quy nạp thẻ vòng 1 (480 ca sai / bỏ của Q2 v3a × 0.70 trên `tr_dev_para_a/b`, 4 shard) | `tests/lrl_accept.py scan` (thẻ check) trên log `c1a/c1b` + log cũ; thẻ nội dung: lọc chặt rồi A/B |

**Kết quả đánh giá v3a (13/11, `Process/scores_p3.csv`)** — held-out gộp = PA theo số câu trên `pub_ho400c` (348) + `tr_ho400c` (375) + `pub_noind` (112) + `tr_newterm` (54):

| Hệ | pub_full | pub_ho400c | tr_ho400c | pub_noind | tr_newterm | held-out gộp |
|---|---|---|---|---|---|---|
| **P + WG1 × 0.35 (bản nộp)** | **0.994** | **0.6983** | **0.9227** | **0.9732** | **0.9074** | **0.8403** |
| Q v3a × 0.35 | 0.891 | 0.6092 | 0.8720 | 0.8661 | 0.8519 | 0.7672 |
| Q v3a × 0.70 | 0.909 | 0.5891 | 0.9173 | 0.8214 | 0.8704 | 0.7739 |
| Q2 v3a × 0.35 | 0.922 | 0.5862 | 0.8747 | 0.8750 | 0.8519 | 0.7604 |
| Q2 v3a × 0.70 | 0.928 | 0.6092 | 0.9040 | 0.8304 | 0.8519 | 0.7762 |

Lớp luật (Q2) cộng ~+2–3 điểm public, ~0 held-out; hệ thuần còn kém P ~7 điểm public, ~6.5 điểm held-out → chưa đạt quy tắc §5.4.
A/B thẻ nội dung vòng 0: không thẻ 0.864, có thẻ 0.854 → nhận 1 thẻ (`docs/16` §9). Ca vòng 1 (`c1a`): PA 0.815 trên `tr_dev_para_a`.

## 3c. Tiến độ P3 (kế hoạch 3, `docs/16_PLAN3_LEARNED_RULES.md`, user duyệt 13/11)

| Gói | Trạng thái (13/11 08:30 UTC, bàn giao) |
|---|---|
| WP3-00 sửa đo + dữ liệu v3 | rulebook theo DN khi đo; **`pub_ho_clean` 1171 / 1373, `tr_ho_clean` 1711** (Qwen tự chấm tương đương; P: `pub_ho` 0.629 → sạch 0.727); `<spec>` từ chương trình; dữ liệu phong cách 5368 (kỳ viết tắt 19.1 %, ViLexNorm); mix v3a 4919 ≈ 12M token; DR / `tr_newterm` (chờ newterm) |
| WP3-01 lớp luật | `src/lrl.py` (thẻ, khớp có dấu cho trigger ngắn, lọc trigger chung theo tần suất 2 %, `# RULES`, check → lượt sửa), `src/lrl_gen.py`, `tests/lrl_accept.py`, DC, đường container (`predict_flow.prepare_lrl`); `tests/test_lrl.py` PASS; Q2 chạy trọn trên CPU |
| WP3-02 sinh thẻ vòng 0 | **xong**: `src/refs/lrl/` compile public 111 / train 79, alias 619 / 1143, translit 85, induced 4 check (quét 175 → 4); 178 thẻ nội dung chờ A/B (`lrl_ab0`); DR 375 + `tr_newterm` 54 câu |
| WP3-03 SFT v3a | **xong 13/11 12:39**: 16.7M token, 77 / 77 bước; chuỗi 4 phần bichthuy93 → duckzuykobip → caotienminh1 → `anderson1202/zc-v3ap4-1`; adapter → `<acc>/zc-v3a-adapter` (8 acc); đánh giá `plan_v3aeval_run.json` đang chạy (§2) |
| WP3-04 P refit 30 DN (WR30) | phần 1 `duckzuykobip/zc-wr30p1-1` **hỏng**: NCCL all-gather treo ở bước 0 (rank 2 chậm, chờ 30 phút) — thêm `ddp_timeout=600`; **dời sang đợt 14/11 00:00** (một acc, ~3.4 h, không cần chuỗi); mốc P trên tập sạch với đo đã sửa: `p3pbase` (hieuho669) |
| v3b (dữ liệu) | **xong**: `training/data/out/v3bm/v3b_mix.jsonl` 5569 bản ghi ≈ 19.5M token (v3a + thẻ trong prompt + 250 DR + 400 phiên âm); huấn luyện ở đợt 00:00 (`plan_w14.json`) |
| WP3-05 đánh giá v3a | **xong** (bảng §2): Q2 v3a × 0.70 tốt nhất trong hệ thuần (public 0.928, held-out 0.776) — kém P; v3b chờ adapter |
| WP3-06 vòng học luật | vòng 0 A/B xong (1 thẻ nhận); vòng 1: quy nạp đang chạy (`lrl1i`) |
| WP3-07 … 10 | chờ (§4) |

## 4. Việc tiếp theo (theo thứ tự)

1. Phiên mới: token, gói, cache, routine (`READING_GUIDE.md` §0).
2. **v3a**: phần 4 (§2) → đăng adapter (`Process/tools/publish.sh <acc> <dir> zc-v3a-adapter`, layout `V3A/adapter/` +
   `info/`) trên các acc đánh giá → `q3a35/q3a70` (Q) + `q3l35/q3l70` (Q2) trên `pub_full,pub_ho400c,tr_ho400c` (mỗi job ~0.8 h, cần ≥ 2 h quota)
   → chấm `Process/tools/score_sets.py` → `Process/scores.csv`, `docs/16` §9.
3. **A/B thẻ nội dung**: `automation/jobs/ab0a.jobs` / `ab0b.jobs` (α theo kết quả bước 2) → `python3 tests/lrl_accept.py ab src/refs/lrl/induced_content_r0.jsonl <logs A> <logs B> <out>` → gộp vào `src/refs/lrl/induced.jsonl`.
4. **14/11 00:00 UTC**: `Process/kaggle/plans/plan_w14.json` (v3b linhmai24, WR30 bichthuy93; mỗi acc một nhánh trọn) + vòng học luật 1 với v3a
   (`training/data/lrl_cases.py` trên log v3a ở `tr_dev_para_*` → `lrl_gen --mode induce` → quét + A/B) → SFT v4 (2 seed) → đánh giá.
5. **Quyết định ~15/11 10:00 UTC** theo `docs/16` §5.4 (held-out gộp ≥ P và public ≥ 0.985), hoặc giữ P (+ WR30 nếu public ≥ 0.993);
   container + `docker/README_SOLUTION.md` + `final_submission/`; báo user build Docker.
6. Sau 16/11: H2 (`docs/16` §6).
