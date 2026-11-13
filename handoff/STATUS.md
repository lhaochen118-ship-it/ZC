# STATUS — trạng thái sống (đọc đầu tiên khi tiếp tục)

*Cập nhật 2026-11-13 08:30 UTC (bàn giao sang khung chat mới). Lịch sử: `Process/history/STATUS_full_until_2026-11-13.md` (đến 13/11),
`Process/history/STATUS_full_until_2026-11-11.md`; điểm mọi lần chạy: `Process/scores.csv`. Kế hoạch hiện tại: `docs/16_PLAN3_LEARNED_RULES.md`.*

## 1. Bản nộp hiện tại (chốt 12/11)

| Thành phần | Giá trị |
|---|---|
| Luồng | `configs/flow_final.json` = **P** (hybrid + `count_check` / `size_use_check` / `ticket_avg_check`) |
| Mô hình | Qwen3.5-9B + **WG1 × 0.35** (`nhatlong1103/zc-final-adapter`, khớp sha256 `src/adapter_hashes.json` — kiểm 13/11) |
| Public | 0.993 / 0.994 / 0.993 (3 seed), container 0.994; `final_submission/submission.jsonl` 0.994 |
| Held-out | gộp 0.7748; **sạch (đo 13/11, rulebook theo DN): `pub_ho400c` 0.6983, `tr_ho400c` 0.9227** — mốc so sánh kế hoạch 3 |
| Docker | `docker/` sẵn sàng; user build 15/11; hạn nộp **16/11 23:59 GMT+7** |

## 2. Kernel lúc bàn giao

| Kernel | Việc | Sau khi xong |
|---|---|---|
| `caotienminh1/zc-v3ap3-1` (07:08) | SFT v3a phần 3 (checkpoint-51 → 77) | nếu đã xong: adapter `results/V3A/adapter` trong output (kiểm `train_metrics.json`: 77 bước) → STATUS §4 bước 2; nếu partial: `automation/chain_next.py` sang acc còn quota |

Routine của phiên cũ (`trig_018LeLfkcTwsLDDxbbaUvUNB`, `trig_01X66grTP7iy1Hv5p6sCCtFv`) **đã tắt** — phiên mới tạo routine riêng.

## 3c. Tiến độ P3 (kế hoạch 3, `docs/16_PLAN3_LEARNED_RULES.md`, user duyệt 13/11)

| Gói | Trạng thái (13/11 08:30 UTC, bàn giao) |
|---|---|
| WP3-00 sửa đo + dữ liệu v3 | rulebook theo DN khi đo; **`pub_ho_clean` 1171 / 1373, `tr_ho_clean` 1711** (Qwen tự chấm tương đương; P: `pub_ho` 0.629 → sạch 0.727); `<spec>` từ chương trình; dữ liệu phong cách 5368 (kỳ viết tắt 19.1 %, ViLexNorm); mix v3a 4919 ≈ 12M token; DR / `tr_newterm` (chờ newterm) |
| WP3-01 lớp luật | `src/lrl.py` (thẻ, khớp có dấu cho trigger ngắn, lọc trigger chung theo tần suất 2 %, `# RULES`, check → lượt sửa), `src/lrl_gen.py`, `tests/lrl_accept.py`, DC, đường container (`predict_flow.prepare_lrl`); `tests/test_lrl.py` PASS; Q2 chạy trọn trên CPU |
| WP3-02 sinh thẻ vòng 0 | **xong**: `src/refs/lrl/` compile public 111 / train 79, alias 619 / 1143, translit 85, induced 4 check (quét 175 → 4); 178 thẻ nội dung chờ A/B (`lrl_ab0`); DR 375 + `tr_newterm` 54 câu |
| WP3-03 SFT v3a | 16.7M token, 77 bước; chuỗi 3 acc: bichthuy93 (→ 22) → duckzuykobip (→ 51) → **`caotienminh1/zc-v3ap3-1` (→ 77, chạy 07:08, xong ~08:30)**; tiếp: đánh giá `automation/jobs/q3{a,l}{35,70}.jobs` |
| WP3-04 P refit 30 DN (WR30) | phần 1 `duckzuykobip/zc-wr30p1-1` **hỏng**: NCCL all-gather treo ở bước 0 (rank 2 chậm, chờ 30 phút) — thêm `ddp_timeout=600`; **dời sang đợt 14/11 00:00** (một acc, ~3.4 h, không cần chuỗi); mốc P trên tập sạch với đo đã sửa: `p3pbase` (hieuho669) |
| v3b (dữ liệu) | **xong**: `training/data/out/v3bm/v3b_mix.jsonl` 5569 bản ghi ≈ 19.5M token (v3a + thẻ trong prompt + 250 DR + 400 phiên âm); huấn luyện ở đợt 00:00 (`plan_w14.json`) |
| WP3-05 … 10 | chờ (STATUS §4) |

## 4. Việc tiếp theo (theo thứ tự)

1. Phiên mới: token, gói, cache, routine (`READING_GUIDE.md` §0).
2. **v3a**: tải output `caotienminh1/zc-v3ap3-1` → đăng adapter (`Process/tools/publish.sh <acc> <dir> zc-v3a-adapter`, layout `V3A/adapter/` +
   `info/`) trên các acc đánh giá → `q3a35/q3a70` (Q) + `q3l35/q3l70` (Q2) trên `pub_full,pub_ho400c,tr_ho400c` (mỗi job ~0.8 h, cần ≥ 2 h quota)
   → chấm `Process/tools/score_sets.py` → `Process/scores.csv`, `docs/16` §9.
3. **A/B thẻ nội dung**: `automation/jobs/ab0a.jobs` / `ab0b.jobs` (α theo kết quả bước 2) → `python3 tests/lrl_accept.py ab src/refs/lrl/induced_content_r0.jsonl <logs A> <logs B> <out>` → gộp vào `src/refs/lrl/induced.jsonl`.
4. **14/11 00:00 UTC**: `Process/kaggle/plans/plan_w14.json` (v3b linhmai24, WR30 bichthuy93; mỗi acc một nhánh trọn) + vòng học luật 1 với v3a
   (`training/data/lrl_cases.py` trên log v3a ở `tr_dev_para_*` → `lrl_gen --mode induce` → quét + A/B) → SFT v4 (2 seed) → đánh giá.
5. **Quyết định ~15/11 10:00 UTC** theo `docs/16` §5.4 (held-out gộp ≥ P và public ≥ 0.985), hoặc giữ P (+ WR30 nếu public ≥ 0.993);
   container + `docker/README_SOLUTION.md` + `final_submission/`; báo user build Docker.
6. Sau 16/11: H2 (`docs/16` §6).
