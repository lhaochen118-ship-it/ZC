# PDF 3 — Handoff (khung chat kế tiếp, từ 13/11/2026)

*Phạm vi: mục tiêu, tiến độ, quyết định đã thống nhất, vị trí mã nguồn / dữ liệu / kết quả, các bước tiếp theo. Kết quả khoa học:
PDF 1; kinh nghiệm kỹ thuật / vận hành: PDF 2; cấu hình máy: `environment_summary.pdf`; trạng thái sống: `handoff/STATUS.md`.*

## 1. Mục tiêu

- Zalo AI Challenge 2026, track **DataMate**: hỏi đáp tiếng Việt trên dữ liệu bảng bẩn; **PA = (đúng − sai) / N**. Ràng buộc: ≤ 15B tham số,
  offline, ≤ 6 h trên 2×RTX 5090, Docker. **Hạn nộp solution 16/11/2026 23:59 GMT+7** (Docker đóng băng; private chạy 20–23/11).
- **Mục tiêu hiện tại (user 13/11): kế hoạch 3** `docs/16_PLAN3_LEARNED_RULES.md` — mô hình thuần (không luật viết tay lúc suy luận) +
  **lớp luật do mô hình tự học**, nhắm public 1.0 mà held-out không giảm; sau 16/11 tiếp H2 (RL thưởng PA, mô hình chuẩn hoá câu).

## 2. Quyết định đã thống nhất (bản đầy đủ: `CLAUDE.md`)

| Ngày | Quyết định |
|---|---|
| 08/11 | Hybrid: mô hình viết chương trình cho mọi câu; luật chỉ làm ngữ cảnh / kiểm tra; cấm luật sinh đáp án |
| 10–12/11 | `final_submission/` = file public PA cao nhất (`automation/update_final_submission.py`); quota < 6 h phải đủ giờ; phiên Kaggle ≤ 12 h; **tốc độ trước, không tiết kiệm acc / GPU** (FSDP 4 GPU, chuỗi checkpoint qua acc, chia shard; không đủ giờ thì không chạy) |
| 12/11 | Mô hình thuần: chỉ giữ A (`ingest`, `automap`) + B (toolkit `dm`, `industry`); bỏ C (gợi ý luật) + D (kiểm tra luật); nhánh song song |
| **13/11** | **Duyệt kế hoạch 3 "thực hiện hết"**: WP3-00 → 10 đến 16/11, rồi H2. **Đổi bản nộp khi hệ mới có held-out gộp ≥ P và public ≥ 0.985** (§5.4). **WP3-04**: huấn luyện lại P trên đủ 30 DN, giữ nếu public ≥ 0.993 |
| bất biến | Không vá theo DN / câu; `data/raw/public_gt.jsonl` chỉ để chấm; **không luật nào học từ public**; không xem từng câu held-out; cổng kiểm tra trước khi chạy (`tests/run_all.sh` PASS) |

## 3. Tiến độ (13/11 ~08:30 UTC; chi tiết: `handoff/STATUS.md` §3c)

| Hạng mục | Trạng thái |
|---|---|
| Bản nộp | **P + WG1 × 0.35** (public 0.993–0.994, container 0.994); Docker sẵn (`docker/`); adapter `nhatlong1103/zc-final-adapter` khớp sha256 |
| Kiểm tra (`docs/15`) | không rò rỉ vào trọng số; số đo P lạc quan; tập đo sạch `*_clean`, mốc P sạch `pub_ho400c` 0.698 / `tr_ho400c` 0.923 |
| WP3-00 / 01 | xong phần code: `<spec>`, dữ liệu phong cách (kỳ, ViLexNorm, phiên âm), DR / `tr_newterm`, lớp luật `src/lrl.py`, `lrl_gen.py`, `lrl_accept.py`, đường container `predict_flow.prepare_lrl` |
| WP3-02 thẻ vòng 0 | xong: `src/refs/lrl/` (compile public 111 / train 79, alias 619 / 1143, translit 85, induced 4 check) |
| WP3-03 SFT v3a | chuỗi: 22 → 51 → **66 / 77** (checkpoint-66 trong output `caotienminh1/zc-v3ap3-1`); còn phần 4 (11 bước, `chain_next.py`) |
| WP3-04 WR30 | phần 1 hỏng (NCCL treo); **chưa chạy lại** (plan `Process/kaggle/plans/plan_w14.json`, một acc ~3.6 h) |
| WP3-05 … 10 | chưa: đánh giá v3a, A/B thẻ nội dung, SFT v3b, vòng học luật 1–2, v4, quyết định, container |

## 4. Vị trí

| Thứ | Ở đâu |
|---|---|
| Repo / branch | `lhaochen118-ship-it/ZC`, `claude/datamate-v3-0dny90`; hướng dẫn đọc: `READING_GUIDE.md` |
| Luồng nộp P | `src/predict_flow.py` → `agent.py --flow hybrid` (`hybrid.py`, `pipeline.py`, `hints.py`, `precheck.py`, `dm.py`, `industry.py`); `configs/flow_final.json` |
| Luồng thuần Q / Q2 | `configs/flow_pure.json` / `flow_pure_lrl.json` (`--pure 1 --lrl <thẻ>`) |
| Lớp luật | `src/lrl.py`, `src/lrl_gen.py`, thẻ `src/refs/lrl/*.jsonl`, nhận thẻ `tests/lrl_accept.py`, test `tests/test_lrl.py` |
| Dữ liệu huấn luyện kế hoạch 3 | bộ sinh `training/data/{spec_from_program,dp_style,dc_cards,dr_newterm,v3_mix,lrl_cases,lrl_collect}.py`, `build_v3b.sh`; bản nén `Process/train_data/p3/` (giải nén vào `training/data/out/{v3a,v3bm,all30,lrl0}/`) |
| Tập đo | `tests/splits/` (`pub_full`, `pub_ho400c`, `tr_ho400c`, `*_clean`, `tr_newterm*`, `lrl_ab0`) |
| Adapter | WG1 nộp: `nhatlong1103/zc-final-adapter`; v3a: chưa xong (checkpoint-66 ở output `caotienminh1/zc-v3ap3-1`, `results/V3A/ckpt/`); QS10: `namhunhh/zc-qs10-adapter`; khác: `Process/kaggle/datasets_by_account.txt` |
| Log / điểm | `Process/runs/<run>/` (nén), `Process/scores.csv`, kernel `Process/kaggle/kernels.csv`, plan `Process/kaggle/plans/`, job `automation/jobs/` |
| Công cụ phân tích | `Process/tools/` (`score_sets.py`, `cmp_pure_p.py`, `period_share.py`, quét tín hiệu `scan_*.py`, `publish.sh`) |

## 5. Các bước tiếp theo (theo thứ tự)

1. Phiên mới: xin user token Kaggle (`~/.kaggle/tok_<acc>`), cài `kaggle==2.2.5 xgrammar duckdb markdown xhtml2pdf`, tạo routine mỗi giờ
   (lời nhắc trong `CLAUDE.md`, đổi sang kế hoạch 3); dựng lại cache theo `Process/README.md`.
2. **v3a**: tải output `caotienminh1/zc-v3ap3-1` (checkpoint-66) → phần 4 bằng `automation/chain_next.py … --remaining-h 0.6` (11 bước) → đăng adapter
   (`Process/tools/publish.sh <acc> <dir> zc-v3a-adapter`, layout `V3A/adapter/` + `info/`) → 4 đánh giá `automation/jobs/q3{a,l}{35,70}.jobs`
   (`pub_full,pub_ho400c,tr_ho400c`) → so với P sạch.
3. **A/B thẻ nội dung** (`automation/jobs/ab0{a,b}.jobs`, tập `lrl_ab0`) → `tests/lrl_accept.py ab` → `src/refs/lrl/induced.jsonl`.
4. **14/11 00:00 UTC (quota mới)**: `Process/kaggle/plans/plan_w14.json` (v3b trên linhmai24, WR30 trên bichthuy93) + vòng học luật 1 với
   v3a (ca sai trên `tr_dev_para_*` → `lrl_gen --mode induce` → quét + A/B) → SFT v4 (2 seed) → đánh giá Q / Q2 / P + thẻ.
5. **Quyết định ~15/11 10:00 UTC** theo §5.4 (`docs/16`); hệ thắng → huấn luyện lại 30 DN nếu cần, đường container (cấu hình có `lrl`:
   `predict_flow` tự biên dịch rulebook private), cập nhật `docker/README_SOLUTION.md`, `final_submission/`; báo user build Docker.
6. Sau 16/11: H2 (`docs/16` §6).

## 6. Rủi ro cần nhớ

- Chỉ còn ~3 ngày: mỗi bước GPU phải song song nhiều acc; bản nộp P an toàn đã sẵn — không đổi khi chưa đạt §5.4.
- FSDP có thể treo ngẫu nhiên (đã có `ddp_timeout=600`); kernel nhiều job phải dùng `run_kaggle.py` có khoá pip.
- Thẻ luật sinh lúc suy luận phải tất định (temp 0); `tr_newterm` nhiễu, chỉ là chỉ báo.
