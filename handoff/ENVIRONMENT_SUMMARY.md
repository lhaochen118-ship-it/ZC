# Environment Summary — DataMate (cập nhật 16/11/2026 10:10 UTC)

*Cấu hình các môi trường dự án dùng. Bài học vận hành ở PDF 2; vận hành Kaggle chi tiết ở `docs/02_KAGGLE_OPS.md`.*

## 1. Container phát triển (Claude Code trên cloud)

| Mục | Giá trị |
|---|---|
| Máy | 4 nhân CPU, 15 GB RAM, **không GPU, không Docker daemon**; hạn mức đĩa riêng (~40 GB; `df` "Avail 0" = hết hạn mức) |
| Python | 3.13; pandas 3.x, numpy 2.x, openpyxl, python-calamine, kaggle CLI 2.2.5; trl 1.15 / peft 0.21 / transformers 5.17 / torch CPU (test huấn luyện trên mô hình tí hon) |
| PDF | `pip install markdown xhtml2pdf` (mất khi đổi container); `python3 automation/md2pdf.py <in.md> <out.pdf>` |
| Gói phải cài lại mỗi container | `pip install kaggle==2.2.5 xgrammar duckdb markdown xhtml2pdf` (`tests/test_grammar.py` cần xgrammar, `tests/test_sqlesc.py` cần duckdb; thiếu → `run_all.sh` dừng ở test đó) |
| vLLM | không có — test CPU dùng mô hình giả (`--fake-responder`) |
| Repo | `/home/user/ZC`, branch `claude/datamate-v3-0dny90` |
| Scratch | thư mục scratchpad của phiên (`runs/`, `cache_train/`, `cache_pub/`, `canon/`, worktree) — **mất khi sang khung chat mới**; bản lưu cần thiết ở `Process/` |
| Token Kaggle | `~/.kaggle/tok_<username>` (chmod 600); 15/11 có **19 acc** (18 × 4×L4 + `yeshenhi` T4, danh sách §6); xin lại user mỗi container; username từ token: `KAGGLE_API_TOKEN=... kaggle config view` |
| Tác vụ nền | trần 2 giờ (watcher `--max 115` rồi bật lại); container có thể khởi động lại giữa phiên → `automation/resume.py` |

Dựng lại sau khi mất container:
```bash
kaggle datasets download nhatlong1103/zc-2026 -p data/raw --unzip      # + data/raw/public_gt.jsonl (user gửi)
python3 src/ingest.py data/raw/train/train <S>/cache_train 4
python3 src/ingest.py data/raw/public_test/public_test <S>/cache_pub 4
python3 tests/build_canon.py data/raw/train/train <S>/cache_train <S>/canon      # bảng chuẩn (dữ liệu RFT / DPO)
for f in Process/train_data/wg/*.gz; do mkdir -p training/data/out/wg; gunzip -c $f > training/data/out/wg/$(basename ${f%.gz}); done
python3 tests/build_canon.py data/raw/public_test/public_test <S>/cache_pub <S>/canon_pub   # bảng chuẩn public (alias_values)
mkdir -p <S>/ext/vilexnorm && for s in train dev test; do curl -sL -o <S>/ext/vilexnorm/$s.csv https://raw.githubusercontent.com/ngxtnhi/ViLexNorm/main/data/$s.csv; done
# kế hoạch 3: Process/train_data/p3/*.gz -> training/data/out/{v3a,v3bm,all30,lrl0}/ (README Process/)
```

## 2. Kaggle (nơi chạy GPU)

| Mục | Giá trị |
|---|---|
| Máy | `NvidiaL4` → **4×L4** (22.5 GB / GPU, không NVLink), 48 CPU; không internet |
| Điều kiện L4 | `competition_sources: ["gemma-4-developer-agent"]` + acc đã accept rules trên web (push lỗi "accept this competition's rules" nếu chưa) |
| Mô hình | Kaggle Model `danbth/qwen3-5-9b/transformers/default/1` (bf16, FP8 online `--quant fp8`), kiểm sha256 |
| Thư viện | `nhatlong1103/zc-assets` (wheels vLLM 0.32), `yeshenhi/zc-train-wheels` (huấn luyện: unsloth, trl, peft); code đẩy theo kernel (`<acc>/zc-code`) |
| Dữ liệu | `nhatlong1103/zc-2026`; riêng mỗi acc: `zc-para`, `zc-final-adapter`, `zc-wg-adapters` (`Process/kaggle/datasets_by_account.txt`) |
| Thời gian | đánh giá đủ tập (pub_full + pub_ho + tr_ho = 4,199 câu) ~60–75 phút + cài 7 phút + gộp 2–10 phút; tập con sàng lọc ~35–40 phút; luồng nộp 1,000 câu ~22 phút; huấn luyện 4 nhánh LoRA 14.7M token ~5 h (1 nhánh / GPU, unsloth); **FSDP 4 GPU 1.72k tok/s** (16.7M token ≈ 2.8 h; 0.33 token / ký tự prompt full); quota thực ≈ 1.8–2.2 h / giờ tường (kế hoạch dùng 2.5) |
| Giới hạn | 2 phiên GPU + 5 CPU / acc; **30 h GPU / tuần / acc** (4×L4 ≈ 1.5–2.2 h quota / giờ tường); làm mới thứ Bảy 00:00 UTC; phiên commit ≤ 12 h |

### Tài khoản (13/11; quota: `python3 automation/quota.py`, ảnh chụp `Process/kaggle/quota_2026-11-13.txt`; làm mới 14/11 00:00 UTC)

Token hiện có (14): bichthuy93, caohuyen74, caotienminh1, dinhthinh, duckzuykobip, duylnng, hai1108, hieuho669, huuhien95, linhmai24,
loinguyentran, namhunhh, nguyenmanh0404 (L4 đều chạy được), nhatlong1103 (dữ liệu `zc-2026`, `zc-assets`, adapter nộp; hết quota).
Mỗi acc có `<acc>/zc-para`; `hieuho669`, `dinhthinh` có `zc-eval-adapters` (WG1). Acc cũ (11/11, nhiều acc hết quota) xem
`Process/kaggle/datasets_by_account.txt`.

Acc dùng chung với dự án khác: chỉ thao tác kernel `zc-*`. `automation/accounts.json`: danh sách acc không có L4.

## 3. Máy chấm (BTC) và Docker nộp

| Mục | Giá trị |
|---|---|
| Phần cứng | 2×RTX 5090 (32 GB), ≤ 6 giờ, `--network none`, `/data` chỉ đọc, kết quả `/result` |
| Image | `docker/Dockerfile`: `vllm/vllm-openai:v0.32.0`, pandas 2.3.3, numpy 2.1.3, openpyxl, python-calamine; trọng số gộp lúc build (`docker/merge_assets.py`, `soup.json`) |
| Chạy | `docker/predict.sh` → `src/predict_flow.py --config configs/flow_final.json` (ngân sách 20,400 s, gộp kết quả mỗi 20 s) |
| Hướng dẫn | `docker/SETUP_DOCKER.md` (`fetch_assets.sh` → `build.sh` → `run_public_test.sh` → `export_image.sh`), `docker/README_SOLUTION.md` |
| Đã kiểm | đường container trên Kaggle 4×L4 (0.986); build + chạy trên GPU thật: user làm 15/11 |

## 5. Bổ sung 14/11 (kế hoạch 5)

| Mục | Giá trị |
|---|---|
| Gói container | như §1 (`pip install kaggle==2.2.5 xgrammar duckdb markdown xhtml2pdf pandas openpyxl`); đáp án public: `python3 automation/rebuild_public_gt.py` (cần `git fetch --unshallow origin`) |
| Cache W (CPU) | `training/data/out/_ingest_{train,pub}` (pickle ingest; ~700 MB; tự dựng bởi `w_link.py` / `p_rows.py` / `link.py --cache`) |
| Dataset Kaggle mới | `nhatlong1103/zc-w-final` (adapter WL1, WJR1, WJB1), `{yuan1103,synchan5704}/zc-w-adapters` (WL1, WJB1) |
| Kaggle tự giải nén `.gz` trong dataset code (`--train-data`) | gửi file không nén |
| vLLM 0.32 + fp8 + LoRA | `enable_lora` chạy với Qwen3.5 khi adapter đã qua `training/merge.py vllm`; adapter thô nạp được nhưng không tác dụng |
| Thời gian đường container 1000 câu (4×L4) | W1 unit ~13 phút, gộp adapter ~1.5 phút / cái, W2 link ~8 phút, S ~40 phút, W3 ~8 phút → 56–75 phút |
| Watcher | `Process/tools/plan5/watch_loop.sh` (vòng lặp `watch_done.py` → `watch2.out`) + Monitor `tail -F` |


## 6. Bổ sung 15/11 (kế hoạch v2, chế độ H)

| Mục | Giá trị |
|---|---|
| Acc có token (19) | nhatlong1103, quynhle07, quynhtran35, nguyenhoa99, yuan1103, synchan5704, anderson1202, loinguyentran, yeshenhi (T4), phatle1578, mucnokieu, huyenco, nguynduylm, phuongnehe, thnliu, linhmai24, lelynhan, duylnng, caotienminh1 — quota lúc bàn giao: `Process/kaggle/quota_2026-11-15.txt` (nhiều nhất: caotienminh1, duylnng, lelynhan ~30 h; hết: quynhle07, synchan5704, nguynduylm) |
| Dataset adapter dùng chung | `<acc>/zc-ad3` (V3A, WL1, WJR1, WTB, WTS, S2S1, S2S2) trên anderson1202, caotienminh1, duylnng, huyenco, lelynhan, linhmai24, mucnokieu, phuongnehe, thnliu, phatle1578; `<acc>/zc-ad2` (V3A, WL1, WJR1) trên 11 acc khác |
| Dataset Docker | `nhatlong1103/zc-final-adapter` (S: V3A × 0.70), **`nhatlong1103/zc-w-v2`** (WL1, WJR1 dạng `<NAME>/adapter/`, sha256 `src/adapter_hashes_w.json`) |
| Thư viện huấn luyện | `yeshenhi/zc-train-wheels` (`--ks`), mô hình `danbth/qwen3-5-9b/transformers/default/1` (`--ms`) |
| Tokenizer Qwen cục bộ (đo độ dài bản ghi) | không có mô hình cục bộ; dùng `tokenizer.json` trong một adapter đã tải (`kaggle datasets download <acc>/zc-ad3` → `V3A/adapter/`) |
| Thời gian (4×L4) | đường container public chế độ H k16: 1.8–2.2 h; k24: +35 phút; huấn luyện FSDP 13M token: ~2.3 h (mb 2, max-len 5120, `expandable_segments`); bộ test CPU `tests/run_all.sh`: 30–40 phút |
| Watcher | `Process/tools/v2/watch_loop.sh` (vòng `watch_done.py --max 110` → `watch.out`) + Monitor `tail -F watch.out`; `run_kernel.py` poll `ZC_POLL_S` (150–180 s) để tránh 429 |
| Docker | `predict.sh` mặc định `configs/flow_v2_final.json` (chế độ H); image chép `training/merge.py`, `models/adapters/` (W) và đặt `ZC_ADAPTER_ROOT`; chưa build / chạy trên GPU thật |

## 7. Bổ sung 16/11 (P5: zero-shot, MoE, bản nộp)

| Mục | Giá trị |
|---|---|
| Acc có token (29 acc, 16/11) | danh sách tệp `~/.kaggle/tok_*` (xin lại user khi đổi container); quota lúc bàn giao: `Process/kaggle/quota_2026-11-16.txt` (mọi acc ≤ 5.5 h; làm mới 21/11 00:00 UTC). `yeshenhi` chỉ T4, `mxuanquy` chưa accept rules (không L4) |
| Mô hình Kaggle | 9B `danbth/qwen3-5-9b/transformers/default/1`; MoE bf16 `guohaoli4/qwen3-6-35b-a3b/transformers/bf16/1` (cắt E88 trong kernel); MoE FP8 chính thức tải từ HF trong kernel CPU (`stage_hf.py`, cần `--internet`) |
| MoE E88 + LoRA đã gộp FP8 | output kernel **riêng tư theo acc**: bichthuy93 (`zc-mgs7w-1` MS2×0.7, `zc-mgs2b-1` MS2, `zc-mgl-1` ML1, `zc-mgj2b-1` MJ2), caohuyen74 (`zc-mcs7/mcs10/mcl/mcj-1`), duckzuykobip (`zc-mgs8/mgl8/mgj8-1`); mount bằng `--ks` |
| Adapter MoE (nguồn) | `training/data/out/p5ad/{MS1,MS2,MS3,ML1,MJ1,MJ2}` (không commit); bản sao LFS `Process/adapters/moe_E88/{MS2,ML1,MJ2}` |
| Dataset Docker | 0.952: `nhatlong1103/zc-final-adapter` (V3A×0.70) + `nhatlong1103/zc-w-v2` (WL1, WJR1); 0.994: `nhatlong1103/zc-p-adapter` (WG1 + soup 0.35) |
| Bẫy môi trường | đĩa container ~30 GB khả dụng (đầy 3 lần: `ZC_OUT_PATTERN` + `--train-data` hard link); worker phiên khởi động lại ~mỗi vài giờ (mọi tiến trình nền chết → `automation/resume.py`, chạy lại `after_*.sh` — resume-safe); dataset 1 thư mục gốc bị Kaggle bỏ lớp ngoài (thêm `info/`) |
| Watcher | `Process/tools/p5_watch_loop.sh` → `Process/runs/p5/watch.out` (danh sách `active_runs.txt`) |
