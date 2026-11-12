# Environment Summary — DataMate (cập nhật 11/11/2026)

*Cấu hình các môi trường dự án dùng. Bài học vận hành ở PDF 2; vận hành Kaggle chi tiết ở `docs/02_KAGGLE_OPS.md`.*

## 1. Container phát triển (Claude Code trên cloud)

| Mục | Giá trị |
|---|---|
| Máy | 4 nhân CPU, 15 GB RAM, **không GPU, không Docker daemon**; hạn mức đĩa riêng (~40 GB; `df` "Avail 0" = hết hạn mức) |
| Python | 3.13; pandas 3.x, numpy 2.x, openpyxl, python-calamine, kaggle CLI 2.2.5; trl 1.15 / peft 0.21 / transformers 5.17 / torch CPU (test huấn luyện trên mô hình tí hon) |
| PDF | `pip install markdown xhtml2pdf` (mất khi đổi container); `python3 automation/md2pdf.py <in.md> <out.pdf>` |
| Gói phải cài lại mỗi container | `pip install kaggle==2.2.5 xgrammar` (`tests/test_grammar.py` cần xgrammar; thiếu → `run_all.sh` dừng ở test_grammar) |
| vLLM | không có — test CPU dùng mô hình giả (`--fake-responder`) |
| Repo | `/home/user/ZC`, branch `claude/datamate-v3-0dny90` |
| Scratch | thư mục scratchpad của phiên (`runs/`, `cache_train/`, `cache_pub/`, `canon/`, worktree) — **mất khi sang khung chat mới**; bản lưu cần thiết ở `Process/` |
| Token Kaggle | `~/.kaggle/tok_<username>` (chmod 600), 24 acc; xin lại user mỗi container; username từ token: `KAGGLE_API_TOKEN=... kaggle config view` |
| Tác vụ nền | trần 2 giờ (watcher `--max 115` rồi bật lại); container có thể khởi động lại giữa phiên → `automation/resume.py` |

Dựng lại sau khi mất container:
```bash
kaggle datasets download nhatlong1103/zc-2026 -p data/raw --unzip      # + data/raw/public_gt.jsonl (user gửi)
python3 src/ingest.py data/raw/train/train <S>/cache_train 4
python3 src/ingest.py data/raw/public_test/public_test <S>/cache_pub 4
python3 tests/build_canon.py data/raw/train/train <S>/cache_train <S>/canon      # bảng chuẩn (dữ liệu RFT / DPO)
for f in Process/train_data/wg/*.gz; do mkdir -p training/data/out/wg; gunzip -c $f > training/data/out/wg/$(basename ${f%.gz}); done
```

## 2. Kaggle (nơi chạy GPU)

| Mục | Giá trị |
|---|---|
| Máy | `NvidiaL4` → **4×L4** (22.5 GB / GPU, không NVLink), 48 CPU; không internet |
| Điều kiện L4 | `competition_sources: ["gemma-4-developer-agent"]` + acc đã accept rules trên web (push lỗi "accept this competition's rules" nếu chưa) |
| Mô hình | Kaggle Model `danbth/qwen3-5-9b/transformers/default/1` (bf16, FP8 online `--quant fp8`), kiểm sha256 |
| Thư viện | `nhatlong1103/zc-assets` (wheels vLLM 0.32), `yeshenhi/zc-train-wheels` (huấn luyện: unsloth, trl, peft); code đẩy theo kernel (`<acc>/zc-code`) |
| Dữ liệu | `nhatlong1103/zc-2026`; riêng mỗi acc: `zc-para`, `zc-final-adapter`, `zc-wg-adapters` (`Process/kaggle/datasets_by_account.txt`) |
| Thời gian | đánh giá đủ tập (pub_full + pub_ho + tr_ho = 4,199 câu) ~60–75 phút + cài 7 phút + gộp 2–10 phút; tập con sàng lọc ~35–40 phút; luồng nộp 1,000 câu ~22 phút; huấn luyện 4 nhánh LoRA 14.7M token ~5 h (1 nhánh / GPU, unsloth) |
| Giới hạn | 2 phiên GPU + 5 CPU / acc; **30 h GPU / tuần / acc** (4×L4 ≈ 1.5–2.2 h quota / giờ tường); làm mới thứ Bảy 00:00 UTC; phiên commit ≤ 12 h |

### Tài khoản (11/11; quota: `python3 automation/quota.py`)

| Nhóm | Acc |
|---|---|
| L4, còn giờ đến 14/11 | bichthuy93, phuongnehe, caohuyen74, hieuho669, huuhien95, dinhthinh, loinguyentran, linhmai24 |
| L4, < 6 h / hết | anderson1202, duylnng, thnliu, yuan1103, synchan5704, lelynhan, huyenco, mucnokieu, nguyenhoa99, nguynduylm, phatle1578, quynhle07, quynhtran35 |
| không L4 | mxuanquy (chưa accept rules), yeshenhi (chỉ T4), nguyenmanh0404 |
| chỉ dataset | nhatlong1103 (dữ liệu, adapter nộp) |

Acc dùng chung với dự án khác: chỉ thao tác kernel `zc-*`. `automation/accounts.json`: danh sách acc không có L4.

## 3. Máy chấm (BTC) và Docker nộp

| Mục | Giá trị |
|---|---|
| Phần cứng | 2×RTX 5090 (32 GB), ≤ 6 giờ, `--network none`, `/data` chỉ đọc, kết quả `/result` |
| Image | `docker/Dockerfile`: `vllm/vllm-openai:v0.32.0`, pandas 2.3.3, numpy 2.1.3, openpyxl, python-calamine; trọng số gộp lúc build (`docker/merge_assets.py`, `soup.json`) |
| Chạy | `docker/predict.sh` → `src/predict_flow.py --config configs/flow_final.json` (ngân sách 20,400 s, gộp kết quả mỗi 20 s) |
| Hướng dẫn | `docker/SETUP_DOCKER.md` (`fetch_assets.sh` → `build.sh` → `run_public_test.sh` → `export_image.sh`), `docker/README_SOLUTION.md` |
| Đã kiểm | đường container trên Kaggle 4×L4 (0.986); build + chạy trên GPU thật: user làm 15/11 |
