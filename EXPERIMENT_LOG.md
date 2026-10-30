# Experiment log — DataMate

Mỗi entry: ngày, thí nghiệm (mã E# trong `docs/01_PLAN.md`), giả thuyết, cấu hình (mô hình, prompt, K, ngưỡng), dữ
liệu (split, số câu), kết quả (PA tổng và theo type / loại câu, % abstain, thời gian/câu), phân tích lỗi, quyết định.

| Entry | Ngày | Thí nghiệm | PA (dev) | PA (holdout) | PA public LB | s/câu | Quyết định |
|---|---|---|---|---|---|---|---|

## Entry 0 — 2026-10-30: khởi tạo

- Đọc đề trên portal (Description, Evaluation, Rule, Submission Guidance, Timeline) → `docs/00_TASK.md`.
- Viết `src/score.py` theo luật khớp của BTC; self-test các ví dụ trong bảng Evaluation đều qua.
- Chưa có dữ liệu (`data/` trống) và chưa có token Kaggle trong container → E0/E1 chờ.

## Entry 1 — 2026-10-30: dữ liệu, bộ làm sạch tổng quát, agent v0

- Dữ liệu tải từ `nhatlong1103/zc-2026`; tóm tắt ở `docs/00_TASK.md` §3.
- `src/ingest.py`: làm sạch tự động, chỉ dựa trên quy ước chung (không theo tên DN/cột cụ thể). Audit "cột chỉ gõ kiểu
  được một phần" giảm từ 78 → 0 trên train; public còn 2 cột IMEI (đúng là chữ). Làm sạch toàn bộ: train 57 s, public
  40 s (4 nhân).
- Kiểm chứng: 3 chương trình viết tay trên bảng đã làm sạch (`biz_101`: số đơn hôm qua, doanh thu hôm nay, ngày doanh
  thu cao nhất tháng) ra **đúng tuyệt đối** đáp án BTC.
- Kaggle: kernel T4 có internet; `pip install vllm` (0.32, torch 2.13 + CUDA 13) mất ~6 phút và xung đột torchaudio →
  gỡ torchaudio; L4 grant không cho internet; `yuan1103` hết quota GPU tuần.
- E1 đang chạy: Qwen3-14B-AWQ, không thinking, K = 4, 2 vòng sửa lỗi, DP trên 2 T4, 300 câu train (10/DN).

## Entry 2 — 2026-10-30 (chiều): hạ tầng L4, E2 lần 1 hỏng, sửa bộ làm sạch, chuẩn bị E7

**Giao thức kiểm định (cố định từ đây):** chọn cấu hình / ngưỡng trên `biz_101`–`biz_120` (dev); báo cáo riêng trên
`biz_121`–`biz_130` (holdout, không dùng để chọn gì); public test chỉ để đối chiếu qua portal.

- **E1b / E1c** (T4, Qwen3-14B-AWQ, TP=2, 300 câu): đang chạy (code trước các bản sửa ingest dưới đây).
- **E2 lần 1** (2 phiên × 4 L4, DP=4, compile + 2 nhánh; 300 câu train + 1000 câu public): **hỏng**, mọi câu
  abstain. FlashInfer sampler JIT-compile 20 phút rồi lỗi `ld` trên L4. Sửa: `VLLM_USE_FLASHINFER_SAMPLER=0` trong
  `agent.py`. **E2b** chạy lại cùng cấu hình.
- **Bộ làm sạch:**
  - bộ test tổng hợp `tests/test_ingest_robust.py` (12 biến thể: `;`, UTF-16, header 2 tầng, dòng tổng/ghi chú, header
    lặp, ngày `Ngày 1 tháng 6 năm 2025`, `VND 1,234,567`, `12,5%`, `(12.50)`, mã có số 0 đầu, thập phân `.`/`,`):
    12/12;
  - so từng giá trị cache cũ/mới trên dữ liệu thật → phát hiện **`1,134tr` bị đọc thành 1,134 tỷ** (13 DN:
    dev_003/004/006/007/008/010/013/014, biz_117/124/129) → sửa (dấu trước hậu tố nhân là dấu thập phân). E1b/E1c và
    E2b vẫn dùng code cũ có lỗi này.
- **Độ dài prompt** (tokenizer Qwen3): prompt biên dịch dài nhất 17,974 token (dev_006); thêm bảo vệ trong
  `Engine.fit` (bỏ vòng sửa cũ → giảm max_tokens → bỏ request, không làm hỏng lô).
- **E7 chuẩn bị:** `harvest.py` (STaR), `train_sft.py` (QLoRA DDP), `agent --lora`; Kaggle Models
  `qwen-lm/qwen-3/transformers/14b/1` gắn được bf16 offline (28 GB).
- **Assets public (`nhatlong1103`):** `zc-assets` (wheels vLLM + Qwen3-14B-AWQ), `zc-m-qwen3-14b-fp8`,
  `zc-m-qwen35-9b`, `zc-m-gemma4-12b-qat`.
- **Docker:** `docker/` (build / run như BTC / export + sha256), hướng dẫn `docker/SETUP_DOCKER.md`.

## Entry 3 — 2026-10-30 (tối): E2c public test

Code mới nhất, 4×L4, `--compile --k 4 --kc 4 --dp 4`: 1000/1000 câu; nạp model 230 s (trước 1320 s); thông lượng
vòng trả lời đầu 176–234 tok/s mỗi L4 (+30 %), compile 21–57 tok/s; 4/4 phiếu 26 %, 2/4 45 %; compile 3/14 DN; lỗi
hàng đầu không đổi (ngày ngoài phạm vi, tên cột, `.str`, thiếu ANSWER, đọc file). Không tạo file nộp (user: chỉ khi
kết quả tốt). Chi tiết: `handoff/01_SCIENTIFIC_FINDINGS.md` §3.
