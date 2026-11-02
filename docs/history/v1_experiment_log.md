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

## Entry 3 — 2026-10-30 (tối): khởi động lại, helper ngữ nghĩa, PA đầu tiên

**Hạ tầng.** Token mới của user (13 acc + `nhatlong1103` chỉ để đăng dataset) lưu ở `~/.kaggle/tok_<acc>`. Không có token
`nguyenhoa99` → không kéo được e2b/e2c. Quota GPU tuần còn ở: `lelynhan`, `phuongnehe`, `linhmai24`, `thnliu` (cả 4 đều
4×L4 thật, đã kiểm bằng kernel probe), `mxuanquy`, `anderson1202` (chỉ T4). Hết quota: yuan1103, synchan5704, mucnokieu,
nguynduylm, yeshenhi, huyenco, loinguyentran.

**Sửa lỗi (bước 1 của handoff).** Helper ngày an toàn (`add_months`, `week_range`, `same_days_prev_month`); sandbox chặn
`open` / `pd.read_*`; gợi ý sửa theo loại lỗi (`.str`, day out of range, KeyError kèm danh sách cột, thiếu ANSWER);
`--biz` (chia run theo DN), `--shots` (few-shot truy hồi, bỏ ví dụ cùng DN), `--compile-scope sales`, test cục bộ sửa lỗi symlink.

**Smoke L4 (zc-smk2-2, path B ×4, 30 câu):** chạy đúng môi trường; **PA = −0.40**; câu đồng thuận 4/4 chỉ đúng 40 % → lỗi
hệ thống, không phải nhiễu: đếm dòng thay vì đơn, lấy tổng sau giảm giá đơn làm doanh thu, `<= ngày cuối` trên cột có giờ.

**`src/semantic.py: build_sales`** (mới): cài đúng luật ghi chép của rulebook (ô trống kế thừa dòng đầu đơn, dòng trùng
trong cùng đơn tính một lần, trạng thái đơn từ các dòng, valid, doanh thu = tổng thành tiền dòng). LLM chỉ cần ánh xạ cột.
Kiểm trên biz_101/106/113/124 (phẳng có/không mã đơn, header+chi tiết, có giờ): khớp đáp án doanh thu tháng của BTC.

**E3h kernel 2** (`--helpers --compile --compile-scope sales`, AABB K=4, 752 câu của 9 DN): **PA +0.12 khi chạy (τ=0.5);
τ=0.75 chọn trên dev → dev +0.19, holdout +0.25.** Đồng thuận 4/4: 80 % đúng (smoke: 40 %). Path A (bảng chuẩn) ≫ path B.
Lỗi tự tin còn lại: viết tắt chat không hiểu (`hqua`, `hnay`, `bữa qua` → trả doanh thu toàn thời gian), khách lẻ bị xếp
hạng như khách, quên lọc `valid`, đếm đợt giao theo dòng, quy đổi đơn vị. Sửa: `src/vnnorm.py` (`--norm-q`), khách lẻ →
None trong build_sales, tham số bảng quy đổi `conv=`, `C['valid_orders']`/`C['valid_lines']`. E4 đang đo các sửa này.

## Entry 4 — 2026-10-30 (21:00 UTC): sửa compile + few-shot truy hồi → PA +0.6

Cùng giao thức (τ chọn trên dev biz_101–120, báo cáo holdout biz_121–130). Mọi so sánh trên **cùng tập DN**.

| Run | Cấu hình | DN | top-answer đúng | PA dev | PA holdout |
|---|---|---|---|---|---|
| e3h (k1–3) | `--helpers --compile --compile-scope sales` AABB K=4 | 22 DN, 2251 câu | 1004/2251 | +0.187 (τ 0.75) | +0.101 |
| e3b-1 | `--helpers` BBBB K=4 (không compile) | 16 DN, 1502 câu | 663/1502 (e3h cùng DN: 729) | +0.166 | +0.005 |
| e3h-2 (cắt) | như e3h | 114,116,117,125 (316 câu) | 172/316 | +0.268 | +0.250 |
| e4-2 | + `--norm-q`, khách lẻ, `conv=` | cùng 4 DN | 172/316 (biz_116 compile hỏng) | +0.185 | +0.325 |
| **e7** | + compile 4 vòng sửa, build_sales giữ cột gốc / tự tìm header, `valid_*`, làm tròn tiền, list/table, **`--shots` (2 ví dụ, khác DN)** | cùng 4 DN | **253/316** | **+0.591 (τ 0.5)** | **+0.700** |

Kết quả âm / trung tính (đã loại): ngưỡng τ theo type (holdout 0.096 vs 0.101 của τ chung); trọng số phiếu A > B (dev
không đổi). B-only < AABB: giữ compile. Lỗi compile còn lại trước e7: quên `header=`, truyền nhầm bảng khách làm header,
dùng tên cột gốc sau build_sales, merge theo tên mã đơn gốc → đã xử lý trong `semantic.build_sales`.

Quota: `thnliu`, `phuongnehe` hết quota tuần lúc ~20:45 UTC. Public test chạy với cấu hình e7 (run `p1`, `lelynhan`).

**Ablation shots (cùng code, cùng kết quả compile: biz_105/115 có C, biz_108/118 không), biz_105,108,115,118 (474 câu dev):**

| Run | Thay đổi | top-answer đúng | PA dev (τ chọn trên dev) |
|---|---|---|---|
| e3h-1 (cắt) | baseline helpers | 272/474 | +0.279 (τ 0.75) |
| e5-1 | + norm-q, khách lẻ, conv, auto-header | 293/474 | +0.357 (τ 0.75) |
| e6a | e5 + `--shots` 2 ví dụ | **370/474** | **+0.584 (τ 0.5)** |

→ few-shot truy hồi (chương trình đúng của DN khác, train) là đòn bẩy lớn nhất (+0.23 PA); xác nhận độc lập trên e7.

**E3t (thinking, T4 ×2, `--helpers --think` BB K=2, không shots, 100 câu dev):** trên 65 câu trùng với e3b (helpers,
không think): độ đúng từng mẫu **77/130 (think) vs 52/130 (không think)**; lỗi chạy 11 vs 18. Rất chậm trên T4 (19,401 s cho
100 câu × 2 mẫu). → thử `--think` trên L4 với cấu hình e7 (E10) để đo lợi ích / thời gian.

**E8 (holdout, cấu hình e7 + `--link`, shots v2): biz_121,122,124,127,129,130 — 708 câu, 6/6 DN compile được:
PA holdout = +0.394 khi chạy (τ = 0.5)** (e3h trên 236 câu trùng: +0.114). Đây là ước lượng trung thực nhất hiện có.
Thời gian: ~25 s/câu mỗi L4 (5,941 s cho 236 câu trên 1 GPU).
**E3t-1 (thinking, holdout 100 câu):** từng mẫu think không shots 26/60 vs không think không shots 20/60 (30 câu trùng e3b);
so với cấu hình e8 (không think, có shots) trên 60 câu trùng: 54/120 (45 %) vs 123/240 (51 %). → shots > thinking; cộng dồn
think + shots đang đo (E10, T4, 40 câu).

## Entry 5 — 2026-10-31 00:30 UTC: toàn bộ train với cấu hình tốt nhất

Hợp các run e6a, e7, e7b, e8, e9 (cùng cấu hình `--helpers --norm-q [--link] --compile --compile-scope sales --k 4
--kc 4 --shots --n-shots 2`, shots luôn từ DN khác) phủ **đủ 3000 câu train**:

| | PA dev (1976 câu) | PA holdout (1024 câu) | top-answer đúng |
|---|---|---|---|
| τ = 0.5 (khi chạy) | **+0.471** | **+0.437** | 2130/3000 (71 %) |
| τ = 0.75 | +0.479 | +0.426 | |

→ giữ τ = 0.5. Theo type: number +0.48, percent +0.50, date +0.65, string +0.53, list +0.22, **unanswerable −0.29**,
table −0.16. e9: compile được 10/11 DN. Thư viện shots v4: 4228 chương trình đúng, phủ 2446/3000 câu (`src/shots/`).

**E10 (T4, cấu hình tốt nhất + `--think --max-tokens 8192`, 40 câu của biz_114/116/117/125) vs e7 trên cùng 40 câu:**
từng mẫu đúng **134/160 (84 %) vs 111/160 (69 %)**; PA cuối **+0.75 vs +0.60**. Think + shots cộng dồn được. Mẫu nhỏ (40 câu)
→ cần e10b (holdout) / e10l (L4, 316 câu) để xác nhận; đo thời gian trên L4 (T4: 10,144 s cho 40 câu, gồm compile).

**E10l (L4, cấu hình tốt nhất + `--think --max-tokens 8192`, shots v3, code mới nhất) vs e7 trên cùng 316 câu
(biz_114/116/117/125):** PA dev **+0.692 vs +0.591**, holdout **+0.900 vs +0.700** (40 câu), top-answer đúng **272 vs 253**;
4/4 DN compile được. Thời gian 6,043 s vs ~2,400 s (≈ 2.5×; ~47 s/câu/L4). Lưu ý: e10l còn có code mới hơn (khoảng ngày dữ
liệu, SafePool) và shots v3 (e7: v1), nên chênh lệch không hoàn toàn do thinking; e10 (cùng câu hỏi, T4) cho kết luận
cùng chiều. Ước lượng 2×5090 (≈3–4× L4): 1000 câu ≈ 2 giờ, 2000 câu ≈ 4–4.5 giờ → vừa 6 giờ nhưng sát với 2000 câu (cần
`--budget`, chạy thử thật).

**E10b (T4, như e10, 40 câu holdout biz_121/124/129/130, 4/4 compile được) vs e8 trên cùng câu:** từng mẫu 84/160 vs
79/160 (lỗi chạy 1 vs 14); PA cuối **+0.00 vs +0.30** (thinking kém hơn). → bằng chứng lẫn lộn: dev 316 câu (e10l) +0.10,
dev 40 câu (e10) +0.15, holdout 40 câu (e10b) −0.30. Với 40 câu, sai số của hiệu PA ≈ ±0.15. **Chưa đổi mặc định**; phép thử
quyết định: e11 (thinking, 354 câu holdout biz_121/124/129, L4, đang xếp hàng) so với e8.

## Entry 6 — 2026-10-31 07:45 UTC: thinking trên holdout (phép thử quyết định)

**e11** (L4, code mới nhất: khoảng ngày dữ liệu, SafePool, gợi ý `time=`; shots v4; `--think --max-tokens 8192`) vs **e8**
(không think, shots v2) trên **cùng 354 câu holdout** (biz_121/124/129; e11 compile được 3/3 DN):

| | PA holdout | đúng / sai / abstain | number | percent | list | unanswerable |
|---|---|---|---|---|---|---|
| e8 | +0.254 | 208 / 118 / 28 | +0.25 | +0.24 | +0.20 | +0.00 (9 câu) |
| **e11** | **+0.514** | 256 / 74 / 24 | +0.57 | +0.50 | −0.07 | **+1.00** (9 câu) |

→ gói "code mới nhất + shots v4 + thinking" hơn +0.26 PA trên holdout. Chưa tách riêng phần của thinking (e11 khác e8 ở cả code
và shots). Thời gian: 52–98 s / câu / L4 (biz_124 chậm nhất: 11,602 s cho 118 câu). Ước lượng 2×5090: ~2000 câu ≈ 4–4.5 giờ.
**Cấu hình khuyến nghị mới = cấu hình tốt nhất + `--think --max-tokens 8192`**, kèm `--budget` để an toàn thời gian.

**p1 (public test, cấu hình tốt nhất KHÔNG think, shots v2, code trước SafePool):** Kaggle dừng ở giới hạn 12 giờ; 3/4 shard
xong → **764/1000 câu có đáp án** (number 371, percent 112, string 48, list 37, date 15, unanswerable 11, boolean 7, table 5;
abstain 158). Shard 2 (dev_005, dev_008) kẹt khi thực thi chương trình (đúng lỗi SafePool sửa); dev_012 chưa tới lượt.
Compile hỏng 4 DN public (dev_007, 011, 013, 014): rulebook / cấu trúc public khác train. File nộp đủ 1000 dòng (236 câu
thiếu ghi `abstain`): `results/p1_submission.jsonl`. Chưa có điểm LB (user quyết định nộp).
