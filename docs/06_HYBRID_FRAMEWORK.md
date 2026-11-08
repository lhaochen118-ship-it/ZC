# 06 — Hybrid framework: AI Agent sinh code + luật tổng quát hỗ trợ (thiết kế cho khung chat kế tiếp)

*Viết 08/11/2026 theo yêu cầu user. Đọc sau `handoff/03_HANDOFF.pdf`. Đây là bản thiết kế + kế hoạch thí nghiệm; mọi
lựa chọn phải được kiểm bằng held-out trước khi giữ (§7). Ranh giới được / cấm: §1.*

## 1. Ranh giới (user chốt 08/11 — không hỏi lại)

| Được phép: hard code LOGIC (chuẩn bị ngữ cảnh cho Agent) | Bị cấm: hard code ANSWER |
|---|---|
| Quy tắc tổng quát nhận diện ý định câu hỏi (loại đại lượng, kỳ thời gian, thực thể, bộ lọc, kiểu đáp án) | Luật sinh thẳng code Python / đáp án mà không qua mô hình |
| Đọc file data, dựng schema chuẩn bằng code (`src/ingest.py`, `src/automap.py`); Agent viết thêm mô tả schema khi cần | Đáp án của luật là đáp án cuối (như stage 1 của `src/predict_flow.py` hiện nay) |
| Toolkit nghiệp vụ đã kiểm (`src/dm.py`, `src/industry.py`) = rule engine Agent gọi | Mẫu / nhánh theo id DN, id câu (luật BTC §4) |
| Làm sạch câu (`src/qnorm.py`), truy hồi ví dụ đã kiểm (few-shot), kiểm tra code / kết quả bằng code | Dùng public / private test để train (luật BTC §6) |

**Nguyên tắc vận hành:** với **mọi** câu, chương trình trả lời do mô hình sinh. Luật chỉ cung cấp ngữ cảnh, gợi ý, kiểm
tra. Chương trình do `qparse` dựng (nếu có) được đưa vào prompt dưới dạng gợi ý hoặc dùng để kiểm chéo — không tự
thành đáp án, không là phiếu tự quyết.

## 2. Luồng vận hành

Luồng cơ bản user gợi ý: **câu hỏi tiếng Việt → mô hình tiếp nhận → chuyển thành code truy vấn → mô hình nhận kết quả
truy vấn → mô hình trả lời.** Bản chi tiết đề xuất (mỗi bước ghi rõ ai làm: M = mô hình, C = code / luật):

```
[Offline, mỗi DN một lần — trước khi nhận câu hỏi]
 S1 (C)  ingest: đọc CSV / Excel / SQLite -> bảng thô + card (cột, kiểu, ví dụ giá trị, README)
 S2 (C)  automap: bảng thô -> schema chuẩn C (orders, lines, items, payments, purchases, lots, ...) + cảnh báo
 S3 (M)  nếu automap cảnh báo nặng / thiếu bảng: Agent viết chương trình ánh xạ (đường v28), code kiểm
         (canon.secondary_checks), đồng thuận ánh xạ (consensus.py)
 S4 (M)  Agent viết "data card" ngắn: mô tả schema chuẩn của DN này bằng lời (cột có / không có, quy ước lạ)
 S5 (C)  rulebook của split -> danh sách thuật ngữ + định nghĩa; thuật ngữ mới so với train -> đánh dấu cho prompt

[Mỗi câu hỏi]
 Q1 (C)  làm sạch câu (qnorm) + trích gợi ý: kỳ thời gian (đã quy ra ngày), thực thể khớp dữ liệu (tên hàng / khách /
         NCC, kèm tên đúng trong data), đại lượng ứng viên, kiểu đáp án dự kiến; câu gốc luôn đi kèm
 Q2 (C)  truy hồi k ví dụ đã kiểm gần nhất (câu train + chương trình đúng) theo ý định + từ vựng (fewshot.py)
 Q3 (M)  Agent sinh K chương trình trên toolkit (+ nhận định ngắn: ý định, giả định); có thể sinh pandas / SQL trực
         tiếp khi toolkit không có hàm (đường thoát, chạy sandbox)
 Q4 (C)  kiểm tra TRƯỚC khi chạy (module tổng quát, §4.4): cú pháp, tên hàm / cột có thật, thực thể trong code có trong
         câu (grounding), kỳ thời gian trong code khớp gợi ý Q1, kiểu đáp án khớp câu hỏi
 Q5 (C)  chạy trong sandbox (timeout, read-only) -> kết quả + bảng trung gian nhỏ
 Q6 (M)  Agent đọc kết quả (bước "mô hình nhận kết quả"): kết quả có hợp lý không (rỗng, 0, âm, quá lớn, sai đơn vị);
         nếu không -> sửa code (vòng sửa, tối đa R lần) hoặc đề xuất abstain / unanswerable
 Q7 (C+M) bỏ phiếu các chương trình theo kết quả; điểm tin cậy = tỉ lệ phiếu + tín hiệu kiểm tra + tự đánh giá của
         Agent; dưới ngưỡng -> sinh lại (thêm mẫu, prompt khác) -> vẫn thấp -> abstain
 Q8 (C)  định dạng đáp án theo đề (fmt_answer), ghi ngay vào submission (append + flush)
```

Đã có trong code (tái sử dụng): S1 `ingest.py`; S2 `automap.py`; S3 `agent.py --think-map`, `canon.py`, `consensus.py`;
Q1 `qnorm.py` + phần đọc kỳ / thực thể của `qparse.py` (`parse_period`, `named_*`); Q2 `fewshot.py`; Q3 `agent.py`
(`--k`, `--kc`, `--shots`); Q5 sandbox trong `pipeline.py`; Q6 một phần (`--repairs`, grounding); Q7 `--escalate`,
`vote.py --min-conf`; Q8 `merge_sub.py`. **Còn thiếu:** S4 data card, Q1 gói gợi ý cho prompt (thay vì đáp án), Q4
module kiểm tra trước khi chạy thống nhất, Q6 bước Agent đọc kết quả, Q7 điểm tin cậy kết hợp + sinh lại có điều kiện.

## 3. Tham khảo: các hệ đang hiệu quả nhất (đọc paper, không đọc code repo người khác — nguyên tắc user)

| Hệ / paper | Ý chính | Áp dụng cho DataMate |
|---|---|---|
| **DS-STAR** (Google, 2025; arXiv 2509.21825) — agent khoa học dữ liệu trên nhiều file khác định dạng, đứng đầu DABStep | (1) phân tích mọi file → mô tả văn bản; (2) Planner → Coder → **Verifier** (LLM chấm kế hoạch đủ chưa) → lặp; bỏ mô tả dữ liệu thì độ chính xác bài khó tụt mạnh (26.98 %) | Gần bài toán nhất (dữ liệu bẩn nhiều file). Lấy: S4 data card, Q6 bước kiểm kết quả. **DABStep** là benchmark ngoài gần nhất để thử |
| **ReFoRCE** (arXiv 2502.00675) — đứng đầu Spider 2.0 | nén schema, khám phá cột / giá trị bằng truy vấn thăm dò, tự sửa theo lỗi thực thi, **bỏ phiếu đồng thuận** | Thăm dò giá trị (tên hàng thật trong data) trước khi viết code; tự sửa theo lỗi |
| **CHASE-SQL** (arXiv 2410.01943), **XiYan-SQL** (arXiv 2411.08599, 2507.04701) | sinh **nhiều ứng viên đa dạng** (nhiều prompt / nhiều bộ sinh), sửa, rồi **mô hình chọn** ứng viên; XiYan: lọc schema → nhiều bộ sinh → mô hình chọn | Q3 đa dạng hoá prompt (không chỉ đổi seed); Q7 thay bỏ phiếu thuần bằng chọn có lý do khi bất đồng |
| **CHESS** (arXiv 2405.16755) | truy hồi thực thể / giá trị từ catalog, lọc schema, sinh + kiểm | Q1 khớp thực thể với giá trị trong data (đã có một phần) |
| **OmniSQL** (arXiv 2503.02240) — SynSQL-2.5M | tổng hợp dữ liệu text-to-SQL quy mô lớn (CSDL + câu + SQL + CoT), fine-tune 7B / 14B; bỏ phiếu đa số tăng thêm | Mô hình tham khảo cho **dữ liệu tổng hợp** (§5.1). Giấy phép Apache-2.0 nhưng chỉ SQLite / tiếng Anh |
| **Arctic-Text2SQL-R1** (arXiv 2505.20315, Findings ACL 2026) | RL (GRPO) với **thưởng chỉ dựa trên đúng / sai khi thực thi**; 7B đạt 68.5 BIRD test, vượt hệ 70B cũ | Nếu fine-tune: thưởng = đáp án đúng trên train + câu tổng hợp (§5.1) |
| Chưng cất CoT có kiểm thực thi (ví dụ Qwen3.5-2B distill từ 27B, SQuaD-SQL arXiv 2607.08161) | thầy lớn sinh lời giải, **chỉ giữ lời giải chạy ra đúng** (rejection sampling), LoRA cho học trò | §5.1: thầy lớn trên Kaggle sinh chương trình toolkit cho câu train + câu tổng hợp |
| **TrustSQL** (arXiv 2403.15879) — chấm có phạt, như PA | mô hình phải trả lời hoặc **từ chối**; nhiều phương pháp không vượt baseline "luôn từ chối" khi phạt nặng | Abstain là một năng lực riêng, phải đo bằng PA, không chỉ độ chính xác |
| Selective prediction cho text-to-SQL (arXiv 2607.06799) | tín hiệu dự báo đúng / sai: **self-consistency một mình không cho tập rủi ro thấp đáng tin**; verifier / kết hợp tín hiệu tốt hơn (AUROC 0.82, ECE 0.03) | Q7: kết hợp nhiều tín hiệu (§4.3), hiệu chỉnh trên dev |
| **ViText2SQL** (VinAI, EMNLP Findings 2020) | Spider dịch sang tiếng Việt (~10k cặp); PhoBERT > XLM-R; tách từ giúp ích | Nguồn câu tiếng Việt cho tổng hợp / huấn luyện mô hình phụ (giấy phép nghiên cứu / giáo dục — **dùng được**, quy tắc dataset: `docs/07_DATASETS.md` §1) |

Bài học chung từ các hệ trên: (1) **ngữ cảnh dữ liệu giàu** (mô tả file, giá trị thật) là đòn bẩy lớn nhất; (2) **nhiều
ứng viên + chọn / bỏ phiếu**; (3) **tự sửa theo phản hồi thực thi**; (4) mô hình nhỏ đạt mức mô hình lớn nhờ **dữ liệu
tổng hợp có kiểm thực thi + fine-tune / RL thưởng thực thi**; (5) abstain cần tín hiệu tin cậy kết hợp, không chỉ phiếu.
Lưu ý: benchmark công khai có tỉ lệ lỗi nhãn cao (CIDR 2026: BIRD Mini-Dev ~53 %), nên số leaderboard chỉ để tham khảo.

## 4. Các hướng cải tiến (theo thứ tự ưu tiên đề xuất)

### 4.1 Gói gợi ý từ luật thay cho đáp án luật (bắt buộc, làm đầu tiên)
Tách `qparse` thành hai phần: (a) **trích gợi ý** (kỳ thời gian quy ra ngày, thực thể khớp data, đại lượng, kiểu đáp
án, bộ lọc) → khối `# HINTS` trong prompt; (b) chương trình mẫu → chỉ là "gợi ý chương trình" (có cờ để bật / tắt và đo).
Đo ba biến thể trên held-out: không gợi ý / gợi ý / gợi ý + chương trình mẫu.

### 4.2 Data card do Agent viết (S4)
Mỗi DN một lần: Agent đọc schema chuẩn + vài dòng + README → mô tả ngắn (≤ 300 token) đặt trong phần prompt chung
(prefix cache). Theo DS-STAR, mô tả dữ liệu là thành phần quan trọng nhất ở bài khó.

### 4.3 Điểm tin cậy + sinh lại có điều kiện (Q7)
Agent xuất kèm `confidence` (0–1) và giả định chính. Điểm tin cậy cuối = hàm hiệu chỉnh (logistic, học trên **train +
paraphrase k=0 public**) của: tỉ lệ phiếu, số chương trình lỗi, kết quả kiểm tra Q4 / Q6, tự đánh giá của Agent, độ dài
suy luận. Dưới ngưỡng θ₁ → sinh lại (prompt khác, thêm ví dụ, nhiệt độ khác, tối đa N lần); dưới θ₂ sau khi sinh lại →
abstain. Chọn θ tối đa PA trên dev, kiểm trên held-out. Ghi nhớ: tự tin của LLM thường lệch, self-consistency một mình
không đủ (§3) → phải hiệu chỉnh.

### 4.4 Module kiểm tra tổng quát trước khi chạy (Q4)
Một hàm `precheck(code, question, hints, schema)` trả danh sách vi phạm: hàm / cột không tồn tại; thực thể trong code
không có trong câu (grounding) hoặc thực thể trong câu không có trong code; kỳ thời gian khác gợi ý; kiểu trả về khác kiểu
dự kiến; phép chia không bảo vệ. Vi phạm → đưa lại cho Agent sửa (không âm thầm sửa thay). Đây là "luật logic" được phép.

### 4.5 Bước Agent đọc kết quả (Q6)
Sau khi chạy: Agent nhận câu + code + kết quả rút gọn → trả "OK / SỬA (lý do) / KHÔNG TRẢ LỜI ĐƯỢC". Chỉ gọi khi có tín
hiệu nghi ngờ (bất đồng, kết quả 0 / rỗng / bất thường) để giữ thời gian.

### 4.6 Thêm tool cho Agent (chỉ khi thực sự cần)
Ứng viên: `lookup_values(table, column, pattern)` (thăm dò giá trị như ReFoRCE — tên hàng / khách thật), `describe(table)`,
`sample_rows(table, n)`. Chỉ thêm nếu phân tích lỗi cho thấy lỗi do Agent không biết giá trị / cột; mỗi tool là một vòng
gọi thêm → đo thời gian.

### 4.7 Mô hình phụ (tuỳ chọn, tính vào trần 15B)
Qwen3.5-9B = 9.65B → còn ~5B. Ứng viên: **PhoBERT-base (~135M)** / PhoBERT-large (~370M) fine-tune cho (a) phân loại ý
định / đại lượng, (b) dự báo đúng / sai từ (câu, code, kết quả) làm một tín hiệu tin cậy, (c) embedding truy hồi ví dụ.
Chỉ đáng làm khi đo được lợi trên held-out so với dùng chính Qwen hoặc luật gợi ý; thêm mô hình = thêm rủi ro Docker /
thời gian. Trọng số phải public trước 01/10/2026 và giấy phép cho phép (luật BTC §2); khai báo trong README nộp.

### 4.8 Chưng cất / fine-tune (tốn nhất, cân nhắc sau 4.1–4.5)
- **Dữ liệu ngoài:** chọn theo quy tắc + bảng thống kê ở `docs/07_DATASETS.md` (dataset public là dùng được; chỉ dataset
  uy tín: hội nghị / tạp chí có hạng hoặc nhiều citation).
- **Dữ liệu:** câu train (3000, có đáp án) + **câu tổng hợp** trên dữ liệu train (sinh câu theo mẫu ý định × thực thể ×
  kỳ, đáp án tính bằng toolkit — luật được dùng để *tạo dữ liệu huấn luyện*, không để trả lời lúc chạy) + paraphrase của
  câu train (`src/paraphrase_llm.py`). **Không** dùng câu / đáp án / paraphrase của public test để train (luật BTC §6).
- **Thầy:** mô hình lớn có trọng số public trước 01/10/2026, chạy trên Kaggle 4×L4 (TP 4, FP8) — ví dụ Qwen dòng lớn hơn
  hỗ trợ tiếng Việt; sinh K chương trình toolkit + lời giải ngắn, **chỉ giữ chương trình chạy ra đúng đáp án** (rejection
  sampling, như OmniSQL / chưng cất CoT).
- **Trò:** Qwen3.5-9B, LoRA / QLoRA (bf16 trên L4, DDP 4 GPU, gradient checkpointing); hoặc RL thưởng thực thi kiểu
  Arctic-Text2SQL-R1 nếu còn thời gian. Đánh giá: held-out paraphrase train phải tách theo **DN** (giữ một số DN train
  ngoài tập huấn luyện) để đo tổng quát hoá thật.
- **Rủi ro thời gian:** hạn nộp 16/11; quota 30 h GPU / tuần / acc. Chỉ làm khi 4.1–4.5 đã chạy ổn.

## 5. Giao thức khi gặp câu trả lời sai (bắt buộc)

1. **Không vá câu.** Không thêm luật / regex / prompt chỉ để câu đó đúng.
2. **Phân loại gốc lỗi** (ghi vào `EXPERIMENT_LOG.md`): (a) schema / ánh xạ sai; (b) hiểu sai ý định / đại lượng; (c) kỳ
   thời gian sai; (d) thực thể khớp sai; (e) toolkit thiếu / sai định nghĩa rulebook; (f) code đúng ý nhưng lỗi thực thi;
   (g) đáp án đúng nhưng định dạng sai; (h) đáng ra phải abstain / unanswerable; (i) đáp án tham chiếu / paraphrase đổi
   nghĩa (không sửa).
3. **Tìm cả nhóm:** đếm bao nhiêu câu dev cùng gốc lỗi; chỉ sửa khi là nhóm (≥ vài câu, nhiều DN).
4. **Chọn cách sửa tổng quát nhất, theo thứ tự ưu tiên:** sửa toolkit / định nghĩa đã kiểm → thêm kiểm tra tổng quát
   (precheck / kiểm kết quả) → thêm ví dụ truy hồi đã kiểm → cải thiện gợi ý / data card → câu hướng dẫn trong prompt
   (ít, chỉ khi cần) → tool mới → mô hình phụ / fine-tune.
5. **Kiểm:** test CPU, train + public gốc không tụt, **held-out cải thiện** (paraphrase k=1,2 public, paraphrase train);
   ghi cả kết quả âm.

## 6. Ràng buộc nộp bài cần giữ trong thiết kế

- ≤ 6 giờ trên 2×RTX 5090 kể cả nạp mô hình; ghi đáp án dần (append + flush); |ΔPA| giữa hai lần chạy ≤ 0.03 →
  seed cố định, nhiệt độ thấp cho bước quyết định, đồng thuận ánh xạ.
- Tổng tham số mọi mô hình lúc inference ≤ 15B; offline; không dùng test để train.
- Ước lượng thời gian: đường LLM 4×L4 ~55–75 phút / lần chạy cho 1000 câu public (≈ 3.3–4.5 s / câu / lần chạy với 4 GPU).
  Mỗi bước thêm (Q6, sinh lại, tool) phải đo lại thời gian.

## 7. Kế hoạch thí nghiệm (khung chat kế tiếp)

| # | Thí nghiệm | Thước đo | Giữ khi |
|---|---|---|---|
| E0 | Nền: `agent.py --prefer-map fmt:automap --qw 0`, không `--confirm` | PA public gốc, train, paraphrase public k=0 / k=1,2, paraphrase train; s / câu | (mốc) |
| E1 | + gói gợi ý `# HINTS` (§4.1) | như trên | held-out ≥ E0 |
| E2 | + data card (§4.2) | như trên | held-out ≥ E1 |
| E3 | + precheck (§4.4) + Agent đọc kết quả (§4.5) | như trên + thời gian | held-out tăng, thời gian ≤ ngân sách |
| E4 | + điểm tin cậy hiệu chỉnh + sinh lại (§4.3) | PA, tỉ lệ abstain, đường cong rủi ro | PA held-out tăng |
| E5 | (tuỳ) mô hình phụ / tool / chưng cất (§4.6–4.8) | như trên | held-out tăng rõ, đủ thời gian |

Mốc so sánh: v34 (rule-first) held-out public 0.602; v28 (LLM có luật bỏ phiếu) public gốc 0.921.

## Nguồn

- DS-STAR: https://arxiv.org/abs/2509.21825 ; https://research.google/pubs/ds-star-data-science-agent-via-iterative-planning-and-verification/
- ReFoRCE: https://arxiv.org/abs/2502.00675 ; Spider 2.0: https://arxiv.org/abs/2411.07763
- CHASE-SQL: https://arxiv.org/abs/2410.01943 ; XiYan-SQL: https://arxiv.org/abs/2411.08599 , https://arxiv.org/abs/2507.04701 ; CHESS: https://arxiv.org/abs/2405.16755
- OmniSQL: https://arxiv.org/abs/2503.02240 ; Arctic-Text2SQL-R1: https://arxiv.org/abs/2505.20315
- Chưng cất nhỏ: https://huggingface.co/AlioLeuchtmann/qwen-3.5-sql-27B-Distill-2B ; SQuaD-SQL: https://www.alphaxiv.org/abs/2607.08161
- TrustSQL: https://arxiv.org/abs/2403.15879 ; Selective prediction: https://arxiv.org/abs/2607.06799
- ViText2SQL: https://huggingface.co/datasets/SEACrowd/vitext2sql (VinAI, EMNLP Findings 2020)
- Lỗi nhãn benchmark: https://www.vldb.org/cidrdb/papers/2026/p5-jin.pdf
