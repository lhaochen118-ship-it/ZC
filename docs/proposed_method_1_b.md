# Proposed method 1b — Tăng năng lực mô hình trong giới hạn 15B

*Zalo AI Challenge 2026, track DataMate — 15/11/2026. Tài liệu đề xuất: tổng hợp phân tích và phương án (chưa có thí nghiệm mới).*

## 0. Bối cảnh và câu hỏi

**Framework nộp hiện tại** (`configs/flow_final.json`, cấu hình O):

- Mô hình chính Qwen3.5-9B + LoRA **WG1 × 0.35** (LoRA trên D1 + toàn bộ D2 paraphrase).
- Các LoRA viết câu trả lời, viết luật, viết tool; toolkit Python (`dm`, `industry`).
- Luồng hybrid: mô hình viết mọi chương trình trả lời; luật chỉ gợi ý / kiểm chéo.

**Kết quả:**

| Thước đo | Giá trị |
|---|---|
| Public gốc (2 seed) | 0.986 / 0.985 (tb 0.9855); file nộp tốt nhất 0.991 |
| Held-out | `pub_ho` 0.644 / 0.636, `tr_ho` 0.882 / 0.881, gộp tb **0.778** |
| 3 câu sai ổn định | chọn sai hàm (bỉm size M theo bịch, bình quân phiếu thu, nhập về N viên); còn lại là phiếu tách thay đổi theo seed |

**Câu hỏi của user:** có cách nào thay mô hình 9B bằng mô hình lớn hơn để học được quy luật phức tạp hơn, rồi thu nhỏ để tổng tham số
≤ 15B như yêu cầu của BTC không? Chưng cất (distillation) đã thử, không hiệu quả lắm.

**Ràng buộc:**

- Tổng tham số ≤ 15B (BTC đếm số tham số, không đếm byte).
- Offline, ≤ 6 giờ trên 2×RTX 5090 (32 GB mỗi GPU).
- Huấn luyện trên Kaggle 4×L4 24 GB, không NVLink, phiên ≤ 12 giờ.
- Không train trên public / private test; không vá theo DN / câu.

### 0.1 Bằng chứng mô hình lớn hơn có ích cho bài này

Tỉ lệ chương trình chạy ra đúng khi các mô hình sinh dữ liệu D3 trên câu train, chưa fine-tune (WP5,
`Process/history/STATUS_full_until_2026-11-11.md`):

| Mô hình | Tỉ lệ giữ |
|---|---|
| Qwen3.6-27B (dense) | **73.6 %** |
| Gemma 4 31B | 69.5 % |
| Qwen3.5-35B-A3B (MoE, khoảng 3B active) | 64.4 % |
| self (bộ sinh của chính pipeline; đoán là Qwen3.5-9B, chưa đối chiếu) | 51.8 % |

Khi chưa fine-tune, mô hình lớn hơn mạnh hơn rõ rệt, kể cả MoE chỉ có 3B active (+12.6 điểm so với self). Câu hỏi còn mở: mức chênh
này còn lại bao nhiêu sau khi nén và fine-tune, so với 9B + WG1 (0.9855 / 0.778).

### 0.2 Chỗ yếu thực sự

Khoảng cách lớn nhất hiện nay là **held-out** (`pub_ho` 0.64), không phải public (0.986). Một phần chênh lệch đến từ paraphrase làm đổi
nghĩa câu hỏi (đổi kỳ / đại lượng). Mọi phương án dưới đây nên được đo **trên held-out trước**, theo quy tắc §4 của `docs/12_PLAN2.md`:
2 seed, public tăng ≥ 0.3, held-out gộp không giảm > 0.3.

## Phần A — Mô hình lớn rồi thu nhỏ (các hướng user gửi)

### A.1 So sánh các cách nén mô hình lớn xuống ≤ 15B

| Cách | Kích thước sau khi nén | Đánh giá |
|---|---|---|
| **A1. Tỉa expert của Qwen3.6-35B-A3B** (REAP / RAZOR / EAN) | còn 64 / 256 expert: 10.9B; khoảng 77: 12.1B; khoảng 96: khoảng 14.5B. Active vẫn khoảng 3B nên suy luận nhanh hơn 9B dense | **Nên chọn.** Tỉa 50 % expert giữ khoảng 93 % pass@1 và khoảng 100 % pass@k, không cần healing (arXiv 2611.01133). Ở cùng kích thước, tỉa expert giữ năng lực tốt hơn hẳn tỉa FFN (100 % so với 48 % coverage) và cắt lớp. |
| A2. Tỉa expert của Gemma-4-26B-A4B, khoảng 50 % | khoảng 13–14B (ước tính) | Phương án dự phòng. Gemma 4 31B (dense) làm thầy D3 tốt, nhưng chưa biết bản MoE 26B làm tốt đến đâu. |
| A3. Tỉa chiều rộng / cắt lớp của Qwen3.6-27B dense xuống 15B (kiểu Minitron, ShortGPT) | khoảng 15B | Không nên. Phải cắt khoảng 45 % nên mô hình hỏng nặng, cần healing hàng chục tỷ token; Kaggle 4×L4 không làm được. |
| A4. Lượng tử hoá 27B / 35B | vẫn 27B / 35B tham số | **Không hợp lệ:** BTC đếm số tham số, không đếm byte. |

### A.2 Dữ kiện về hai tài liệu user gửi

**Model card `IAmTheRealGumbo/Qwen3.6-35B-A3B-REAP-64-experts`:**

- Giữ 64 / 256 expert mỗi lớp (25 %), vẫn route 8 expert / token.
- Tổng tham số 35.95B → 10.93B, active vẫn khoảng 3B; 21.9 GB bf16.
- Chọn expert bằng 1,024 mẫu **code tiếng Anh** (evol-codealpaca), không fine-tune lại sau khi tỉa.
- HumanEval pass@1: 94.5 → 85.4 (chỉ đo HumanEval). Giấy phép Apache-2.0.

**arXiv 2611.01133** — Song & Zheng, "Rethinking LLM Pruning via Test-Time Scaling" (31/10/2026):

- Mô hình: Qwen3.6-35B-A3B (40 lớp, 256 expert / lớp, top-8), GLM-4.7-Flash, Gemma-4-26B-A4B-it; đối chứng dense Qwen3.5-9B / 4B / 2B.
- Ba kiểu tỉa: expert (RAZOR; EAN và tần suất route làm baseline), FFN (Taylor), cắt lớp (kiểu SLEB). **Không healing.**
- Kích thước Qwen3.6 (tổng / active): tỉa expert 50 % 18.54B / 2.94B; 70 % 12.12B / 2.93B; 80 % 8.85B / 2.93B.
- Tỉa 50 % expert: giữ khoảng 92.8 % accuracy và khoảng 100.6 % coverage (pass@k).
- Tỉa nặng: chuỗi suy luận đa dạng nhưng lặp và bị cắt nhiều hơn; số token cho mỗi bài giải được có thể tăng mạnh.
- Trộn nhiều kiểu tỉa (12.45B) kém hơn tỉa expert 70 % (12.12B).

**Điểm có lợi cho mình:** mô hình đã tỉa giữ pass@k tốt hơn pass@1. Pipeline của mình vốn sinh nhiều mẫu (k 2 → 8, review, bộ chọn M6,
kiểm tra bằng thực thi), nên khai thác đúng điểm này. Rủi ro đi kèm: phiếu bầu đa số yếu đi khi pass@1 giảm; tỉa nặng làm đầu ra lặp
hoặc bị cắt nhiều hơn.

Tỉa nhẹ hơn bản REAP-64 (khoảng 63 % thay vì 75 %) và chọn expert bằng dữ liệu tiếng Việt + pandas + toolkit `dm` của mình sẽ giữ được
nhiều hơn hẳn: chỉ cần giữ các expert mà bài này thực sự dùng.

### A.3 Quy trình đề xuất cho A1

1. **Thí nghiệm dò năng lực dư** (khoảng 1 phiên GPU, không train): chạy luồng O, không LoRA, trên `pub_ho400` và `tr_ho400` với bốn mô
   hình: Qwen3.5-9B gốc, Qwen3.6-35B-A3B gốc, bản REAP-64 có sẵn, bản tự tỉa còn khoảng 96 expert. Nếu bản tỉa khoảng 14B **không**
   hơn 9B gốc ít nhất 5 điểm thì dừng hướng này.
2. **Tỉa có định hướng theo bài toán:** tính độ quan trọng của từng expert theo REAP (trọng số cổng × độ lớn đầu ra expert) trên khoảng
   2–3k prompt đúng định dạng suy luận (D1, D2, D3, câu train). Không dùng public hay private test.
3. **"Học trên mô hình to rồi mới thu nhỏ":**
   - Fine-tune LoRA trên mô hình 35B đầy đủ, **chỉ ở attention + shared expert, router đóng băng** → phần học được nằm ở chỗ không bị tỉa.
   - Gộp LoRA, đo độ quan trọng trên mô hình đã fine-tune, tỉa xuống khoảng 14B.
   - Fine-tune lại ngắn bằng công thức WG1 (D1 + D2).
   - Phương án rẻ hơn để so sánh: tỉa trước rồi mới fine-tune LoRA.
4. **Quyết định** theo §4 `docs/12_PLAN2.md` (2 seed).

**Rủi ro kỹ thuật:**

- Huấn luyện MoE 35B trên L4 24 GB phải dùng QLoRA hoặc FSDP, chậm; Unsloth hỗ trợ MoE chưa ổn định.
- vLLM nạp được checkpoint đã tỉa (chỉ đổi `num_experts`) nhưng phải chạy thử trước theo cổng preflight.
- Bản cuối khoảng 14B bf16 = 28 GB: vừa một RTX 5090 32 GB nhưng còn ít chỗ cho KV cache → lượng tử FP8 lúc suy luận (vẫn hợp lệ vì
  số tham số không đổi).

## Phần B — Nhận xét ba ví dụ khác user đưa

| Ví dụ | Đánh giá |
|---|---|
| **Recursive Transformer** (Bae et al., ICLR 2025) | Hướng này **nén mô hình**, không tăng năng lực: dùng lại một khối lớp theo vòng lặp, thêm LoRA riêng cho từng lớp. Trong bài, Gemma 2B nén xuống khoảng 1B cần huấn luyện tiếp 15–60 tỷ token mới gần bằng bản gốc. Áp lên Qwen3.6-27B để xuống khoảng 14B cần lượng tính toán tương tự → **không khả thi** với Kaggle 4×L4. |
| **Qwen 9B + Qwen 4B làm verifier** | Hợp lệ (tổng khoảng 13B). Nhưng một LoRA verifier trên chính backbone 9B cho tín hiệu tương tự mà không tốn thêm VRAM; hướng C3 dưới đây còn rẻ hơn nữa. |
| **Một backbone, nhiều LoRA chuyên biệt** (A: sinh chương trình, B: kiểm tra, C: giải thích business rules / câu khó) | Rẻ nhất; vLLM phục vụ được nhiều LoRA cùng lúc. Lưu ý từ dữ liệu của mình: súp không hơn thành viên tốt nhất, nên dùng các LoRA tách vai trò thay vì trộn lại. |

## Phần C — Năm hướng mới

Đã đối chiếu với những gì repo đã thử (M0–M9, bộ chọn M6, truy hồi embedding S1, DPO M4, RFT M2, súp) để không đề xuất lại. Năm hướng
này khác tỉa MoE, Recursive Transformer, verifier 4B, multi-LoRA và chưng cất.

### C1. Mở rộng chiều sâu Qwen3.5-9B lên khoảng 11–13.5B (Depth Up-Scaling / Block Expansion)

Đi thẳng tới mục tiêu "mô hình to hơn để học quy luật phức tạp hơn" mà **không cần bước thu nhỏ**: dùng hết trần 15B ngay từ đầu.

- **Cách làm:**
  - Qwen3.5-9B có 32 lớp = 8 nhóm × (3 lớp Gated DeltaNet + 1 lớp attention), hidden 4096.
  - Chèn thêm 2–4 nhóm, chép trọng số từ nhóm bên cạnh, đặt bằng 0 các phép chiếu đầu ra (`out_proj`, `down_proj`) → lớp mới là
    ánh xạ đồng nhất, mô hình ban đầu cho đầu ra y hệt 9B.
  - **Chỉ huấn luyện các lớp mới**, đóng băng lớp cũ → không quên kiến thức cũ.
- **Kích thước:** mỗi lớp khoảng 0.22–0.25B (ước tính, cần đếm chính xác trên checkpoint). +8 lớp ≈ 11B; +16 lớp ≈ 13B.
- **Bằng chứng:**
  - LLaMA Pro (Wu et al., ACL 2024): LLaMA2-7B + 8 khối = 8.3B, chỉ huấn luyện khối mới trên code / toán → năng lực code / toán tăng,
    kiến thức chung không giảm.
  - SOLAR 10.7B (Kim et al., NAACL 2024): ghép lớp từ 7B lên 10.7B rồi huấn luyện tiếp (Depth Up-Scaling).
- **Rủi ro:** dữ liệu của mình (D1 + D2 + D3, khoảng vài triệu token) ít so với 2–4B tham số mới → bắt đầu với 8 lớp, trộn thêm
  dataset ngoài M8 (SynSQL-2.5M, BIRD, Spider). Suy luận chậm hơn khoảng 1.25–1.5 lần.
- **Thử đầu tiên:** 9B + 8 lớp, huấn luyện lớp mới bằng dữ liệu kiểu WG1, so với 9B + WG1 trên `pub_ho400` và `tr_ho400`.

### C2. RL với phần thưởng từ thực thi (GRPO / DAPO) — M5, có trong kế hoạch nhưng chưa từng chạy

- **Khác DPO (M4, đã loại):** DPO học offline từ cặp (đúng, sai) cố định nên bị khớp cặp (margin 9.2, không tổng quát). GRPO là
  on-policy: mô hình tự sinh nhóm chương trình, chạy toolkit để chấm, học từ chính phân phối của nó.
- **Phần thưởng:** 1 nếu đáp án đúng, 0.1 nếu chương trình chạy được (cách đơn giản của Arctic-Text2SQL-R1); có thể thêm điểm từng phần
  cho kỳ, thực thể, định dạng như Reasoning-SQL / SQL-R1.
- **Dữ liệu:** câu train + paraphrase train có đáp án, bỏ `tr_ho`.
- **Vì sao hợp lỗi hiện tại:** câu sai còn lại là **chọn sai hàm** và **phiếu bị tách theo seed**. RL thường làm tăng pass@1 nhưng ít
  mở rộng pass@k; với mình đó lại đúng thứ cần: đáp án đúng đã nằm trong k mẫu, chỉ cần dồn phiếu về nó.
- **Chi phí:** 8–20 GPU-giờ (TRL GRPO + vLLM, trên LoRA); runner đã hỗ trợ checkpoint qua giới hạn 12 giờ.

### C3. Bộ kiểm "tự soi" từ trạng thái ẩn của chính mô hình 9B (gần như 0 tham số thêm)

- **Cách làm:**
  - Với mỗi chương trình ứng viên (+ tóm tắt kết quả thực thi), chạy thêm một lượt prefill qua 9B (trùng prefix cache nên rẻ).
  - Lấy hidden state ở các token cuối của chương trình / đáp án; huấn luyện probe tuyến tính hoặc MLP nhỏ dự đoán đúng / sai.
  - Đưa điểm này vào bộ chọn M6 (hiện chỉ có 13 đặc trưng bề mặt: tỉ lệ phiếu, `lp_mean`, …).
- **Bằng chứng:** "LLMs Know More Than They Show" (Orgad et al., ICLR 2025): tín hiệu đúng / sai tập trung ở token đáp án; mô hình có
  thể "biết" đáp án đúng mà vẫn sinh đáp án sai. Probe không chuyển được giữa các dataset → huấn luyện trên log của chính bài toán.
- **Dữ liệu:** log có sẵn (m0dev, m0para, m0d3) chỉ cần chạy lại prefill để lấy hidden state, không sinh mới → **rẻ nhất trong 5 hướng**.
  Cần log của cấu hình nộp (WG1) để probe khớp phân phối.
- **So với verifier 4B:** không tốn VRAM, không cần huấn luyện mô hình riêng, đọc được "suy nghĩ" của chính bộ sinh.

### C4. Thích nghi theo từng doanh nghiệp lúc suy luận (Test-Time Adaptation)

- **Cách làm:**
  - Trước khi trả lời một DN mới, mô hình dùng các khung câu hỏi train (`retrieval.skeleton` đã có) tự sinh cặp câu hỏi → chương trình
    **trên chính bảng của DN đó**, rồi chạy thử.
  - Chỉ giữ cặp qua kiểm tra vòng: diễn đạt lại câu hỏi → sinh lại chương trình → kết quả phải giống.
  - Các cặp này làm ví dụ few-shot đúng schema (hiện ví dụ lấy từ DN khác). Biến thể mạnh hơn: huấn luyện nhanh một LoRA riêng cho DN.
- **Bằng chứng:** Akyürek et al. (ICML 2025): LoRA riêng cho từng bài lúc test tăng tới 6 lần trên ARC; mô hình 8B + ensemble với sinh
  chương trình đạt 61.9 %, ngang người; BBH +7.3 điểm so với few-shot thường.
- **Vì sao hợp:** held-out thấp (0.64) chủ yếu ở DN và cách hỏi chưa gặp — đúng chỗ thiếu ví dụ đúng schema.
- **Cần user quyết:** biến thể chỉ dùng ví dụ trong ngữ cảnh không cập nhật trọng số (chỉ là tính toán lúc suy luận). Biến thể LoRA theo
  DN cập nhật trọng số từ dữ liệu test (không nhãn) → có thể trái nguyên tắc "không train trên private test" nếu hiểu theo nghĩa đen.
  Ngân sách 6 giờ trên 2×5090 cũng phải đủ.

### C5. Đầu ra là kế hoạch truy vấn có kiểu (IR / DSL) + sinh có ràng buộc

- **Cách làm:** mô hình sinh kế hoạch JSON ngắn (đại lượng, bộ lọc, kỳ, nhóm, hàm gộp, top-k, so sánh); trình biên dịch tất định dịch
  sang lệnh toolkit `dm`; IR không diễn đạt được thì quay về Python tự do.
- **Lợi ích:**
  - Đầu ra khoảng 30–60 token.
  - Ngữ pháp nhỏ → sinh có ràng buộc nhanh (ràng buộc trên cả Python từng chậm 32 lần, QĐ12).
  - Không còn lỗi cấu trúc; bỏ phiếu so sánh IR trực tiếp, sạch hơn so sánh đáp án số.
- **Hợp luật hybrid:** mô hình vẫn viết chương trình (bằng DSL); trình biên dịch là luật logic, không phải luật trả lời.
- **Bằng chứng:** biểu diễn trung gian là hướng kinh điển trong text-to-SQL: IRNet (ACL 2019), NatSQL (Findings of EMNLP 2021).
- **Rủi ro:** thay đổi kỹ thuật lớn nhất; phải dựng lại dữ liệu SFT sang dạng IR.

## Phần D — Xếp hạng và lộ trình

| Ưu tiên | Hướng | Tăng năng lực mô hình? | Chi phí | Lợi kỳ vọng |
|---|---|---|---|---|
| 1 | **C3. Bộ kiểm tự soi** | không, chọn tốt hơn | thấp (vài GPU-giờ, log sẵn) | câu phiếu tách, bỏ trả lời đúng lúc |
| 2 | **C2. GRPO thưởng thực thi** | có, sửa phân phối | trung bình (8–20 GPU-giờ) | chọn sai hàm, phiếu tách |
| 3 | **C1. Mở rộng chiều sâu 11–13B** | **có, thêm dung lượng thật** | trung bình–cao | held-out, câu nghiệp vụ phức tạp |
| 4 | **A1. Tỉa expert 35B-A3B → khoảng 14B** | có (mô hình nền mạnh hơn) | cao (MoE 35B trên L4) | held-out; suy luận nhanh (3B active) |
| 5 | **C4. Thích nghi theo DN** | không / có (biến thể LoRA) | trung bình, tốn giờ suy luận | held-out (DN mới) |
| 6 | **C5. IR / DSL** | không, thu hẹp bài toán | cao (kỹ thuật) | lỗi cấu trúc, tốc độ |

**Kết hợp:** C1 (hoặc A1) → SFT kiểu WG1 → C2 (GRPO); C3 đặt lên trên bất kỳ mô hình nào; C4 / C5 độc lập với lựa chọn mô hình.

**Thực tế về thời gian:** hạn nộp **16/11 23:59 (GMT+7)**, còn khoảng 1.5 ngày. Không hướng nào kịp qua quy tắc 2 seed trước hạn →
**giữ nguyên bản nộp hiện tại**. Nếu muốn thử ngay, C3 là ứng viên duy nhất có thể có số trong một ngày (cần token Kaggle). Các hướng còn
lại dành cho vòng sau.

## Tài liệu tham khảo

- IAmTheRealGumbo, Qwen3.6-35B-A3B-REAP-64-experts (model card) — huggingface.co/IAmTheRealGumbo/Qwen3.6-35B-A3B-REAP-64-experts
- Song & Zheng, Rethinking LLM Pruning via Test-Time Scaling, arXiv 2611.01133 (2026)
- Bae et al., Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA, ICLR 2025 — arXiv 2410.20672
- Wu et al., LLaMA Pro: Progressive LLaMA with Block Expansion, ACL 2024 — aclanthology.org/2024.acl-long.352
- Kim et al., SOLAR 10.7B: Scaling LLMs with Simple yet Effective Depth Up-Scaling, NAACL 2024 (Industry)
- Orgad et al., LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations, ICLR 2025 — arXiv 2410.02707
- Akyürek et al., The Surprising Effectiveness of Test-Time Training for Few-Shot Learning, ICML 2025 — arXiv 2411.07279
- Pourreza et al., Reasoning-SQL: RL with SQL Tailored Partial Rewards, arXiv 2503.23157
- Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL, arXiv 2505.20315
- SQL-R1: Training Natural Language to SQL Reasoning Model by Reinforcement Learning, arXiv 2504.08600
- Guo et al., IRNet: Towards Complex Text-to-SQL with Intermediate Representation, ACL 2019
- Gan et al., NatSQL: Making SQL Easier to Infer from Natural Language Specifications, Findings of EMNLP 2021
- Qwen3.5-9B model card — huggingface.co/Qwen/Qwen3.5-9B
