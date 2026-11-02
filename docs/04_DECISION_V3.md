# 04 — Đánh giá phương án cũ và quyết định v3

*2026-11-02, khung chat 3. Bước 1 của `docs/ZAIC_REQUIREMENTS.md`: phân tích bản chất bài toán và hạn chế của phương án
hiện tại, rồi quyết định giữ hay bỏ.*

## 1. Bản chất bài toán (đọc trực tiếp dữ liệu, rulebook, câu hỏi)

- **Dữ liệu sinh tự động:** mỗi DN là một mô phỏng chuẩn (đơn, dòng hàng, khách, nhập, kho, nợ, chi phí…) được *xuất*
  ra nhiều kiểu file bẩn (phẳng / header + chi tiết / SQLite nhiều bảng / mỗi tháng một file / trạng thái nằm trong ghi chú…).
  Đáp án do bộ sinh tính trên dữ liệu chuẩn theo rulebook.
- **Câu hỏi theo khuôn mẫu** (~55 mẫu phủ ~90 % train: doanh thu theo kỳ, % thay đổi, số đơn, tỉ lệ huỷ, ngày cao nhất,
  top-k, giá bán, công nợ, kho, nhập, lãi…) cộng nhiễu văn chat (viết tắt, không dấu, gõ telex sai).
- **Rulebook đổi theo split:** public thêm ngành (mẹ & bé, sửa xe, điện thoại) và luật mới (khách trung thành, giờ cao
  điểm, nhà cung cấp chính…). **Private có ngành mới.** Luật chung (§1–§5, §7–§10) gần như giữ nguyên.
- **Hệ quả:** bài toán tách thành 3 việc độc lập, mỗi việc có lời giải đúng duy nhất:
  1. *ánh xạ* file bẩn của một DN → bảng chuẩn (mỗi DN một lần);
  2. *ngữ nghĩa* rulebook (kỳ, đơn hợp lệ, doanh thu, xếp hạng + hoà theo mã, nợ, kho…) — viết một lần, kiểm chứng được;
  3. *hiểu câu hỏi* → chọn mẫu + tham số (ngắn, lặp lại giữa các DN).

## 2. Phương án v1 (branch `claude/peaceful-carson-0dny90`): LLM tự viết chương trình pandas cho từng câu

| Chỉ số | Giá trị (nguồn: `docs/history/v1_findings.md`) |
|---|---|
| PA train (cấu hình tốt nhất, 3000 câu) | dev +0.47 / holdout +0.44; top-answer đúng 71 % |
| + thinking (354 câu holdout) | +0.51 |
| Tốc độ | 25–35 s/câu/L4; thinking 52–98 s/câu/L4 |
| Public test | **764/1000 câu sau 12 giờ** trên 4×L4 (chưa xong); compile hỏng 4/14 DN |

**Public LB (02/11, chỉ tính SLM ≤ 15B):** 1.0, 1.0, 1.0, 1.0, 0.94, 0.936, 0.925, 0.913, 0.912, 0.898… → v1 (≈ 0.45)
cách xa top và còn quá chậm (thời gian là tiêu chí phụ khi hoà điểm).

**Nguyên nhân mang tính bản chất (không sửa được bằng prompt / shots):**
1. Mô hình 14B phải **tự diễn giải lại toàn bộ rulebook trong mỗi chương trình dài** (kỳ, đơn hợp lệ, đơn vị cơ bản,
   hoà theo mã, khách lẻ…). Mỗi câu có hàng chục chỗ sai độc lập; lỗi mang tính **hệ thống** (cùng sai ở mọi mẫu) nên bỏ
   phiếu K = 4 không sửa được (đồng thuận 4/4 chỉ đúng 80 %).
2. Chương trình dài + K mẫu + vòng sửa lỗi + thinking → **chậm** (không kịp 1000 câu trong 12 giờ trên 4×L4).
3. Bước compile (LLM tự viết bảng chuẩn đầy đủ) **giòn**: hỏng ở 4/14 DN public, mất phần lớn lợi ích của DN đó.

→ **Quyết định: BỎ v1.**

## 3. Phương án v2 (branch `claude/funny-euler-ceut3o`): toolkit nghiệp vụ + chương trình ngắn

- `dm.py`: ngữ nghĩa rulebook thành hàm có kiểm chứng (kỳ, doanh thu, xếp hạng, nợ, kho, lô, lãi…).
- Mỗi DN: chương trình **ánh xạ ngắn** (đổi tên cột → `finalize`, nơi cài luật §8 "cách ghi trong file").
- Mỗi câu: chương trình 1–3 dòng gọi toolkit, từ bộ đọc mẫu câu `qparse.py` hoặc LLM (few-shot câu train → chương trình).

**Kiểm lại trong khung chat này (CPU, không LLM, ánh xạ tham chiếu của chính DN):** 3000 câu train → **2651 đúng /
51 sai / 298 chưa đọc** (khớp số đã báo). Nghĩa là: *khi ánh xạ đúng*, lớp ngữ nghĩa + mẫu câu sai ~2 % số câu đọc được.

**Hạn chế còn lại (đúng là phần cần LLM):**
1. **Ánh xạ DN chưa thấy:** chuyển giao tham chiếu chỉ được 4/14 DN public; 11/30 DN train (leave-one-out). Phần còn lại
   cần LLM viết ánh xạ — **chưa từng chạy với LLM**.
2. **Câu chưa đọc được** (~10 % train, nhiều hơn ở public): cần LLM sinh chương trình toolkit.
3. **Ngành / luật mới** (public, private): toolkit chưa có hàm → cần LLM viết code từ rulebook.

## 4. Kiến trúc v3: Map → Parse → Execute → Vote

```
data ─► [1] ingest tổng quát (CPU, cache) ─► bảng T + schema card
        [2] ánh xạ DN → bảng chuẩn C:  (a) bộ chuyển đổi định dạng tổng quát / tham chiếu khớp cột
                                       (b) LLM viết chương trình ánh xạ (few-shot 2 tham chiếu gần nhất), K ứng viên,
                                           vòng sửa lỗi theo dm.check; chọn ứng viên bằng kiểm tra + đồng thuận đáp án
        [3] câu hỏi → chương trình toolkit: (a) qparse tất định  (b) LLM K mẫu, few-shot chương trình đã kiểm chứng
                                           (khác DN), đầu ra ~100 token
        [4] câu ngoài toolkit / ngành mới: LLM viết pandas trên C + T, kèm mục rulebook liên quan
        [5] thực thi sandbox song song (CPU) ─► bỏ phiếu (gộp theo luật chấm) ─► answer / unanswerable / abstain
```

- **Mọi lời gọi LLM là việc hẹp, đầu ra ngắn** → nhanh, ít sai hệ thống; ngữ nghĩa nằm ở code đã kiểm chứng.
- **Ước lượng thời gian:** ánh xạ ~30 DN × K = 8 × ~1.5k token; câu hỏi ~2000 × K = 4 × ~120 token; với vLLM + prefix
  cache trên 2×5090: dưới 30 phút (v1: nhiều giờ).
- **Không vi phạm luật:** không logic theo id; tham chiếu là chương trình tổng quát theo tên cột, chạy train luôn loại
  tham chiếu của chính DN (leave-one-out).

## 5. Trục mô hình (user bổ sung 02/11: dùng SOTA ≤ 15B và / hoặc pipeline riêng, chạy song song)

| Ứng viên (public < 2026-10-01) | Tham số | Ghi chú |
|---|---|---|
| Qwen3-14B (AWQ / FP8) | 14.8B | đã có wheels + trọng số offline (`nhatlong1103/zc-assets`, `zc-m-qwen3-14b-fp8`) |
| Qwen3.5-9B | 9B | 02–03/2026, LiveCodeBench v6 65.6 (model card); trọng số `nhatlong1103/zc-m-qwen35-9b` |
| Qwen3.5-9B + Qwen3.5-4B | 13B | bỏ phiếu chéo hai mô hình (tổng ≤ 15B) |
| Gemma 4 | E2B/E4B nhỏ; 26B MoE, 31B vượt 15B | chỉ bản nhỏ hợp lệ |

Pipeline độc lập với mô hình → so cặp trên cùng câu (dev biz_101–120, holdout biz_121–130).

## 6. Kế hoạch thí nghiệm (theo thứ tự)

| # | Thí nghiệm | Đo |
|---|---|---|
| V0 | Port + kiểm lại toolkit, ingest (sửa tên file `#Uxxxx`), test CPU | 2651/51/298 tái lập |
| V1 | v3 trên train leave-one-out, Qwen3-14B-AWQ, 4×L4 | tỉ lệ DN ánh xạ được, PA dev / holdout, s/câu |
| V2 | Cùng V1 với Qwen3.5-9B (song song, acc khác) | PA so cặp, tốc độ |
| V3 | Public test với cấu hình tốt nhất → file nộp thử (user nộp) | PA public LB |
| V4+ | Sửa theo nhóm lỗi: ánh xạ (bộ chuyển đổi tổng quát), câu chưa đọc, ngành mới, unanswerable | PA |
| Vn | Docker 2×5090: FP8, tất định, thời gian thật | ΔPA ≤ 0.03, < 6 giờ |
