# 14 — Mô hình thuần sinh code (user 12/11)

**Mục tiêu (user 12/11):** nhanh chóng có mô hình **không cần luật hard code**, thuần túy mô hình sinh code truy vấn, mà vẫn đạt
score cao. **Ranh giới (user chốt):** lúc suy luận chỉ giữ **A** đọc dữ liệu / schema (`ingest`, `automap`) và **B** toolkit nghiệp vụ
(`dm`, `industry`); bỏ **C** gợi ý đọc từ câu hỏi bằng luật (`hints`: kỳ, thực thể, kiểu đáp án, từ rulebook, câu đã làm sạch, top-k) và
**D** kiểm tra / cờ theo luật (`precheck` size / unit / len_mask / count / size_use / ticket_avg, cờ kỳ / ngưỡng, review, fmt_fix).
Phản hồi của chính toolkit (cú pháp, tên / tham số không có trong API, giá trị không có trong dữ liệu) và lỗi khi chạy vẫn đưa lại cho
mô hình (như lỗi của bất kỳ thư viện nào). **Vai trò:** nhánh song song; bản nộp vẫn là P cho đến khi mô hình thuần thắng theo §4 của
`docs/12_PLAN2.md`.

## Cấu hình Q (`configs/flow_pure.json`, `--pure 1`)

`src/hybrid.py --pure 1`: không tính gợi ý (`hints = {}`) → prompt không có khối `# HINTS`, ví dụ truy hồi chỉ theo chữ câu hỏi
(`retrieval.skeleton(question, {})`), precheck chỉ còn phần API / cú pháp, không cờ theo luật, không top-k đọc từ câu. Lấy mẫu như O:
prompt full, 2 → 8 mẫu, vòng 2 think, trần 1536 token, bỏ phiếu, bỏ trả lời < 0.5.

## Cách làm: chưng cất từ hệ có luật

Hệ có luật (P: 0.993) chỉ dùng **lúc tạo dữ liệu**: các chương trình đã kiểm của câu train (D1), paraphrase train gắn chương trình của
câu gốc (D2, D2b). `training/data/pure_records.py` dựng lại chính các bản ghi đó trong prompt của cấu hình Q (không gợi ý, ví dụ truy hồi
theo chữ câu) — mô hình phải tự học kỳ thời gian, thực thể, kiểu đáp án từ câu hỏi + dữ liệu + ví dụ.

| Bước | Việc | Khi |
|---|---|---|
| Q0 | Mốc: WG1 × 0.35 (huấn luyện **có** gợi ý) chạy cấu hình Q, đủ tập (`namhunhh/zc-q0wg1-1`) | 12/11 chiều |
| Q1 | 4 học sinh thuần, mỗi nhánh một acc, FSDP 4 GPU: QS1 D1p + D2p (công thức WG1), QS2 D1p + D2mix p, QS3 D1p + D2p + D2bp 21M token, QS4 = QS1 seed khác | 12/11 chiều → tối |
| Q2 | Đánh giá mỗi học sinh cấu hình Q đủ tập (× 0.35 và α lớn hơn) | 12/11 tối |
| Q3 | Expert iteration: học sinh tốt nhất trả lời câu train dev + D3 + paraphrase train (cấu hình Q), giữ chương trình chạy ra đúng đáp án train → vòng 2 | đêm 12/11 → 13/11 |
| Q4 | So với P (public / held-out, 2 seed); thắng → đường container → đổi bản nộp; báo user, lập kế hoạch tiếp | 13/11 |

Mốc cần vượt: hướng M (không luật kế hoạch 2) WF3 + bộ chọn 0.954 / 0.775; mục tiêu tiến sát P (0.993 / 0.775).
