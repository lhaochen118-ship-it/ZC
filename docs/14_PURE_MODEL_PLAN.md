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

## Kết quả (12/11)

| Mô hình (cấu hình Q, thuần) | Public | pub_ho400 / tr_ho400 (tb) |
|---|---|---|
| Q0: WG1 × 0.35 (huấn luyện có gợi ý), đủ tập | 0.841 | gộp đủ tập 0.719 (pub_ho 0.5375, tr_ho 0.855) |
| QS1 (D1p + D2p) × 0.35 / × 0.70 | 0.859 / 0.841 | 0.668 / 0.676 |
| QS2 (D1p + D2mix p) × 0.35 / × 0.70 | 0.871 / 0.848 | 0.668 / 0.681 |
| QS4 (= QS1 seed khác) × 0.35 / × 0.70 | 0.867 / 0.835 | 0.670 / **0.714** |

Học sinh chỉ học lại dữ liệu cũ không gợi ý: +2–3 public so với Q0, held-out ≈; α lớn hơn: public ↓, held-out ↑. Lỗi còn lại là đúng phần
module gợi ý làm (kỳ viết tắt 'Th 12/25', 'T4/2025' → quý; 'o đờ' = đơn; tên giá trị; hàm tự bịa). Hướng tiếp: **chưng cất module gợi ý
vào đầu ra** (`pure_records.py --self-hints`: nhãn bắt đầu bằng `<hints>` mà luật đọc được từ chính câu đó; lúc suy luận mô hình tự viết,
không luật nào chạy) — QS8 = QS1 + tự gợi ý, QS7 = D1 + D2 / D2N + tự gợi ý (12/11 15:35). Đang chạy: QS3 (sàng lọc), QS5 / QS6 (huấn luyện).

**Cập nhật 12/11 20:10** (sàng lọc, cấu hình Q):

| Mô hình | Dữ liệu | × 0.35: public / held-out tập con | × 0.70 |
|---|---|---|---|
| QS3 | D1p + D2p + D2bp | 0.868 / 0.691 | 0.838 / 0.694 |
| QS5 | D1p + D2p + D2Np (câu nhiễu) | 0.874 / 0.678 | 0.874 / 0.703 |
| QS6 | D1p + D2p + D2bp + D2Np, 26M | 0.885 / 0.700 | 0.875 / **0.7125** |
| QS8 | tự gợi ý, D1 + D2 (= QS1 + tự gợi ý) | 0.877 / 0.658 | 0.863 / 0.691 |
| **QS7** | **tự gợi ý, D1 + ½ D2 + ½ D2N** | **0.905** / 0.656 | 0.893 / 0.689 |

Tự gợi ý + câu nhiễu = public thuần cao nhất (0.905, +4.6 so với QS1; P có luật 0.993). Câu nhiễu (D2N) giúp cả public lẫn held-out;
tự gợi ý giúp public (+1.8–2.2 so với cùng dữ liệu). Huấn luyện vòng 2: QS9 (D1p + D2p + D2Np + RFT của QS5, không tự gợi ý) và
**QS10** (tự gợi ý + D1 + RFT + ⅓ D2 / ⅓ D2b / ⅓ D2N, 26M token; chuỗi checkpoint qua acc, phần 1 nguyenmanh0404).

**Cập nhật 13/11 02:00** (vòng 2, cấu hình Q):

| Mô hình | Dữ liệu | × 0.35: public / held-out tập con | × 0.70 |
|---|---|---|---|
| QS9 | D1p + D2p + D2Np + RFT của QS5 (không tự gợi ý) | 0.880 / 0.6475 | 0.875 / 0.695 |
| QS10 | tự gợi ý + D1 + RFT + ⅓ D2 / ⅓ D2b / ⅓ D2N, 26M (chuỗi checkpoint 2 acc) | 0.889 / 0.6725 | 0.896 / 0.694 |

Mốc có luật P + WG1 (tập con): public 0.993–0.994, pub_ho400 0.6125–0.6225, tr_ho400 0.8575–0.8775 (tb ≈ 0.74–0.75).

**Kết luận hướng thuần (vòng 1–2):** public thuần cao nhất vẫn là **QS7 × 0.35 = 0.905**; held-out tập con cao nhất **QS6 × 0.70 = 0.7125**.
Vòng 2 (RFT, thêm dữ liệu đến 26M token) không vượt vòng 1: chưng cất thêm cùng loại dữ liệu đã bão hoà ở public ≈ 0.89–0.905.
Khoảng cách tới P (≈ 9 điểm public) gần như toàn bộ nằm ở câu kiểu public (pub_ho: thuần 0.50–0.55 so với P 0.61–0.62; tr_ho: thuần
0.82–0.88 ≈ P 0.86–0.88): kỳ viết tắt, tiếng lóng / viết tắt, tên giá trị — đúng phần module gợi ý + kiểm tra luật đang làm. Mô hình thuần
**không thắng P** theo §4 của `docs/12_PLAN2.md` → bản nộp giữ P + WG1. Hướng tiếp (đưa vào thảo luận kế hoạch mới với user): dữ liệu huấn
luyện mang đúng phân bố câu public (paraphrase kiểu lóng / viết tắt / kỳ viết tắt sinh có kiểm), không chỉ thêm dữ liệu cùng loại.
