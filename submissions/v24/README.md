# File nộp public test — hệ v24

Code chạy trên Kaggle: commit `f3d21c9` (v23 + các sửa ở EXPERIMENT_LOG entry 15 / 16); kiểm tra ánh xạ tính lại bằng `19fed21`, `finalize` mới cho dev_013 ở commit đi kèm file này. Ba lần chạy public (seed
0 / 1 / 2) trên `thnliu`. Lần chạy train v24 không chạy được vì gần như mọi acc đã hết quota GPU tuần; mọi thay đổi đã được
kiểm bằng oracle train (2967 / 1 / 32) và test CPU.

Thay đổi chính so với v23:
- **Câu hỏi về phẩm chất chủ quan** mà rulebook không định nghĩa (ngon nhất, thân thiện, khó tính, nhiệt tình, uy tín, đáng
  tin cậy, trung thực, chất lượng tốt) → `unanswerable` (rulebook §7).
- **Ngành đã có toolkit:** xử lý như DN thường (luật chuẩn trọng số 4, luật ngành 2, bỏ mẫu bảng thô). Thêm luật đọc câu
  cho các hàm ngành, cho giao hàng theo đơn đặt và đơn bán chịu theo ngày phải trả.
- **Nợ chỉ ghi trong ghi chú:** `debt_from_note`, cột chuẩn `orders.debt_end`.
- **Bẫy API:** lọc trạng thái huỷ / trả xét mọi trạng thái; "tháng này so với tháng trước" so cùng số ngày.
- **Làm sạch câu và từ địa phương:** "o đờ" = đơn, "xi ô đi" = COD, "Tại tiệm" = tại quầy, "pho mát" = "phô mai",
  "xây trát" = "xây tô".
- **Kiểm tra ánh xạ mới:** tên thợ sai so với sổ, sổ góp / bảo hành / thu máy cũ bị bỏ sót, cột tiền của đơn mang một giá
  trị cho mọi đơn, nhóm khách bị bỏ sót, ghi chú nợ mâu thuẫn với C. Lấy thêm một đợt mẫu ánh xạ khi mọi ứng viên có cảnh
  báo nặng.
- **dev_013, chạy lại offline** bằng `automation/jobs/replay_biz.py` (cùng chương trình LLM đã sinh, ánh xạ dựng lại bằng
  `finalize` mới). Các dòng "Voucher", "Dùng N điểm", "Phí ship" là điều chỉnh của hoá đơn, không phải dòng hàng. Doanh thu
  (rulebook §1) không trừ voucher, không gồm phí ship, nhưng mọi bản từ v19 đến v24 đều tính các dòng này vào doanh thu.
  Ví dụ tháng 6/2026: 429.012.000 → 432.382.000. Tự kiểm: các câu không bị ảnh hưởng giữ nguyên đáp án đã ghi.

`submission.jsonl` = bỏ phiếu 3 lần chạy có tie-break theo ánh xạ (điểm phạt tính bằng các kiểm tra mới nhất). Đã kiểm định
dạng: 1000 id, 0 lỗi, **14 câu abstain** (v22: 76, v23: 49), 33 câu unanswerable, 0 đáp án bằng 0.

- Ba lần chạy public: **903 nhất trí / 71 đa số / 26 tách** (v22: 856 / 68 / 76; v23: 834 / 101 / 65). Mỗi lần chạy mất
  65–70 phút trên 4×L4 (v23: khoảng 2 giờ).
- So với v22 (portal 0.71): 855 câu giống hệt; 66 câu thêm trả lời (dev_014: 35); 4 câu thêm abstain; 75 câu đổi đáp án
  (dev_013: 32, do doanh thu sửa ở trên; dev_014: 9; dev_011: 8; dev_008: 8; dev_004: 8).
- So với v23: 879 câu giống hệt; 44 câu thêm trả lời; 9 câu thêm abstain; 68 câu đổi đáp án.
