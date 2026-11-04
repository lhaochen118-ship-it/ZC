# File nộp public test — hệ v25

Code chạy trên Kaggle: commit `98d601f` (v24 sửa lỗi + điều chỉnh hoá đơn giữ trong doanh thu như v23) + `f4be0c0`.
6 kernel 4×L4 song song trên 3 acc mới (loinguyentran, anderson1202, phatle1578): train + **5 lần chạy public** (seed 0–4),
mỗi lần chạy public 55–75 phút. Điểm phạt ánh xạ để phá hoà tính lại bằng code mới nhất (`31e67fc` … `2cbfb07`: kiểm tra nợ,
trạng thái trả / huỷ, biển số).

**Train (seed 0): PA 0.9893** (2973 đúng / 5 sai / 22 abstain trên 3000) — cao nhất từ trước tới nay (v15-1: 0.982).

`submission.jsonl` = bỏ phiếu 5 lần chạy (`src/vote.py`, ngưỡng 0.4, tie-break theo điểm phạt ánh xạ). Định dạng: 1000 id,
0 lỗi, **5 abstain**, 31 unanswerable.

Thay đổi so với v24 (portal 0.747) — giữ cái tốt của v23 (0.87), bỏ cái sai của v24:
- dev_013: các dòng "Voucher" / "Dùng N điểm" / "Phí ship" **giữ trong doanh thu** như v23 (v24 bỏ ra, nghi là nguyên nhân
  tụt điểm), nhưng đánh dấu `is_adjustment` và chép sang cột đơn → câu voucher / phí ship / điểm trả lời được.
- dev_014: alias tiếng Việt cho sổ góp / sổ nợ; ánh xạ thiếu sổ góp bị phạt nặng (v24 seed 1 chọn ánh xạ thiếu sổ góp).
- Giữ của v24: câu phẩm chất chủ quan → unanswerable; toolkit ngành (điện thoại, sửa xe, mẹ và bé); nợ ghi trong ghi chú.

So với v23: 923 giống hệt, 46 câu thêm trả lời (dev_014: 17, dev_004: 8, dev_012: 7), 29 câu đổi, 2 câu thêm abstain.
Kiểm tay trên dữ liệu thô 8 câu v23 và v25 khác nhau: **v23 đúng 0/8, v25 đúng 6/8** (nợ dev_008 = 4.549.169 đ / 15 khách
theo ghi chú; doanh thu từ đầu năm dev_014; xe khách quen dev_012 33 / 200; Wave Alpha 50). Hai câu v25 còn sai (dev_014:
ốp silicon tháng 1 = 86, iPhone 13 tháng 9 = 9: tên viết tắt ở sheet "Ghi tắt") được bản v26 sửa.
