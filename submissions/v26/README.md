# File nộp public test — hệ v26 (đồng thuận ánh xạ trên 5 lần chạy v25)

Cùng 5 lần chạy public của v25 (không chạy GPU thêm). `src/consensus.py` (CPU, 33 phút) ở commit hiện tại:
1. gộp ứng viên ánh xạ của **cả 5 lần chạy** (310 ứng viên), dựng lại bằng code mới nhất;
2. mỗi DN chọn **một** ánh xạ chung (xếp hạng như pipeline, bỏ điểm phạt theo chủ đề ra khỏi xếp hạng chung);
   câu hỏi về **nợ / trả góp / xe / thợ** dùng ứng viên tốt nhất **không có cảnh báo nặng đúng chủ đề đó**, không có thì
   abstain;
3. chạy lại mọi chương trình đã ghi của mọi lần chạy trên ánh xạ đó, bỏ phiếu lại.

Kiểm tra mới trong lần này (chung, không theo DN): nợ trên > 25% số đơn, ghi chú "đã trả đủ" mà C vẫn nợ, sổ nợ không đọc
được, đơn trả / huỷ ghi trong ghi chú mà C coi là hợp lệ, biển số bị bỏ, dòng hàng mất tên hàng; lớp làm sạch: tên viết
tắt khác hoa / thường / dấu được đổi sang tên đầy đủ; toolkit: `paid_at_purchase` + các lần trả sau.

So với v25 (portal 0.825): 967 giống, 28 đổi (dev_014: 19, dev_011: 7), 1 thêm trả lời, 4 thêm abstain (3 câu nợ dev_014
mọi ứng viên đều sai: 22,9 tỉ / 949 khách). 8 abstain, 30 unanswerable, định dạng 0 lỗi. Trên 8 câu kiểm tay bằng dữ liệu
thô: v23 0/8, v25 6/8, **v26 7/8**.
Train oracle 2967 / 1 / 32 không đổi; test CPU ALL PASS; 36 ánh xạ tham chiếu train: 0 cảnh báo sai.
