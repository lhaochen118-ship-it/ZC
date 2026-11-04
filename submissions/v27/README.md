# File nộp public test — hệ v27 (đồng thuận ánh xạ trên 5 lần chạy v25, code mới)

Cùng 5 lần chạy public của v25, chạy lại `src/consensus.py` (CPU, 35 phút) với các sửa chung sau v26:
- kiểm tra ánh xạ: loại máy mới / máy cũ / phụ kiện lệch bảng tra cứu, giá vốn = 0 mọi hàng, dòng hàng mất tên, biển số
  không khớp sổ; tên viết tắt khác hoa / thường / dấu đổi sang tên đầy đủ;
- đọc câu: dòng xe / hãng máy được nêu trong câu luôn giữ bộ lọc ("xe Air Blade khác nhau" không còn đếm mọi xe); "thứ 3" =
  thứ Ba (chương trình dùng sai thứ không được bỏ phiếu);
- toolkit: cột bảo hành toàn False coi như không có (đọc ghi chú), giá trị tồn kho cần giá vốn thật;
- xếp hạng ánh xạ: lỗi câu thuộc chủ đề (nợ / trả góp / xe / thợ) không trừ điểm chung; đồng thuận chỉ tính giữa các ứng viên
  không có cảnh báo nặng.

**31 câu kiểm bằng dữ liệu thô** (dev_008 / 011 / 012 / 014; chỉ để đánh giá, không dùng để chọn đáp án): v25 13 / 31, v26
15 / 31, **v27 31 / 31**. So với v26: 968 giống, 26 đổi (dev_014: 18), 2 thêm trả lời, 4 thêm abstain. 10 abstain, định dạng
0 lỗi. Train oracle 2967 / 1 / 32 không đổi; 36 ánh xạ tham chiếu train: 0 cảnh báo sai. Lần chạy train v28 (code mới):
PA 0.9897.

**Cập nhật (bản thay thế):** bản v27 gửi trước đó để 6 câu "đơn bán chịu ghi ngày phải trả" của dev_004 thành unanswerable
(lỗi xếp ánh xạ theo chủ đề; v25 trả lời đúng, X1 đã xác nhận). File hiện tại đã sửa (slot chủ đề xếp theo khả năng trả lời
câu của chủ đề đó): 37 câu kiểm → 36 / 37; 9 abstain.
