# File nộp public test — hệ v28 (đồng thuận ánh xạ trên 8 lần chạy: 5 của v25 + 3 mới)

3 lần chạy public mới (seed 5 / 8 / 9) với code hiện tại (train cùng code: PA 0.9897) + 5 lần chạy v25; seed 6 / 7 không chạy
được (mọi acc hết quota GPU tuần). `src/consensus.py` gộp 491 ứng viên ánh xạ, chọn theo các kiểm tra mới nhất, thêm so với v27:
- slot chủ đề xếp theo khả năng trả lời câu của chủ đề; câu mà ánh xạ chính không trả lời được dùng ứng viên hợp lý tiếp theo
  có chương trình luật trả lời được (dev_004 "đơn bán chịu ghi ngày phải trả trong th2/2026");
- sổ nợ còn nợ mà C không ghi nợ → cảnh báo; ngày trả đọc mềm (dòng tiêu đề lẫn vào dữ liệu không làm hỏng hàm nợ).

37 câu kiểm (dữ liệu thô / X1; chỉ để đánh giá): **37 / 37** (v25: 19, v26: 15+, v27 bản thay thế: 36). So với v27: 998 giống.
10 abstain (3 câu tổng nợ dev_014: mọi ứng viên đều thiếu sổ nợ hoặc sổ góp → abstain thay vì trả lời sai). Định dạng 0 lỗi.
