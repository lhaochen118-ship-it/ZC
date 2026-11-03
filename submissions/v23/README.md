# File nộp public test — hệ v23

Code: commit `92dd902` = v22 + toolkit ngành mới của rulebook public (`src/industry.py`).
- **Điện thoại:** máy / phụ kiện, dòng máy, hãng, thu máy cũ, khách phải trả, trả góp (còn lại / quá hạn / kỳ tới hạn) tính vào nợ, bảo hành.
- **Sửa xe:** phiếu, việc trong ô "Làm gì", xe theo biển số, dòng xe viết tắt, thợ, hoa hồng.
- **Mẹ và bé:** voucher / điểm / ship, thành viên mới, tỉ lệ bịch bỉm theo size / hãng.

Bảng chuẩn của ngành đưa vào prompt ánh xạ; toolkit đưa vào prompt trả lời, chỉ cho DN thuộc ngành đó. Ngành được nhận
theo README hoặc theo dữ liệu (dev_010). Mẫu escalation nhận cùng ngữ cảnh với mẫu đầu.

`submission.jsonl` = bỏ phiếu 3 lần chạy (seed 0 / 1 / 2) có tie-break theo ánh xạ. Đã kiểm định dạng: 1000 id, 0 lỗi,
**49 câu abstain** (v22: 76), 34 câu unanswerable, 4 đáp án bằng 0.

- Train (một lần chạy, `linhmai24/zc-v23-1`): PA 0.978. Chỉ biz_123 tụt (0.48): 15/16 lần thử ánh xạ lỗi tên cột, may rủi
  ánh xạ của DN khó đã biết. 29 DN còn lại ≥ 0.98, như v22.
- Ba lần chạy public: 834 nhất trí / 101 đa số / 65 tách.
- Ánh xạ LLM giờ dựng bảng ngành: dev_014 có instalments, instalment_payments, tradeins, warranty; dev_012 có staff;
  dev_013 có customers.
- So với v22 (`submissions/v22`, portal **0.71**): 903 câu giống hệt; 48 câu thêm trả lời (dev_014: 29); 21 câu thêm
  abstain; 28 câu đổi đáp án (dev_013: 7, dev_008: 6, dev_014: 6, dev_011: 5, dev_012: 4).
- Lỗi đã thấy trên log v23, sửa trong v24: dev_012 có ánh xạ tự viết bảng tên gọi → họ tên sai; mẫu bảng thô của ngành
  mới kéo đáp án về 0; dev_008 ánh xạ coi mọi đơn đã trả đủ; dev_010 mất trọng số luật chuẩn.
