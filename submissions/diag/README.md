# File chẩn đoán v23 → v24 (không phải phiên bản mới)

v23 = 0.87, v24 = 0.747: v24 tụt 0.123 trên **121 câu** hai bản trả lời khác nhau. Ba file dưới đây là **v23**, chỉ thay
đáp án của **một nhóm** câu bằng đáp án v24. Các nhóm không chồng nhau, nên điểm cộng dồn chính xác:
Δ_nhóm = điểm_file − 0.87; Δ_còn_lại = −0.123 − (Δ1 + Δ2 + Δ3).

| File | Nhóm (số câu khác nhau) | Thay đổi của v24 được đo | Dự đoán |
|---|---|---|---|
| `D1_dev013.jsonl` | dev_013 (40) | dòng "Voucher" / "Dùng N điểm" / "Phí ship" bị tách khỏi doanh thu (rulebook §1 nói doanh thu không trừ voucher, không gồm phí ship); câu voucher / ship / điểm có đáp án | nếu điểm < 0.87: đáp án gốc tính doanh thu = tổng TT mọi dòng của hoá đơn |
| `D2_dev014.jsonl` | dev_014 (32) | ánh xạ trả góp của v24 (2/3 lần chạy chọn bảng góp thiếu cột) + câu máy / dòng máy / bảo hành mới trả lời | trả góp của v23 khớp toolkit trên ánh xạ đúng → nhiều khả năng âm |
| `D3_dev004.jsonl` | dev_004 (13) | đơn bán chịu theo ngày phải trả chỉ tính đơn **đã đặt tới hôm nay** (v23 đếm cả đơn đặt sau ngày hỏi: "tới 2/3/2025 có 2444 đơn hạn muộn hơn hôm nay"); giao hàng theo đơn đặt | nhiều khả năng dương |

Nhóm còn lại (36 câu: dev_012 9, dev_008 7, dev_011 6, dev_006 4, dev_010 3, dev_003 3, dev_002 2, dev_009 1, dev_007 1):
câu phẩm chất chủ quan → unanswerable, nợ ghi trong ghi chú (dev_008), "o đờ" = đơn, thợ / hoa hồng, máy bán ra...
