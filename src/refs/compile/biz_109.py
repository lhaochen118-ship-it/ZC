# POS export with generic sheet names: DuLieuBanHang = invoice headers, _1 products, _2 customers, _3 stock, _4 purchase
# headers, _5 purchase lines, _6 payments (one row per payment), _7 unit conversions
h, d, k, sp = T["DuLieuBanHang"], T["DanhSachHoaDonChiTiet"], T["DuLieuBanHang_2"], T["DuLieuBanHang_1"]
orders = frame(h, order_id="Mã hóa đơn", order_time="Thời gian tạo", order_date="Thời gian", customer_id="Mã KH", customer_name="Khách hàng", channel="Kênh bán", staff="Người bán", payment_method="Phương thức thanh toán", status=map_status(h["Trạng thái"], {"đã lấy": "hoàn thành"}), order_discount="Giảm giá", total_payable="Tổng sau giảm giá")
orders["customer_group"] = orders["customer_id"].map(dict(zip(k["Mã khách hàng"], k["Nhóm khách hàng"])))
items = frame(sp, item_code="Mã hàng", item_group="Nhóm hàng(3 Cấp)", base_unit="ĐVT", cost="Giá vốn", price="Giá bán")
lines = frame(d, order_id="Mã hóa đơn", line_no="STT", item_code="Mã hàng", item_name="Tên hàng", unit="Đơn vị tính", qty="Số lượng", qty_base="Số lượng quy đổi", unit_price="Đơn giá", line_discount="Giảm giá", line_amount="Thành tiền", unit_cost="Giá vốn")
orders, lines = finalize(lines, orders, items=items)
q = T["DuLieuBanHang_6"]
payments = frame(q, order_id="Mã hóa đơn", pay_time="Thời gian", pay_date=q["Thời gian"].dt.normalize(), amount="Giá trị", method="Phương thức thanh toán")
cv = conversions_table(T["DuLieuBanHang_7"], "Mã hàng", "ĐVT", "Quy đổi")
pu = T["DuLieuBanHang_5"].merge(T["DuLieuBanHang_4"], on="Mã nhập hàng", how="left")
purchases = frame(pu, receipt_id="Mã nhập hàng", receipt_date=pu["Thời gian"].dt.normalize(), supplier="Mã NCC", item_code="Mã hàng", unit="ĐVT", qty="Số lượng", unit_cost="Đơn giá", amount=pu["Số lượng"] * pu["Đơn giá"], cancelled=pu["Trạng thái"].map(lambda x: canon_status(x) == "huỷ"), note="Ghi chú")
st = T["DuLieuBanHang_3"]
stock = frame(st, item_code="Mã hàng", item_name="Tên hàng", qty_base="Tồn cuối kỳ")
C = {"orders": orders, "lines": lines, "items": items, "payments": payments, "purchases": purchases, "stock": stock, "conversions": cv}
NOTES = "Status 'đã lấy' = completed. Stock snapshot = end of the last data day (base units)."
