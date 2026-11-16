# hand-kept sales book, one sheet per month: one row per order line; returns are lines with negative quantity in the same
# order; time of day in "Giờ"; purchases, stock snapshot, price list, unit conversions (DanhMuc)
t = T["SoBanHang"]
lines = frame(t, order_id="Mã hóa đơn", order_date="Ngày giao dịch", order_time="Giờ", status=map_status(t["Status"], {"đã lấy": "hoàn thành"}),
              customer_name="Khách", item_name="Hàng", unit="Unit", qty="so_luong", unit_price="Giá lẻ", line_amount="Trị giá")
dmh = T["DanhMuc"]
cv = conversions_table(dmh, "Tên món", "ĐV nhập", "Hệ số", "Đơn vị gốc")
orders, lines = finalize(lines, conversions=cv)
n = T["SoNhapHang"]
purchases = frame(n, receipt_id="Số phiếu", receipt_date="Ngày nhập", supplier="Nhà cung cấp", item_name="Tên mặt hàng", unit="ĐVT",
                  qty="SL nhập", unit_cost="Giá nhập", amount="Thành tiền", note="Ghi chu")
purchases["cancelled"] = False
stock = frame(T["TonKho"], item_name="Hàng hóa", base_unit="Đơn vị tính", qty_base="Tồn cuối")
prices = frame(T["BangGia"], date="Ngày áp dụng", item_name="ten_hang", price="Giá mới", old_price="Giá cũ")
C = {"orders": orders, "lines": lines, "purchases": purchases, "stock": stock, "prices": prices, "conversions": cv}
NOTES = "Returned items are negative-quantity lines (the whole order is 'trả hàng'). Order time from 'Giờ' (for return lines: time of the return). No item codes, no customer codes, no cost price on lines, no debt."
