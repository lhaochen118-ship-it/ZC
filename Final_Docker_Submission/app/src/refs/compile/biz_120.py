# pharmacy POS export: invoice headers (debt left at the end in "Còn nợ") + lines, lots sold per line, lots received,
# drug info (active ingredient, prescription), purchases, stock
h, d, k = T["DanhSachHoaDon"], T["DanhSachHoaDonChiTiet"], T["DanhSachKhachHang"]
orders = frame(h, order_id="Mã hóa đơn", order_time="Thời gian tạo", order_date="Thời gian", customer_id="Mã KH", customer_name="Khách hàng", channel="Kênh bán", status=map_status(h["Trạng thái"]), order_discount="Giảm giá", total_payable=h["Tổng tiền hàng"] - h["Giảm giá"], paid_total="Khách đã trả", debt_end="Còn nợ")
sp = T["DanhSachSanPham"]
items = frame(sp, item_code="Mã hàng", item_group="Nhóm hàng(3 Cấp)", base_unit="ĐVT", price="Giá bán")
lines = frame(d, order_id="Mã hóa đơn", line_no=d.groupby("Mã hóa đơn").cumcount() + 1, item_code="Mã hàng", item_name="Tên hàng", qty="Số lượng", qty_base="Số lượng quy đổi", unit_price="Đơn giá", line_discount="Giảm giá", line_amount="Thành tiền")
orders, lines = finalize(lines, orders, items=items)
cv = readme_conversions(README)
ph, pdt = T["DanhSachNhapHang"], T["DanhSachChiTietNhapHang"]
pu = pdt.merge(ph, on="Mã nhập hàng", how="left")
purchases = frame(pu, receipt_id="Mã nhập hàng", receipt_date=pu["Thời gian"].dt.normalize(), supplier="Mã NCC", item_code="Mã hàng", unit="ĐVT", qty="Số lượng", unit_cost="Đơn giá", amount=pu["Số lượng"] * pu["Đơn giá"], lot_id="Lô", expiry="Hạn sử dụng", cancelled=pu["Trạng thái"].map(lambda x: canon_status(x) == "huỷ"), note="Ghi chú")
stock = pd.DataFrame({"item_code": T["BaoCaoTonKho"]["Mã hàng"], "item_name": T["BaoCaoTonKho"]["Tên hàng"], "qty_base": T["BaoCaoTonKho"]["Tồn cuối kỳ"]})
th = T["DanhSachThuoc"]
items["active_ingredient"] = items["item_code"].map(dict(zip(th["Mã hàng"], th["Hoạt chất"])))
items["prescription"] = items["item_code"].map(dict(zip(th["Mã hàng"], th["Thuốc kê đơn"].map(lambda x: str(x).strip().lower() in ("true", "1", "1.0", "có", "x")))))
for c in ("active_ingredient", "prescription"): lines[c] = lines["item_code"].map(dict(zip(items["item_code"], items[c])))
lo = T["DanhSachLoHang"]
lots = frame(lo, lot_id="Lô", receipt_id="Mã nhập hàng", item_code="Mã hàng", expiry="Hạn sử dụng", qty_in_base="Số lượng (ĐVT cơ bản)")
sl = T["DanhSachHoaDonChiTietLo"]
lot_sales = frame(sl, order_id="Mã hóa đơn", line_no="STT", lot_id="Lô", qty_base="Số lượng (ĐVT cơ bản)", returned_base="Số lượng trả (ĐVT cơ bản)")
C = {"lots": lots, "lot_sales": lot_sales, "orders": orders, "lines": lines, "items": items, "purchases": purchases, "stock": stock, "conversions": cv}
NOTES = "Debt is only known at the end of the data (orders.debt_end); no payment dates. No cost price on lines. Lots: one per receipt line."
