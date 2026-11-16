# POS export: invoice headers (debt left at the end of the data in "Còn nợ") + lines; customers with groups; purchases; stock
h, d, k = T["DanhSachHoaDon"], T["DanhSachHoaDonChiTiet"], T["DanhSachKhachHang"]
orders = frame(h, order_id="Mã hóa đơn", order_time="Thời gian tạo", order_date="Thời gian", customer_id="Mã KH", customer_name="Khách hàng", channel="Kênh bán", staff="Người bán", status=map_status(h["Trạng thái"]), order_discount="Giảm giá", total_payable="Tổng sau giảm giá", paid_total="Khách đã trả", debt_end="Còn nợ")
orders["customer_group"] = orders["customer_id"].map(dict(zip(k["Mã khách hàng"], k["Nhóm khách hàng"])))
sp = T["DanhSachSanPham"]
items = frame(sp, item_code="Mã hàng", item_group="Nhóm hàng(3 Cấp)", base_unit="ĐVT", price="Giá bán")
lines = frame(d, order_id="Mã hóa đơn", line_no="STT", item_code="Mã hàng", item_name="Tên hàng", unit="Đơn vị tính", qty="Số lượng", qty_base="Số lượng quy đổi", unit_price="Đơn giá", line_discount="Giảm giá", line_amount="Thành tiền")
orders, lines = finalize(lines, orders, items=items)
cv = readme_conversions(README)
ph, pdt = T["DanhSachNhapHang"], T["DanhSachChiTietNhapHang"]
pu = pdt.merge(ph, on="Mã nhập hàng", how="left")
purchases = frame(pu, receipt_id="Mã nhập hàng", receipt_date=pu["Thời gian"].dt.normalize(), supplier="Mã NCC", item_code="Mã hàng", unit="ĐVT", qty="Số lượng", unit_cost="Đơn giá", amount=pu["Số lượng"] * pu["Đơn giá"], cancelled=pu["Trạng thái"].map(lambda x: canon_status(x) == "huỷ"), note="Ghi chú")
stock = pd.DataFrame({"item_code": T["BaoCaoTonKho"]["Mã hàng"], "item_name": T["BaoCaoTonKho"]["Tên hàng"], "qty_base": T["BaoCaoTonKho"]["Tồn cuối kỳ"]})
C = {"orders": orders, "lines": lines, "items": items, "purchases": purchases, "stock": stock, "conversions": cv}
NOTES = "Debt is only known at the end of the data (orders.debt_end); there are no payment dates. No cost price on lines."
