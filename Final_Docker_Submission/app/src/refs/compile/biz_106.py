# POS export: invoice headers + invoice lines; payments (one row per payment) in SoQuy; catalogue, stock, purchases
h, d = T["DanhSachHoaDon"], T["DanhSachHoaDonChiTiet"]
orders = frame(h, order_id="Mã hóa đơn", order_time="Thời gian tạo", order_date="Thời gian", customer_id="Mã KH", customer_name="Khách hàng", payment_method="Phương thức thanh toán", status=map_status(h["Trạng thái"]), order_discount="Giảm giá", total_payable="Tổng sau giảm giá")
items = pd.DataFrame({"item_code": T["DanhSachSanPham"]["Mã hàng"], "item_group": T["DanhSachSanPham"]["Nhóm hàng(3 Cấp)"],
                      "base_unit": T["DanhSachSanPham"]["ĐVT"], "cost": T["DanhSachSanPham"]["Giá vốn"], "price": T["DanhSachSanPham"]["Giá bán"]})
lines = frame(d, order_id="Mã hóa đơn", line_no="STT", item_code="Mã hàng", item_name="Tên hàng", unit="Đơn vị tính", qty="Số lượng", qty_base="Số lượng quy đổi", unit_price="Đơn giá", line_discount="Giảm giá", line_amount="Thành tiền", unit_cost="Giá vốn")
orders, lines = finalize(lines, orders, items=items)
q = T["SoQuy"]
payments = frame(q, order_id="Mã hóa đơn", pay_time="Thời gian", pay_date=q["Thời gian"].dt.normalize(), amount="Giá trị", method="Phương thức thanh toán")
cv = conversions_table(T["DanhSachDonViTinh"], "Mã hàng", "ĐVT", "Quy đổi")
ph, pdt = T["DanhSachNhapHang"], T["DanhSachChiTietNhapHang"]
purchases = pdt.merge(ph, on="Mã nhập hàng", how="left")
purchases = pd.DataFrame({"receipt_id": purchases["Mã nhập hàng"], "receipt_date": purchases["Thời gian"].dt.normalize(),
                          "supplier": purchases["Mã NCC"], "item_code": purchases["Mã hàng"], "unit": purchases["ĐVT"],
                          "qty": purchases["Số lượng"], "unit_cost": purchases["Đơn giá"], "amount": purchases["Số lượng"] * purchases["Đơn giá"],
                          "cancelled": purchases["Trạng thái"].map(lambda x: "huỷ" == canon_status(x)), "note": purchases["Ghi chú"]})
stock = pd.DataFrame({"item_code": T["BaoCaoTonKho"]["Mã hàng"], "item_name": T["BaoCaoTonKho"]["Tên hàng"], "qty_base": T["BaoCaoTonKho"]["Tồn cuối kỳ"]})
C = {"orders": orders, "lines": lines, "items": items, "payments": payments, "purchases": purchases, "stock": stock, "conversions": cv}
NOTES = "Stock snapshot = end of the last data day (base units). Supplier only as code. No staff, channel or customer group."
