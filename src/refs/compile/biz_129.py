# restaurant relational export: orders, lines (incl. "Phí phục vụ"), payments, customers (with group), products
h, d, k, sp, q = T["don_hang"], T["chi_tiet_don"], T["khach_hang"], T["san_pham"], T["thanh_toan"]
orders = frame(h, order_id="Số phiếu", order_date="Ngày bán", customer_id="Mã khách mua", payment_method="Hình thức thanh toán",
               status=map_status(h["Tình trạng đơn"]), total_payable="Phải thu", debt_end="Dư nợ")
orders["customer_name"] = orders["customer_id"].map(dict(zip(k["Mã khách lẻ/sỉ"], k["KH"])))
orders["customer_group"] = orders["customer_id"].map(dict(zip(k["Mã khách lẻ/sỉ"], k["Nhóm"])))
items = frame(sp, item_code="Mã HH", item_name="Tên sản phẩm", item_group="Nhom hang", base_unit="Đơn vị tính", price="Giá hiện tại")
lines = frame(d, order_id="Số CT", line_no="TT dòng", item_code="Mã hàng bán", unit="ĐV tính", qty="S.L", qty_base="Quy ra ĐV lẻ",
              unit_price="Giá 1 đv", line_discount="Giảm giá", line_amount="Tiền bán")
orders, lines = finalize(lines, orders, items=items)
lines["is_service"] = lines["item_name"].map(lambda x: key(x).startswith("phi phuc vu"))
payments = frame(q, order_id="Mã giao dịch", pay_time="Thời điểm", amount="Tiền", method="PTTT")
payments["pay_date"] = payments["pay_time"].dt.normalize()
C = {"orders": orders, "lines": lines, "items": items, "payments": payments, "conversions": readme_conversions(README)}
NOTES = "No time of day, purchases, stock or costs. 'Phí phục vụ' lines are service fees, not items."
