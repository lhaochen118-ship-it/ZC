# restaurant relational export: orders (debt left at the end in "Còn nợ"), lines (incl. "Phí phục vụ"), payments, customers, products
h, d, k, sp, q = T["don_hang"], T["chi_tiet_don"], T["khach_hang"], T["san_pham"], T["thanh_toan"]
orders = frame(h, order_id="Mã chứng từ", order_date="Ngày", customer_id="Customer ID", channel="Loại bán", payment_method="Trả bằng",
               status=map_status(h["Tình trạng"]), total_payable="Tổng đơn", debt_end="Còn nợ")
orders["customer_name"] = orders["customer_id"].map(dict(zip(k["Mã KH"], k["Khách hàng (tên)"])))
items = frame(sp, item_code="Mã sản phẩm bán", item_name="Tên SP", item_group="Danh mục", base_unit="Đơn vị cơ bản", price="Giá niêm yết")
lines = frame(d, order_id="Mã đơn hàng", line_no="No.", item_code="Mã mặt hàng", unit="DVT", qty="SoLuong", qty_base="Số lượng lẻ",
              unit_price="Đơn giá (đ)", line_amount="Amount")
orders, lines = finalize(lines, orders, items=items)
lines["is_service"] = lines["item_name"].map(lambda x: key(x).startswith("phi phuc vu"))
payments = frame(q, order_id="Số CT", pay_time="Ngày thu", amount="Giá trị", method="HTTT")
payments["pay_date"] = payments["pay_time"].dt.normalize()
C = {"orders": orders, "lines": lines, "items": items, "payments": payments, "conversions": readme_conversions(README)}
NOTES = "No customer groups, time of day, purchases, stock or costs. 'Phí phục vụ' lines are service fees, not items."
