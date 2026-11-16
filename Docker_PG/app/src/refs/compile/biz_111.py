# relational export: orders, order lines, payments (one row per payment), customers (with group), products
h, d, k, sp, q = T["don_hang"], T["chi_tiet_don"], T["khach_hang"], T["san_pham"], T["thanh_toan"]
orders = frame(h, order_id="Mã chứng từ", order_date="NgayBan", customer_id="Số KH", channel="Bán qua", payment_method="Phương thức TT",
               status=map_status(h["Xử lý"]), total_payable="Tổng bill")
orders["customer_name"] = orders["customer_id"].map(dict(zip(k["ma_kh"], k["Khách lẻ/sỉ"])))
orders["customer_group"] = orders["customer_id"].map(dict(zip(k["ma_kh"], k["Nhóm"])))
items = frame(sp, item_code="SKU", item_name="Sản phẩm", item_group="Nhom hang", base_unit="Đơn vị", price="Giá")
lines = frame(d, order_id="Ma don", line_no="Thứ tự", item_code="ID SP", unit="Đơn vị (tính)", qty="SL", qty_base="SL quy đổi",
              unit_price="Giá đơn vị", line_amount="T.Tiền")
orders, lines = finalize(lines, orders, items=items)
payments = frame(q, order_id="Số HĐ", pay_time="Thời gian thu", amount="Tiền", method="Phương thức")
payments["pay_date"] = payments["pay_time"].dt.normalize()
C = {"orders": orders, "lines": lines, "items": items, "payments": payments, "conversions": readme_conversions(README)}
NOTES = "Customer names come from the customer list (name - phone). No time of day, purchases, stock or costs."
