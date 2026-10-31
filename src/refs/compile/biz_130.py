# sales book export: one row per order line, order info repeated on every line
d, sp = T["So_chi_tiet_ban_hang"], T["Danh s#U00e1ch h#U00e0ng h#U00f3a"]
items = frame(sp, item_code="Mã hàng hóa", item_name="Tên hàng hóa", item_group="Nhóm hàng hóa", base_unit="Đơn vị tính", cost="Giá vốn", price="Giá bán")
lines = frame(d, order_id="Mã đơn hàng", order_date="Ngày", customer_name="Khách hàng", status=map_status(d["Trạng thái"]), item_name="Tên hàng hóa",
              unit="Đơn vị tính", qty="Số lượng", qty_base="Số lượng quy đổi", unit_price="Đơn giá", line_amount="Thành tiền")
orders, lines = finalize(lines, items=items)
C = {"orders": orders, "lines": lines, "items": items, "conversions": readme_conversions(README)}
NOTES = "No time of day, no payments / debt amounts, no purchases or stock. No customer names."
