# sales book export: one row per order line, order info repeated on every line; delivery trips; credit terms of credit orders
d, sp = T["So_chi_tiet_ban_hang"], T["Danh s#U00e1ch h#U00e0ng h#U00f3a"]
items = frame(sp, item_code="Mã hàng hóa", item_name="Tên hàng hóa", item_group="Nhóm hàng hóa", base_unit="Đơn vị tính", cost="Giá vốn", price="Giá bán")
lines = frame(d, order_id="Mã đơn hàng", order_date="Ngày", customer_name="Khách hàng", status=map_status(d["Trạng thái"]), item_name="Tên hàng hóa",
              unit="Đơn vị tính", qty="Số lượng", qty_base="Số lượng quy đổi", unit_price="Đơn giá", line_amount="Thành tiền")
orders, lines = finalize(lines, items=items)
C = {"orders": orders, "lines": lines, "items": items, "conversions": readme_conversions(README)}
g = T["B#U1ea3ng k#U00ea phi#U1ebfu giao h#U00e0ng"]
deliveries = frame(g, order_id="Mã đơn hàng", delivery_date="Ngày giao hàng", trip_no="Lần giao", item_name="Tên hàng hóa", unit="Đơn vị tính",
                   qty_base="Số lượng giao", driver="Người giao hàng")
deliveries["plate"] = RAW["B#U1ea3ng k#U00ea phi#U1ebfu giao h#U00e0ng"].loc[g.index, "Biển số xe"].values if "Biển số xe" in RAW["B#U1ea3ng k#U00ea phi#U1ebfu giao h#U00e0ng"].columns else None
h = T["Chi ti#U1ebft c#U00f4ng n#U1ee3 theo h#U1ea1n"]
C["deliveries"] = deliveries
C["terms"] = frame(h, order_id="Mã đơn hàng", credit_days="Số ngày được nợ", due_date="Hạn thanh toán")
NOTES = "No time of day, no payments / debt amounts, no purchases or stock. Deliveries: one row per delivered item per trip (base units). Terms: credit days / due date of credit orders."
