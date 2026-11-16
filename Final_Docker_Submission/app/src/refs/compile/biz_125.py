# restaurant flat sales book: one row per order line (incl. "Phí phục vụ" service lines); base quantity in "Số lượng lẻ"
t = T["ban_hang"]
lines = frame(t, order_id="Số phiếu", order_date="Ngày bán", status=map_status(t["Trạng thái xử lý"]), customer_name="Khách hàng (tên)",
              item_name="Hàng bán", unit="dvt", qty="SL bán", qty_base="Số lượng lẻ", unit_price="Giá 1 đv", line_amount="Amount")
orders, lines = finalize(lines)
lines["is_service"] = lines["item_name"].map(lambda x: key(x).startswith("phi phuc vu"))
C = {"orders": orders, "lines": lines, "conversions": readme_conversions(README)}
NOTES = "No item / customer codes, no time of day, no purchases, stock, costs or debt. 'Phí phục vụ' lines are service fees, not items."
