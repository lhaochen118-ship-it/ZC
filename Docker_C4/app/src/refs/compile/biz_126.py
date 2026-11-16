# restaurant hand-kept sales book without order ids or customers: every row is one sale record (incl. "Phí phục vụ" service lines)
t = T["SoBanHang"]
lines = frame(t, order_id=pd.Series(range(len(t)), index=t.index).astype(str), order_date="Ngày tháng", status=map_status(t["Trạng thái đơn"]),
              item_name="Ten hang", unit="Đơn vị tính", qty="Số lượng bán", qty_base="SL lẻ", unit_price="Price", line_amount="Thành tiền (đ)")
orders, lines = finalize(lines, dedupe=False)
lines["is_service"] = lines["item_name"].map(lambda x: key(x).startswith("phi phuc vu"))
C = {"orders": orders, "lines": lines, "conversions": readme_conversions(README)}
NOTES = "No order ids: each row is one record (order counts, customers and order values are not recorded). No codes, purchases, stock or debt."
