# hand-kept sales book without order ids or customers: every row is one sale record; status in "Ghi chú TT"; base quantity in "Số lượng lẻ"
t = T["SoBanHang"]
lines = frame(t, order_id=pd.Series(range(len(t)), index=t.index).astype(str), order_date="Ngày mua", status=map_status(t["Ghi chú TT"]),
              item_name="Mặt hàng", unit="DonVi", qty="SL bán", qty_base="Số lượng lẻ", unit_price="Đơn giá bán", line_amount="thanh_tien")
orders, lines = finalize(lines, dedupe=False)
C = {"orders": orders, "lines": lines, "conversions": readme_conversions(README)}
NOTES = "No order ids: each row is one record (order counts, customers and order values are not recorded). No codes, purchases, stock or debt."
