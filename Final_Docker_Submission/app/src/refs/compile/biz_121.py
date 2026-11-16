# online-shop export: one row per order line (order cells may be blank on later lines); unit conversion table;
# pharmacy: generic sheet names DuLieuXuat_1 products, _2 receipt headers, _3 receipt lines = lots, _4 lots sold per order, _5 drug info
d, q = T["DonHang_ChiTiet"], T["DuLieuXuat"]
lines = frame(d, order_id="Mã đơn hàng", order_date="Ngày đặt hàng", status=map_status(d["Trạng thái đơn hàng"]), customer_name="Tên khách hàng",
              item_name="Tên sản phẩm", unit="Đơn vị tính", qty="Số lượng sản phẩm", unit_price="Giá sản phẩm", line_amount="Tổng tiền")
cv = conversions_table(q, "Tên sản phẩm", "Đơn vị quy đổi", "Tỉ lệ quy đổi", "Đơn vị cơ bản")
sp, th = T["DuLieuXuat_1"], T["DuLieuXuat_5"]
items = frame(sp, item_code="Mã SKU", item_name="Tên sản phẩm", item_group="Loại sản phẩm", base_unit="Đơn vị")
items["active_ingredient"] = items["item_code"].map(dict(zip(th["Mã SKU"], th["Hoạt chất"])))
items["prescription"] = items["item_code"].map(dict(zip(th["Mã SKU"], th["Thuốc kê đơn"].map(lambda x: str(x).strip().lower() in ("true", "1", "1.0", "có", "x")))))
orders, lines = finalize(lines, conversions=cv, items=items)
C = {"orders": orders, "lines": lines, "conversions": cv, "items": items}
ph, pl, ls = T["DuLieuXuat_2"], T["DuLieuXuat_3"], T["DuLieuXuat_4"]
lots = frame(pl.merge(ph, on="Mã đơn nhập", how="left"), receipt_id="Mã đơn nhập", receipt_date="Ngày nhập", supplier="Mã nhà cung cấp",
             item_code="Mã SKU", lot_id="Mã lô", expiry="Hạn sử dụng", qty_in_base="Số lượng nhập")
C["lots"] = lots
C["lot_sales"] = frame(ls, order_id="Mã đơn hàng", lot_id="Mã lô", qty_base="Số lượng", returned_base="Số lượng trả")
NOTES = "No item codes, no customer codes (customers by name), no time of day, no payments, purchases or stock. Lots: one per receipt line; no purchase prices."
