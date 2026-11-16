# hand-kept sales book: one row per order line; returns are negative-quantity lines of the same order; time in "Giờ";
# deliveries (trips), credit terms, purchases, stock snapshot, price list, unit conversions (DanhMuc)
t = T["SoBanHang"]
lines = frame(t, order_id="Mã phiếu", order_date="Ngày lập", order_time="Giờ", status=map_status(t["Ghi chú TT"], {"đã lấy": "hoàn thành"}),
              customer_name="TenKH", item_name="Tên SP", unit="DonVi", qty="Số lượng hàng", unit_price="Giá", line_amount="Tiền")
cv = conversions_table(T["DanhMuc"], "Hàng", "ĐV lớn", "Hệ số QĐ", "ĐV nhỏ")
orders, lines = finalize(lines, conversions=cv)
n = T["SoNhapHang"]
purchases = frame(n, receipt_id="Số phiếu", receipt_date="Ngày nhập", supplier="Nhà cung cấp", item_name="Product", unit="ĐVT",
                  qty="SL nhập", unit_cost="Giá nhập", amount="Thành tiền", note="Nội dung")
purchases["cancelled"] = False
stock = frame(T["TonKho"], item_name="Product", base_unit="Đơn vị", qty_base="Tồn thực")
prices = frame(T["BangGia"], date="Ngày áp dụng", item_name="TenHang", price="Giá mới", old_price="Giá cũ")
g = T["SoGiaoHang"]
deliveries = frame(g, order_id="Mã bill", delivery_date="Ngày chở", trip_no="Lượt giao", item_name="Tên hàng", base_unit="ĐV gốc",
                   qty_base="SL thực giao", driver="Người chở")
deliveries["plate"] = RAW["SoGiaoHang"].loc[g.index, "Biển số xe"].values
terms = frame(T["SoHanNo"], order_id="So hoa don", credit_days="Hạn nợ (ngày)", due_date="Han tra")
C = {"orders": orders, "lines": lines, "purchases": purchases, "stock": stock, "prices": prices, "deliveries": deliveries, "terms": terms, "conversions": cv}
NOTES = "Returned items are negative-quantity lines (the whole order is 'trả hàng'). No item codes, no customer codes, no cost price on lines, no payments."
