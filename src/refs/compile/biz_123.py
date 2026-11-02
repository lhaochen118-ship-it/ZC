# full relational database: orders, lines (cost at sale), payments, returns, stock ledger (the_kho: stock after every
# movement), lots, lots sold per line, purchases, products, customers (with type), unit conversions, price history, drugs
h, d, k, sp = T["don_hang"], T["chi_tiet_don"], T["khach_hang"], T["san_pham"]
orders = frame(h, order_id="MaHoaDon", order_date="NgayLap", order_time="ThoiGianLap", customer_id="MaKhachHang", staff="MaNhanVien",
               channel="KenhBanHang", payment_method="PhuongThucTT", status=map_status(h["TinhTrang"]), order_discount="ChietKhau",
               shipping_fee="PhiVanChuyen", vat="ThueGTGT", total_payable="TongThanhToan", promo="MaKhuyenMai", shift_id="MaCaLam", note="DienGiai")
orders["customer_name"] = orders["customer_id"].map(dict(zip(k["MaKhachHang"], k["TenGoi"])))
orders["customer_group"] = orders["customer_id"].map(dict(zip(k["MaKhachHang"], k["LoaiKhachHang"])))
items = frame(sp, item_code="MaHangHoa", item_name="TenGoi", item_group="DanhMuc", base_unit="DonViCoSo", cost="GiaVonHang", price="GiaBanLe", vat_rate="TyLeThue")
th = T["thong_tin_thuoc"]
items["active_ingredient"] = items["item_code"].map(dict(zip(th["MaHangHoa"], th["ThanhPhanChinh"])))
items["prescription"] = items["item_code"].map(dict(zip(th["MaHangHoa"], th["BanTheoDon"].map(lambda x: str(x).strip().lower() in ("true", "1", "1.0", "có", "x")))))
lines = frame(d, order_id="MaHoaDon", line_no="SoThuTu", item_code="MaHangHoa", unit="DonViTinh", qty="SL", qty_base="SLQuyDoi",
              unit_price="GiaDonVi", line_discount="ChietKhau", line_amount="TriGia", unit_cost="GiaVonLucBan")
orders, lines = finalize(lines, orders, items=items)
q = T["thanh_toan"]
payments = frame(q, order_id="MaHoaDon", pay_time="NgayGioThu", amount="SoTienThu", method="PhuongThuc")
payments["pay_date"] = payments["pay_time"].dt.normalize()
r = T["tra_hang"]
returns = frame(r, return_id="MaPhieuTra", order_id="MaHoaDon", return_time="NgayGioTra", reason="NguyenNhan", refund="TienHoanTra", refund_method="CachHoanTien")
ph, pdt = T["phieu_nhap"], T["chi_tiet_nhap"]
pu = pdt.merge(ph[["MaPhieuNhap", "MaNhaCungCap", "NgayNhapKho", "TinhTrang", "DienGiai"]], on="MaPhieuNhap", how="left")
purchases = frame(pu, receipt_id="MaPhieuNhap", receipt_date="NgayNhapKho", supplier="MaNhaCungCap", item_code="MaHangHoa", unit="DonViTinh",
                  qty="SL", unit_cost="DonGiaNhap", lot_id="MaLo", expiry="HanSuDung", note="DienGiai")
purchases["amount"] = purchases["qty"] * purchases["unit_cost"]
purchases["cancelled"] = pu["TinhTrang"].map(lambda x: canon_status(x) == "huỷ")
ncc = T["nha_cung_cap"]
purchases["supplier_name"] = purchases["supplier"].map(dict(zip(ncc["MaNhaCungCap"], ncc["TenGoi"])))
lo = T["lo_hang"]
lots = frame(lo, lot_id="MaLo", receipt_id="MaPhieuNhap", item_code="MaHangHoa", expiry="HanSuDung", qty_in_base="SLNhapQuyDoi", unit_cost_base="DonGiaNhap")
sl = T["chi_tiet_don_lo"]
lot_sales = frame(sl, order_id="MaHoaDon", line_no="SoThuTu", lot_id="MaLo", qty_base="SLQuyDoi", returned_base="SLTraQuyDoi")
tk = T["the_kho"]
ledger = frame(tk, item_code="MaHangHoa", time="NgayGioPhatSinh", kind="LoaiGiaoDich", qty_base="SLQuyDoi", doc_type="KieuChungTu", doc_id="MaChungTu", stock_after="TonCuoi")
prices = frame(T["lich_su_gia"], item_code="MaHangHoa", date="NgayApDung", price="GiaBanLe")
cv = conversions_table(T["don_vi_sp"], "MaHangHoa", "DonViTinh", "HeSoQuyDoi")
staff = frame(T["nhan_vien"], staff="MaNhanVien", staff_name="TenGoi", role="ChucVu", start="NgayNhanViec", end="NgayThoiViec")
C = {"orders": orders, "lines": lines, "items": items, "payments": payments, "returns": returns, "purchases": purchases, "lots": lots,
     "lot_sales": lot_sales, "ledger": ledger, "prices": prices, "conversions": cv, "staff": staff}
NOTES = "Stock at a time = stock_after of the item's last ledger row up to that time. Customer type (lẻ / thân thiết / sỉ) = customer_group."
