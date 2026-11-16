#!/usr/bin/env python3
"""General Vietnamese retail / service glossary -> lexicon cards for the rule-card library (src/refs/lrl/glossary_vi.jsonl).

Written from general knowledge of Vietnamese commerce (abbreviations and chat spellings, money / quantity slang, units) and of the
five kinds of shop in the organisers' TRAINING split (grocery, café / milk tea, pharmacy, building materials, restaurant / beer
place). No test split was read. Cards only explain WORDS (meaning, unit, abbreviation); every computation rule stays the split's
own rulebook, which takes precedence. At run time a card reaches a question's prompt only when one of its trigger words occurs in
the question (src/lrl.py Library.match; triggers frequent in the run's own questions are pruned).
Usage: python3 training/data/build_glossary.py  (rewrites src/refs/lrl/glossary_vi.jsonl)"""
import json, os, hashlib

G = [
 # --- general commerce: abbreviations and chat spellings
 (["dt", "dthu", "d.thu", "doanh so"], "'dt' / 'dthu' / 'doanh số' (trong câu hỏi về bán hàng) = doanh thu"),
 (["sp", "s.p"], "'sp' = sản phẩm (mặt hàng)"),
 (["kh", "k.h"], "'kh' = khách hàng"),
 (["ncc", "nha cc"], "'ncc' = nhà cung cấp"),
 (["sl", "s.l"], "'sl' = số lượng"),
 (["dh", "đh", "đơn hàng"], "'đh' = đơn hàng"),
 (["hđ", "hd", "hoá đơn", "bill"], "'hđ' / 'bill' = hoá đơn (một đơn bán)"),
 (["ck"], "'ck' = chuyển khoản (cách thanh toán) hoặc chiết khấu (giảm giá): đọc theo ngữ cảnh câu hỏi"),
 (["tm", "tiền mặt", "cash"], "'tm' / 'cash' = tiền mặt (cách thanh toán)"),
 (["nv", "n.v"], "'nv' = nhân viên"),
 (["tb", "t.b", "bq", "bình quân"], "'tb' / 'bq' / 'bình quân' = trung bình"),
 (["bnhiu", "bnhieu", "nhiêu", "bn"], "'bnhiu' / 'bn' / 'nhiêu' = bao nhiêu"),
 (["dc", "đc", "dk", "đk"], "'đc' / 'dc' / 'đk' = được"),
 (["ko", "k0", "hong", "hông", "khum"], "'ko' / 'hông' / 'khum' = không"),
 (["vs"], "'vs' = với (hoặc 'vệ sinh' khi nói về dịch vụ)"),
 (["hqua", "hôm wa", "hom wa", "qua"], "'hqua' / 'hôm wa' = hôm qua"),
 (["hnay", "hn", "bữa nay", "bua nay"], "'hnay' / 'bữa nay' = hôm nay"),
 (["bữa qua", "bua qua", "bữa trước"], "'bữa qua' = hôm qua; 'bữa' = ngày (miền Nam)"),
 (["tuần rồi", "tuan roi", "tuần qua"], "'tuần rồi' / 'tuần qua' = tuần trước (tuần lịch liền trước tuần này)"),
 (["tháng rồi", "thang roi"], "'tháng rồi' = tháng trước"),
 (["năm ngoái", "nam ngoai", "năm rồi", "năm trước"], "'năm ngoái' / 'năm rồi' = năm trước năm nay"),
 (["cùng kỳ", "cung ky"], "'cùng kỳ' = cùng khoảng thời gian đó của kỳ trước (tháng trước / năm trước) để so sánh"),
 (["ytd", "lũy kế", "luỹ kế", "luy ke"], "'luỹ kế' / 'YTD' = cộng dồn từ đầu kỳ (đầu năm) đến ngày hỏi"),
 (["mom", "tháng so tháng"], "'MoM' = so với tháng trước; 'YoY' = so với cùng kỳ năm trước"),
 (["yoy"], "'YoY' = so với cùng kỳ năm trước"),
 (["top"], "'top N' = N đứng đầu theo thứ hạng"),
 # money / number slang
 (["củ", "cu"], "'củ' (tiền) = triệu đồng: '5 củ' = 5.000.000đ"),
 (["tr", "tr đồng", "triệu"], "'tr' = triệu đồng"),
 (["lít", "lit"], "'lít' (tiền, tiếng lóng) = 100.000đ: '3 lít' = 300.000đ; với chất lỏng thì là đơn vị thể tích"),
 (["tỏi", "toi"], "'tỏi' (tiền, tiếng lóng) = tỷ đồng"),
 (["k", "nghìn", "ngàn", "nghin", "ngan"], "'k' / 'nghìn' / 'ngàn' = 1.000đ: '50k' = 50.000đ"),
 (["chục"], "'chục' = mười: 'mấy chục' = vài chục"),
 # payment, debt
 (["ghi sổ", "ghi so", "mua chịu", "mua chiu", "bán chịu", "ban chiu", "ghi nợ", "ghi no", "thiếu", "còn thiếu"], "'ghi sổ' / 'mua chịu' / 'còn thiếu' = mua nợ, chưa trả đủ tiền (công nợ của khách)"),
 (["công nợ", "cong no", "nợ đọng"], "'công nợ' = tiền khách (hoặc shop) còn nợ chưa trả"),
 (["cod"], "'COD' = thu tiền khi giao hàng"),
 (["quẹt thẻ", "quet the", "pos"], "'quẹt thẻ' = thanh toán bằng thẻ ngân hàng"),
 (["ví điện tử", "vi dien tu", "momo", "zalopay", "vnpay"], "'MoMo' / 'ZaloPay' / 'VNPay' = ví điện tử (cách thanh toán)"),
 # sales notions (meaning only; how to compute is the rulebook's)
 (["lãi", "lời", "loi nhuan", "lợi nhuận", "lãi gộp", "lời gộp"], "'lãi' / 'lời' = lợi nhuận (cách tính theo rulebook của shop, thường là doanh thu trừ giá vốn)"),
 (["vốn", "giá vốn", "tiền vốn", "gia von"], "'vốn' / 'giá vốn' = chi phí mua vào của hàng đã bán"),
 (["tồn", "tồn kho", "còn trong kho", "ton kho"], "'tồn' / 'tồn kho' = hàng còn lại trong kho"),
 (["nhập hàng", "nhập về", "nhap hang", "lấy hàng", "lay hang"], "'nhập hàng' / 'lấy hàng' = mua hàng vào từ nhà cung cấp"),
 (["hàng trả", "trả hàng", "khách trả", "hoàn hàng", "đổi trả"], "'trả hàng' / 'hoàn hàng' = khách trả lại hàng đã mua"),
 (["huỷ", "hủy", "huy", "bom hàng", "bom hang"], "'huỷ' / 'bom hàng' = đơn bị huỷ, không thành"),
 (["ship", "giao", "giao hàng", "vận chuyển"], "'ship' = giao hàng; 'phí ship' = phí giao hàng"),
 (["bán chạy", "ban chay", "đắt hàng", "hot"], "'bán chạy' / 'đắt hàng' = bán được nhiều"),
 (["ế", "bán chậm", "ban cham", "ít người mua"], "'ế' / 'bán chậm' = bán được ít"),
 (["khách sỉ", "bán sỉ", "ban si", "mua sỉ"], "'sỉ' = bán buôn số lượng lớn; 'lẻ' = bán lẻ từng món"),
 (["khách lẻ", "bán lẻ", "ban le", "khách vãng lai"], "'khách lẻ' / 'khách vãng lai' = khách không có tên / mã trong danh sách khách"),
 # --- grocery (tạp hoá)
 (["tạp hoá", "tạp hóa", "tap hoa", "tiệm tạp hoá"], "'tạp hoá' = cửa hàng bán hàng tiêu dùng (đồ ăn, đồ uống, hoá mỹ phẩm)"),
 (["lốc", "loc"], "'lốc' = gói nhiều hộp / chai bán cùng (thường 4 hoặc 6); hệ số quy đổi theo bảng đơn vị của shop"),
 (["thùng", "thung", "két", "ket"], "'thùng' / 'két' = đơn vị lớn đóng gói nhiều chai / lon; hệ số quy đổi theo bảng đơn vị của shop"),
 (["bịch", "bich", "túi"], "'bịch' / 'túi' = gói (miền Nam: bịch)"),
 (["lạng", "lang"], "'lạng' = 100 gam"),
 (["yến", "yen"], "'yến' = 10 kg; 'tạ' = 100 kg; 'tấn' = 1000 kg"),
 (["tạ", "ta"], "'tạ' = 100 kg"),
 (["cân", "can", "ký", "ky", "kí"], "'cân' / 'ký' / 'kí' = kilôgam"),
 (["nước ngọt", "nuoc ngot"], "'nước ngọt' = đồ uống có ga / đóng chai (Coca, Pepsi, 7Up...)"),
 # --- café / milk tea
 (["size m", "size l", "ly lớn", "ly nhỏ"], "size của đồ uống: M = ly vừa / nhỏ, L = ly lớn; tên món có '(M)' / '(L)'"),
 (["topping", "trân châu", "tran chau", "thạch", "pudding"], "'topping' = phần thêm vào đồ uống (trân châu, thạch, pudding...)"),
 (["mang đi", "mang di", "take away", "takeaway", "tại chỗ", "tai cho", "uống tại quán"], "'mang đi' / 'take away' = khách mua mang về; 'tại chỗ' = dùng tại quán"),
 (["ly", "cốc", "coc"], "'ly' / 'cốc' = một phần đồ uống"),
 (["cf", "cafe", "cà fê", "cafe sữa", "cf sữa", "bạc xỉu", "bac xiu"], "'cf' / 'cafe' = cà phê; 'bạc xỉu' = cà phê nhiều sữa"),
 (["ts", "trà sữa", "tra sua"], "'ts' = trà sữa"),
 (["sinh tố", "sinh to", "nước ép", "nuoc ep"], "'sinh tố' = trái cây xay với sữa / đá; 'nước ép' = nước trái cây ép"),
 # --- pharmacy
 (["nhà thuốc", "nha thuoc", "hiệu thuốc", "quầy thuốc"], "'nhà thuốc' / 'hiệu thuốc' / 'quầy thuốc' = cửa hàng bán thuốc"),
 (["kê đơn", "ke don", "thuốc kê đơn", "rx", "etc"], "'thuốc kê đơn' (Rx, ETC) = thuốc cần đơn của bác sĩ; xác định theo danh mục thuốc của shop"),
 (["không kê đơn", "otc", "thuốc otc"], "'không kê đơn' (OTC) = thuốc bán không cần đơn"),
 (["hoạt chất", "hoat chat", "thành phần"], "'hoạt chất' = chất có tác dụng chữa bệnh trong thuốc (paracetamol, ibuprofen...), ghi trong danh mục thuốc"),
 (["biệt dược", "biet duoc", "tên thương mại"], "'biệt dược' = tên thương mại của thuốc (khác tên hoạt chất)"),
 (["hàm lượng", "ham luong", "mg"], "'hàm lượng' = lượng hoạt chất mỗi đơn vị (mg, g, ml, %)"),
 (["vỉ", "vi"], "'vỉ' = tấm đựng nhiều viên thuốc; quy cách 'hộp N vỉ x M viên' = một hộp có N×M viên"),
 (["viên", "vien", "viên nén", "viên nang", "viên sủi"], "'viên' = đơn vị cơ bản của thuốc dạng viên (nén, nang, sủi)"),
 (["siro", "si rô", "hỗn dịch", "hon dich"], "'siro' / 'hỗn dịch' = thuốc dạng lỏng, đóng chai (ml)"),
 (["tuýp", "tuyp"], "'tuýp' = ống kem / gel bôi"),
 (["lô", "lo", "số lô", "so lo"], "'lô' = một đợt nhập của một mặt hàng, có số lô và hạn dùng riêng"),
 (["hạn dùng", "han dung", "hsd", "date", "hết date", "cận date", "can date", "hết hạn", "het han"], "'HSD' / 'date' = hạn sử dụng; 'cận date' = sắp hết hạn; 'hết date' = đã quá hạn"),
 (["tpcn", "thực phẩm chức năng", "thuc pham chuc nang"], "'TPCN' = thực phẩm chức năng (vitamin, bổ sung), không phải thuốc"),
 # --- building materials
 (["vlxd", "vật liệu xây dựng", "vat lieu xay dung"], "'VLXD' = vật liệu xây dựng (xi măng, thép, cát, đá, gạch, tôn, sơn, ống nước, dây điện)"),
 (["sắt", "sat", "thép", "thep", "phi", "fi"], "'sắt' = thép xây dựng; 'phi' / 'fi' / 'D' + số = đường kính thanh thép tính bằng mm (D10 = phi 10)"),
 (["cây", "cay"], "'cây' = một thanh (thép, ống nước) theo chiều dài ghi trong tên hàng"),
 (["bao", "bao xi"], "'bao' = bao xi măng (thường 50 kg)"),
 (["khối", "khoi", "m3"], "'khối' = mét khối (m3), đơn vị của cát, đá, bê tông"),
 (["xe cát", "xe đá", "chuyến"], "'chuyến' / 'xe' (cát, đá) = một lần chở giao hàng"),
 (["đợt giao", "dot giao", "giao nhiều đợt"], "'đợt giao' = một lần giao một phần của đơn hàng; một đơn có thể giao nhiều đợt"),
 (["cuộn", "cuon"], "'cuộn' = cuộn dây (điện, ống mềm) theo chiều dài ghi trong tên hàng"),
 (["tấm", "tam"], "'tấm' = tấm tôn, thạch cao, ván"),
 (["dem", "zem", "ly"], "'dem' / 'zem' (tôn) = 0,1 mm độ dày: tôn 4 dem = 0,4 mm; 'ly' = milimét"),
 (["thùng sơn", "lon sơn"], "'thùng' / 'lon' (sơn) = đơn vị đóng gói theo lít ghi trong tên hàng"),
 (["công trình", "cong trinh", "thầu", "thợ hồ", "tho ho"], "'công trình' = nơi xây dựng khách mua hàng giao tới; 'thợ hồ' = thợ xây"),
 # --- restaurant / beer place
 (["quán nhậu", "quan nhau", "nhà hàng", "quán ăn"], "'quán nhậu' / 'quán ăn' = quán phục vụ món ăn, bia tại bàn"),
 (["bàn", "ban"], "'bàn' = bàn khách ngồi tại quán; một đơn ăn tại quán gắn với một bàn"),
 (["món", "mon", "phần", "suất", "dĩa", "đĩa"], "'món' = một món ăn trong thực đơn; 'phần' / 'suất' / 'đĩa' = một khẩu phần"),
 (["lẩu", "lau"], "'lẩu' = món nấu tại bàn, thường bán theo nồi / phần"),
 (["khui", "két bia", "ket bia"], "'khui' = mở (bia); 'két bia' = 20–24 chai, 'thùng' = 24 lon"),
 (["phí phục vụ", "phi phuc vu", "service charge"], "'phí phục vụ' = khoản thu thêm trên hoá đơn"),
 (["ca sáng", "ca chiều", "ca tối", "ca trưa"], "'ca' = khung giờ làm việc trong ngày (sáng / trưa / chiều / tối)"),
]


# accent-free single words that collide with common words once accents are dropped ('ban' = bán / bàn, 'bao' = bao nhiêu, 'ky' = kỳ,
# 'phi' = phí, 'toi' = tôi, 'qua' = qua ...): never a trigger; their accented forms stay (short words match with their accents)
COLLIDE = {"qua", "ban", "bao", "cay", "tam", "lo", "vi", "ly", "cu", "toi", "ta", "can", "ky", "lit", "yen", "loc", "lang", "mon",
           "lau", "sat", "phi", "fi", "dem", "hong", "hn", "ket", "vien", "giao", "thung", "hot", "date", "tr", "triệu", "chục", "ship",
           "lãi", "lời", "vốn", "tồn", "món", "bàn", "dĩa", "đĩa", "phần", "ly", "cốc", "lô", "viên", "cây", "tấm", "cuộn", "bao", "huỷ", "hủy",
           "huy", "nghìn", "ngàn", "nghin", "ngan", "thiếu", "top", "mg", "giao hàng"}


def main():
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src", "refs", "lrl", "glossary_vi.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for words, content in G:
            words = [w for w in words if w.lower() not in COLLIDE]
            if not words: continue
            cid = "G" + hashlib.md5(content.encode()).hexdigest()[:8]
            f.write(json.dumps({"kind": "lexicon", "trigger": {"words": words}, "content": content, "code_hint": None,
                                "source": "glossary:general", "id": cid}, ensure_ascii=False) + "\n")
    print(f"{len(G)} cards -> {out}")


if __name__ == "__main__":
    main()
