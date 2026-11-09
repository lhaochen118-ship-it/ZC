# 10 — Phân tích lỗi sau kế hoạch 1 (Bước 2 của `docs/13_TASK_AFTER_PLAN1.md`)

*09/11/2026. Chỉ phân tích câu của **public gốc** (đáp án `data/raw/public_gt.jsonl`, chỉ để chấm) và **train** (DN không giữ ngoài);
held-out (pub_ho, tr_ho) chỉ dùng số tổng. Công cụ: `tests/list_wrong.py` (liệt kê từng câu sai: id, câu hỏi, đáp án đúng,
đáp án mô hình, code, log), `tests/hint_audit.py` (đo gợi ý trên CPU), `tests/replay_vote.py` (phát lại bước bỏ phiếu trên mẫu đã log).*

## 1. Mốc

| Lần chạy (cấu hình cuối WP7: súp WF1–4 α 0.35, prompt full, think vòng 2) | Đúng | Sai | Bỏ | PA public gốc | pub_ho |
|---|---|---|---|---|---|
| `pf_wf035_think` (phân tích dưới đây) | 956 | 17 | 27 | **0.939** | 0.592 |
| `pf_wf05_think` (α 0.5) | 955 | 16 | 29 | 0.939 | 0.588 |
| `pf_wf05_think_s1` (α 0.5, seed 1) | 951 | 21 | 28 | 0.930 | 0.586 |
| `final_predict` (đường container `predict_flow.py` + `configs/flow_final.json`, 4×L4, 1423 s) | 955 | 17 | 28 | 0.938 | — |

44 câu chưa đúng ở lần tốt nhất; hợp 3 lần: 63 câu; sai ở cả 3 lần: 29 câu (lỗi ổn định). Danh sách đầy đủ từng câu (câu
hỏi, đáp án đúng, đáp án mô hình, phiếu bầu, gợi ý, mọi mẫu: plan / code / precheck / kết quả chạy / postcheck):
`tests/list_wrong.py <run> --md wrong_pub.md` (file sinh ra, không commit vì chứa đáp án public).

## 2. Phân loại 44 câu (lần `pf_wf035_think`) theo taxonomy 7 nhóm

Mỗi câu một nhóm chính (nguyên nhân gần nhất); cột "gốc" = mức mô hình (M) hay hệ thống (H).

| # | Nhóm | Số câu | Tỉ lệ | Câu (id) |
|---|---|---|---|---|
| 1 | Hiểu sai ý định (kỳ, phép đo, đơn vị đếm, câu không trả lời được) | 10 | 23 % | 35b891c4, 611bf24d, 6830bf77, a897b21c (dữ liệu không ghi — gợi ý có, mô hình bỏ qua); 3b309727 (T1/2025 viết thành quý dù gợi ý là tháng); 4809f845, f5fae09c ("mấy bữa" = số ngày, mô hình đếm đơn); 569dcd86, 8ce76594 (lượt xe nhưng hàm mặc định doanh thu); d99e832f (tổng cả máy mới lẫn cũ, gợi ý sai "list") |
| 2 | Schema / entity linking | 1 | 2 % | 51621c16 (nhóm "máy mới" → mô hình kết luận không có) |
| 3 | Code đúng ý nhưng không chạy / bị cắt / hết lượt sửa | 12 | 27 % | 06de4545, 0ca19cd3, 1c004627, 3da61213, 7305db18, 94cca7f3, a18fefbb, af96aaf1, b88ec465, c3439f11, fb409424, fbaf2023 |
| 4 | Sai logic nghiệp vụ (định nghĩa rulebook không được đưa ra / không theo) | 7 | 16 % | a7dd46bc ("ưa chuộng nhất" = có mặt trong nhiều đơn nhất); 5a5a452f ("chênh ... mấy %" = % thay đổi, mô hình lấy tỉ số); 033e37dd, 95427850, c6bfd1c2, b362a46f, c28a03d7 ("bán được bao nhiêu <đơn vị>" = chỉ mặt hàng có đơn vị cơ bản đó) |
| 5 | Định dạng đáp án | 4 | 9 % | 5c0acaa8, 7c8c12f2, b2eb1684 (`[5]` và `5` tách phiếu → bỏ); d41b61e3 (`"2025-07"` kiểu string thay vì date) |
| 6 | Câu hỏi bẩn / làm sạch sai | 4 | 9 % | 00bf8df3, fe5788da ("quan" → "quân" do tên khách, phải là "quán"); 4d20e306 ("chay hang" → "chạy hàng", phải là "cháy hàng" = hết hàng); 8a1878ff ("bnhiu may Samsung" → "mấy", phải là "máy") |
| 7 | Chọn sai giữa nhiều ứng viên / điểm tin cậy | 6 | 14 % | 701bf339, 78b9ca05, a2935540, a652f5fc, d08785ea (7/8 mẫu đúng nhưng mỗi phiếu bị nửa trọng số vì cờ postcheck từ gợi ý kiểu sai → bỏ); 393267cf (4 đúng / 2 "unanswerable") |

Không có câu nào sai do ingest dữ liệu (nhóm 6 ở đây là câu hỏi bẩn, không phải bảng bẩn).

## 3. Nguyên nhân gốc chung (đo được, không theo từng câu)

| Mã | Phát hiện | Số liệu | Mức | Nhóm bị ảnh hưởng |
|---|---|---|---|---|
| F1 | **Mẫu bị cắt ở trần 512 token**: súp LoRA viết lập luận dạng văn bản trước `<plan>` (rò "suy nghĩ" khi tắt thinking) | 10.7 % lượt sinh không có khối code (gốc Qwen3.5-9B: 0.7 %; mọi súp 10–15 %); câu bị ảnh hưởng mất phiếu → bỏ | H (ngân sách decoding) + M (định dạng sau fine-tune) | 3, 7 |
| F2 | **Gợi ý kiểu đáp án sai ở shop điện thoại**: "máy" (điện thoại) và "mấy" giống nhau khi bỏ dấu → đếm 2 lần hỏi → "list" | 27 câu public gốc gợi ý "list" thay vì "number"; postcheck gắn cờ mọi đáp án số → nửa phiếu → 9 câu bỏ dù đa số mẫu đúng | H (luật gợi ý) | 1, 7 |
| F3 | **Định nghĩa rulebook không được khớp**: so khớp từ bỏ dấu đòi đủ mọi từ ("Mặt hàng được ưa chuộng nhất" cần "mặt"); placeholder `<đơn vị>` không bao giờ khớp; dương tính giả ("mấy ... đồng" → "Dòng máy", 18 câu) | 0 / 54 câu "bao nhiêu <đơn vị>" có định nghĩa; "ưa chuộng" 0 / 1 | H | 4 |
| F4 | **Làm sạch câu hỏi**: tên khách trong dữ liệu kéo "quan" → "quân"; "cháy hàng", "nhiu ... nhứt", "bao nhiêu may", "th nay"; câu gõ nửa có dấu không được phục hồi dấu | 4 câu sai trực tiếp; nhiều câu khác mất định nghĩa rulebook | H | 6, 4 |
| F5 | **Cảnh báo kỳ của precheck bị bỏ qua**: mẫu có kỳ khác kỳ đọc trong câu chỉ đúng 2 / 23 (mẫu không cảnh báo: 0.925) | 2 lần chạy | H (không dùng tín hiệu sẵn có) | 1 |
| F6 | **Gợi ý "dữ liệu không ghi"** đúng 29 / 29 (recall 0.935) nhưng mô hình vẫn trả lời ở 4 câu | `hint_audit` | M (bỏ qua gợi ý) + H (không kiểm) | 1 |
| F7 | **Định dạng**: đáp án một phần tử `[x]` cho câu hỏi số, ngày tháng trả kiểu string | 4 câu | H | 5 |
| F8 | Mẫu số `/ n` của độ tin cậy: tính trên mẫu hợp lệ thay vì mọi mẫu tăng public nhưng **giảm pub_ho 1–2 điểm** → giữ nguyên | `replay_vote` | — | 7 |

Mức mô hình còn lại sau khi sửa hệ thống: các câu nhóm 1 (bữa = ngày, lượt xe, T1) và nhóm 3 khó (đếm mặt hàng có mặt trong
≥ N đơn, đơn bán chịu theo hạn trả) — cần ví dụ / tool / dữ liệu huấn luyện tốt hơn (kế hoạch 2).

## 4. Train (DN không giữ ngoài)

Súp được huấn luyện trên chương trình đã kiểm của các DN train (D1), nên lỗi trên train dev là lỗi "trong mẫu". Phân tích
WP9 (EXPERIMENT_LOG entry 39, 44) trên train: các nhóm chính là định dạng / kiểu đáp án và gợi ý đơn vị (đã sửa ở WP9). Lần
chạy vòng 1 của kế hoạch 2 có thêm tập `tr_dev500` (500 câu train dev, `tests/splits/tr_dev500.txt`) để cập nhật bảng này
với cấu hình mới (mục 6).

## 5. Đã kiểm trên CPU cho vòng 1 (trước GPU)

| Thay đổi | Đo CPU |
|---|---|
| F2: đếm "mấy" trên câu có dấu, bỏ "máy" cạnh từ thiết bị, "cả X lẫn Y / gộp / tính cả" = một tổng, "Mấy lô ... bao nhiêu" = một câu hỏi | kiểu gợi ý đúng 0.967 → **0.999** (public gốc, 971 câu có gợi ý) |
| F3: so khớp có dấu, danh từ đầu ngữ ("Mặt hàng ...") không bắt buộc, placeholder `<đơn vị>` khớp đơn vị đã liên kết, từ đồng nghĩa chat (bữa = ngày, củ = triệu, cháy hàng = hết hàng) | +54 câu có định nghĩa "<đơn vị>", +1 "ưa chuộng", −18 "Dòng máy" giả, −21 "(ế)" giả |
| F4: cap đóng góp unigram của tên shop, "quan" đơn lẻ → "quán", "nhiu ... nhứt" → "nhiều", "bao nhiêu may" → "máy", "th nay" → "tháng này", câu nửa có dấu được phục hồi dấu từng từ (chỉ khi có bigram hỗ trợ) | mô phỏng câu train bỏ dấu 15–50 % từ: phục hồi đúng **95–97 %**, từ đúng bị đổi: 0 |
| F5 + F7 + vote | phát lại 3 lần chạy với gợi ý mới: public gốc **0.939 → 0.950 / 0.939 → 0.948 / 0.930 → 0.939** (chưa tính thay đổi chương trình mô hình viết) |

## 6. Vòng sau

Kết quả GPU của vòng 1 (cấu hình `configs/p2_a/b/c.json`) và các câu còn sai được ghi vào `docs/12_PLAN2.md` mục "Vòng 1";
phân tích lỗi của các vòng tiếp theo bổ sung vào file này (mục 7+).
