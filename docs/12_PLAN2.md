# 12 — Kế hoạch 2 (Bước 4 của `docs/13_TASK_AFTER_PLAN1.md`)

*09/11/2026. Dựa trên `docs/10_ERROR_ANALYSIS.md` (nguyên nhân) và `docs/11_LITERATURE.md` (kỹ thuật). Mục tiêu dừng: public gốc
1000 / 1000 bằng cải tiến tổng quát, held-out (pub_ho, tr_ho; paraphrase train) không giảm. Ràng buộc giữ nguyên: ≤ 15B tham số,
offline, ≤ 6 giờ trên 2×RTX 5090, mô hình viết mọi chương trình trả lời, không vá theo câu / DN, không train trên public /
private. Hạn: user build Docker 15/11, nộp trước 16/11 23:59 GMT+7 → mọi thứ vào bản nộp phải xong **14/11**.*

Mốc xuất phát: súp WF1–4 α 0.35 + prompt full + think vòng 2 = public gốc **0.939** (956 / 17 / 27), pub_ho 0.592, tr_ho ~0.87.
Thành thật: 100 % public với mô hình 9B là đích rất xa (BIRD: hệ tốt nhất 82 %, người 93 %); kế hoạch ưu tiên phần tăng chắc,
rẻ trước, và lặp vòng đến hết thời gian.

## 1. Ma trận nhóm lỗi → giải pháp

| Nhóm (số câu / 44) | Giải pháp | Tăng kỳ vọng public | GPU / giờ | Rủi ro |
|---|---|---|---|---|
| 7 Chọn (6) + 5 Định dạng (4) + phần 1 do gợi ý sai | **P2-01** sửa gợi ý kiểu (máy/mấy, tổng nhiều nhóm, ngày), **P2-03** chuẩn hoá định dạng theo kiểu gợi ý, cờ "kỳ lệch" | +1.0–1.5 điểm (phát lại: +0.9–1.1) | 0 (CPU) + 1 lần đo | thấp; gợi ý sai ở câu lạ → kiểm bằng held-out |
| 4 Nghiệp vụ (7) + 6 Câu bẩn (4) | **P2-01** khớp định nghĩa rulebook có dấu + `<đơn vị>`, làm sạch câu (quán/quân, cháy hàng, nửa có dấu) | +0.5–1.0 | 0 + đo | prompt đổi so với lúc huấn luyện súp → có thể lệch; đo |
| 3 Bị cắt / không chạy (12) | **P2-02** trần 1536 token cho mẫu thường (10.7 % mẫu bị cắt ở 512); phương án 2: ép bắt đầu bằng `<plan>`; phương án 3: SFT lại định dạng gọn | +0.5–1.2 | +10–20 % thời gian sinh | chậm hơn; mẫu dài có thể kém tin cậy (think rescue đã âm) |
| 1 Ý định — dữ liệu không ghi (4) | **P2-04** lượt xem lại khi gợi ý nói dữ liệu không ghi (`--review 2`) | +0.3–0.4 | ~0 | mô hình đổi đáp án đúng (precision gợi ý 29 / 29 → thấp) |
| 1 Ý định — phép đo / đơn vị đếm (bữa = ngày, lượt xe, T1) | **P2-05** gợi ý "đối tượng được đếm / phép đo" + mô tả mặc định của hàm (`by=`) + ví dụ chọn theo chương trình nháp (DAIL) | +0.3–0.6 | ~0 | thêm token prompt |
| 3 + 4 câu khó (≥ N đơn, đơn bán chịu theo hạn trả, vốn lô hết hạn) | **P2-06** hàm toolkit tổng quát cho mẫu câu hay gặp còn thiếu + ví dụ đã kiểm | +0.3–0.5 | 0 | phải tổng quát (không theo câu test): lấy mẫu câu từ train + rulebook |
| 7 Chọn khi phiếu tách | **P2-07** bộ chọn: verifier học được (LEVER) / so cặp (CHASE) trên log mẫu train + paraphrase | +0.3–0.8 | 2–4 GPU-giờ (sinh dữ liệu + train 1 mô hình nhỏ hoặc LoRA 9B) | quá khớp; held-out quyết định |
| Mọi nhóm (năng lực mô hình) | **P2-08** fine-tune vòng 2 theo prompt mới (gợi ý mới, định dạng gọn), lưới song song; **P2-09** DPO / GRPO theo thưởng thực thi (thưởng từng phần) | +1–3 (không chắc) | 8–30 GPU-giờ | lệch held-out (bài học WP6: fine-tune đơn < gốc trên held-out) → chỉ giữ qua súp + WiSE |

## 2. Gói việc (theo thứ tự; rẻ trước, GPU dài chạy nền song song với CPU)

| Gói | Nội dung | Đặc tả | Đo | Trạng thái |
|---|---|---|---|---|
| **WP-P2-01** | Gợi ý + làm sạch câu | `hints.answer_type(t, clean)` + `n_asks`; `rule_terms_of(q, rows, entities)` (có dấu, đầu ngữ tuỳ chọn, `<đơn vị>`, đồng nghĩa); `qnorm`: `add_names` (cap unigram tên), quy tắc quán / máy / nhiều / tháng này, câu nửa có dấu (`MIXED_MAX` 0.9, cần bigram hỗ trợ); test `tests/test_p2_fixes.py` | `tests/hint_audit.py` (CPU), GPU `p2_a` | CPU xong |
| **WP-P2-02** | Ngân sách decoding | `--max-tokens 1536` (`configs/p2_b.json`) | GPU `p2_b` vs `p2_a` | chờ GPU |
| **WP-P2-03** | Bỏ phiếu / định dạng | `--fmt-fix 1` (`hybrid.fmt_fix`), `--period-flag 1` (cảnh báo kỳ = cờ: nửa phiếu, không dừng sớm); θ hiệu chỉnh lại trên dev sau mỗi vòng (`tests/replay_vote.py`, `tests/calibrate.py`) | phát lại + GPU | CPU xong |
| **WP-P2-04** | Xem lại khi thiếu dữ liệu | `--review 2` (`configs/p2_c.json`) | GPU `p2_c` vs `p2_b` | chờ GPU |
| **WP-P2-05** | Gợi ý đối tượng đếm / phép đo | `hints.counts` (danh từ sau "mấy / bao nhiêu", đồng nghĩa bữa → ngày), mặc định `by=` trong help của hàm, truy hồi ví dụ theo chương trình nháp | hint_audit + GPU | sau vòng 1 |
| **WP-P2-06** | Hàm toolkit cho mẫu câu thiếu | từ lỗi train + public: đếm mặt hàng có mặt trong ≥ N đơn, đơn bán chịu theo tháng hạn trả, giá trị vốn lô hết hạn trong khoảng; mỗi hàm có test và mô tả trong `api_spec` | test CPU + GPU | sau vòng 1 |
| **WP-P2-07** | Bộ chọn | dữ liệu: log mẫu (câu, chương trình, kết quả, đúng / sai) của train dev + paraphrase train (không public); mô hình: logistic trên đặc trưng (đã có `confidence.py`) → LoRA 0.8B / 9B "đúng / sai" nếu đủ dữ liệu | replay trên log + GPU | vòng 2 |
| **WP-P2-08** | Fine-tune vòng 2 | prompt mới (sau P2-01/05), D1 dựng lại + D2 + D2N + D3 thầy (qwen27, gemma31) lọc thực thi; nhãn **không lập luận ngoài `<plan>`** | lưới mục 3 | vòng 2–3 |
| **WP-P2-09** | DPO / GRPO | cặp đúng / sai của chính súp trên câu train + paraphrase; thưởng: đúng kết quả + kỳ khớp gợi ý + thực thể khớp + định dạng | quick400 + held-out | tuỳ thời gian |
| **WP-P2-10** | Chốt + nộp | cấu hình tốt nhất → `configs/flow_final.json`, adapter mới (nếu có) → dataset + `src/adapter_hashes.json`, θ, README, STATUS | public × 2 + held-out + đường container | 14/11 |

## 3. Thí nghiệm song song (khi hệ thống đã cao)

Lưới vòng 2–3 (mỗi biến thể một GPU L4 trên acc 4×L4: 4 biến thể / phiên, 2 phiên / acc):

| Trục | Giá trị |
|---|---|
| Học sinh | Qwen3.5-9B (chính); Qwen3.5-9B base-instruct khác seed |
| Thầy (D3) | self, qwen27 (Qwen3.6-27B-FP8), gemma31 (Gemma 4 31B) — chỉ giữ chương trình đúng khi chạy trên train |
| Dữ liệu | D1v3 (prompt mới); D1v3 + D2; D1v3 + D3qwen27; D1v3 + D2N |
| Ghép | súp 4 thành viên α ∈ {0.35, 0.5} |

Quy trình: nhánh train 1 epoch (Unsloth, ~1.5 h / nhánh / L4) → đo nhanh `quick400` + `pub_ho` trên chính phiên đó →
**loại sớm** nhánh có (quick400 < gốc − 2) hoặc (pub_ho < gốc − 1.5); nhánh còn lại vào súp → đo đủ `pub_full` × 2 +
`pub_ho` + `tr_ho`. Mọi biến thể đánh giá trên **cùng** bộ held-out.

## 4. Quy tắc giữ / bỏ

- Giữ một thay đổi khi: public gốc tăng (≥ +0.3 điểm một lần chạy, hoặc trung bình 2 lần) **và** held-out gộp
  (pub_ho × 1373 + tr_ho × 1826) / 3199 không giảm quá 0.3 điểm (nhiễu: pub_ho ±1, tr_ho ±0.7).
- Thay đổi tăng public nhưng giảm held-out quá ngưỡng → **loại** (chống quá khớp).
- Thay đổi chỉ là sửa lỗi hiển nhiên (bug gợi ý / định dạng) đo trên CPU: giữ khi CPU không xấu đi và GPU không giảm.
- θ chỉ chọn trên dev (pub_full, pub_k0, train dev), kiểm trên held-out.
- Báo cả hai số (public, held-out) ở mọi mốc trong `handoff/STATUS.md` ("Tiến độ P2").

## 5. Lộ trình

| Ngày | Việc |
|---|---|
| 09/11 tối | Vòng 1 GPU: `p2_a`, `p2_b`, `p2_c` song song (3 phiên); phân tích tr_dev500 |
| 10/11 | Vòng 2: P2-05, P2-06 (CPU) → GPU; dữ liệu bộ chọn P2-07; dựng D1v3 / D3 với prompt mới |
| 10–12/11 | Lưới fine-tune P2-08 (nền, nhiều acc); P2-07 bộ chọn |
| 12–13/11 | P2-09 nếu còn thời gian; vòng phân tích lỗi tiếp theo |
| 14/11 | Chốt cấu hình, đo đường container, cập nhật Docker assets + README + STATUS |
| 15/11 | User build Docker, chạy `run_public_test.sh`, nộp |

## Vòng 1 (09/11) — kết quả

Code `1d759c8`. Cùng súp WF1–4 α 0.35, mỗi cấu hình một phiên 4×L4 (merge trong kernel + eval 4,199 câu ≈ 1.5 h).

| Cấu hình | Public gốc PA (đúng / sai / bỏ) | pub_ho | tr_ho | Held-out gộp | Quyết định |
|---|---|---|---|---|---|
| Mốc (flow_final, 5 lần) | 0.934 tb (0.930–0.939) | 0.592 | 0.873 | 0.752 | — |
| `p2_a` = WP-P2-01 + 03 (gợi ý, làm sạch, định dạng, cờ kỳ), trần 512 | 0.965 (976 / 11 / 13) | 0.603 | 0.871 | 0.756 | giữ |
| `p2_b` = a + trần 1536 (P2-02) | 0.967 (979 / 12 / 9) | 0.603 | 0.866 | 0.753 | giữ |
| `p2_c` = b + review thiếu dữ liệu (P2-04) | **0.972** (981 / 9 / 10) | 0.601 | 0.874 | **0.757** | **giữ (tốt nhất vòng 1)** |

Train dev (`tr_dev500`, p2_a): 0.992 (498 / 2 / 0) — "lời gộp" đọc thành lãi ròng; tăng trưởng cùng kỳ năm trước tính so với tháng trước.

## Vòng 2 (09/11) — kết quả

Code `c49da84` (+ gợi ý `counted` / `thresholds` khi `--hint-counts 1`, prefill `<plan>`, đồng nghĩa % / chênh).

| Cấu hình | Public gốc | pub_ho | tr_ho | Gộp | Quyết định |
|---|---|---|---|---|---|
| `p2_d` = c + gợi ý đếm | **0.975** (985 / 10 / 5) | 0.600 | 0.868 | 0.753 | chưa quyết (public +0.3, held-out −0.4: trong nhiễu) → lặp seed vòng 3 |
| `p2_e` = d với prefill `<plan>`, trần 512 | 0.898 (939 / 41 / 20) | 0.575 | 0.874 | 0.746 | **loại**: lập luận tự do trước `<plan>` của súp có ích |
| `p2_f` = b + gợi ý đếm | 0.967 (980 / 13 / 7) | 0.605 | 0.869 | 0.756 | gợi ý đếm ≈ trung tính |

d sửa 7 câu của c (2 câu "mấy bữa", cùng kỳ năm trước, "cháy hàng"…), thêm 3 (size sữa "số 4", đơn vị "gói", lô).

## Vòng 3 (09/11, đang chạy)

Code `3131884` / `c918ff2`: + thực thể **việc sửa xe** (qua bảng việc của toolkit, chỉ cụm sau động từ) và **size** danh mục ("số 4", "size M"),
đồng nghĩa lời = lãi, precheck `named_periods` (cờ `--named-periods-flag`: mẫu không dùng đủ các kỳ câu nêu đúng 85 / 106).
Lặp seed để đo nhiễu: `p2_g` (= d + named-periods), `p2_g_s1`, `p2_d` (code mới), `p2_c_s1`.

## Vòng 4 (code sẵn, chờ vòng 3)

Code `6bf3b19`: toolkit `lots_expiring` / `items_expiring` nhận **N ngày** ("hết hạn trong N ngày tới" — 3 câu sai ở mọi lần chạy vì
`TODAY + 120` lỗi kiểu / tự tính ngày); precheck `threshold` + `--threshold-flag` (ngưỡng câu đặt không có trong chương trình: mẫu đúng
13 / 19 vs 0.96; phát lại +0.001 public, +0.0015 pub_ho). Câu ổn định còn sai (sau d): đếm mặt hàng có mặt trong ≥ N đơn (quên lọc ngưỡng),
lô hết hạn trong N ngày, đơn vị đếm ("bao nhiêu ly / gói" chọn `size_qty`), tỉ trọng máy theo số lượng, size bỉm, nhập theo viên.
