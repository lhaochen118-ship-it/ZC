# 16 — Kế hoạch 3: mô hình thuần + lớp luật do mô hình tự học (bản nháp 13/11, chờ user duyệt)

**Mục tiêu (user 13/11):** mô hình thuần (không luật viết tay lúc suy luận) đạt public = 1.0, held-out không giảm. **Ý tưởng của
user:** luật viết sẵn không tổng quát được khi vận hành thật (ở đây Claude phân tích lỗi rồi viết luật; khi chạy thật không có ai viết
luật cho Qwen) → huấn luyện chính Qwen, hoặc thêm một mô hình phụ, có khả năng **tự sinh luật tổng quát**; luật đó chạy như một lớp
kiểm tra, hoặc ghép vào schema / prompt, hoặc là một tool cho mô hình gọi.

## 0. Thực tế phải biết trước

1. **Hạn nộp solution: 16/11 23:59 GMT+7** (Docker đóng băng); private chạy 20–23/11 bằng bản đóng băng. Việc làm sau 16/11 không đổi
   được điểm private. Cửa sổ thật: **13/11 → sáng 16/11 (~3 ngày)**; quota GPU làm mới 14/11 00:00 UTC (14 acc × 30 h ≈ 420 giờ quota
   ≈ 190 giờ tường 4×L4) — **không thiếu GPU**; giới hạn là thời gian tường và số vòng lặp.
2. "Đã có đội đạt 1.0" (`docs/00_TASK.md` §7: 20 đội đạt 1.0 từ 26–28/10, tức 1–2 ngày sau khi mở): vòng public chỉ nộp file jsonl,
   không bắt buộc chạy bằng mô hình ≤ 15B offline → điều này **không** chứng minh mô hình thuần 9B đạt được 1.0. Nó cho biết public đủ
   "sạch" để một hệ mạnh đạt tuyệt đối. P (có luật) đã 0.994, nên trần của dữ liệu + toolkit là ≈ 1.0.
3. **Ước lượng của tôi** (chưa đo): trong cửa sổ đến 16/11, mô hình thuần + lớp luật học được đạt khoảng **0.94–0.97 public**; 1.0 trong
   cửa sổ này **khó (< 15 %)**. Kế hoạch vẫn nhắm 1.0 và được thiết kế để đi tiếp sau 16/11 nếu user muốn (§6 H2).
4. Bản nộp giữ **P + WG1** cho đến khi hệ mới thắng theo quy tắc §5.4 (user chốt).

## 1. Bằng chứng (findings) quyết định thiết kế

| Bằng chứng | Nguồn | Hệ quả cho kế hoạch |
|---|---|---|
| Chưng cất thêm dữ liệu cùng loại đã bão hoà: QS1 0.859 → QS7 0.905 → QS10 0.896 (26M token, RFT) | `docs/14` | không thêm "cùng loại"; phải đổi **loại** dữ liệu và thêm cơ chế luật |
| Lỗi public của mô hình thuần (QS10 × 0.70, 48 sai + 8 bỏ): chương trình khác P ở **kỳ** 24 câu, **thực thể / tham số** 32 câu, **hàm** 17 câu (chồng nhau) | `docs/15` §6, `cmp_pure_p.py` | 3 nhóm luật: kỳ, từ vựng / tham số, định nghĩa nghiệp vụ |
| Kiểu kỳ "T4/2025", "Th12/25": 25.9 % câu public, **0 %** câu train; QS7 PA 0.845 trên nhóm này vs 0.934 phần còn lại | `docs/15` §6 | **đòn bẩy số 1**: tăng cường cách viết kỳ trong dữ liệu |
| Ví dụ lỗi định nghĩa: "Khách sộp nhất" (dòng **mới** của rulebook public: giá trị TB / đơn, ≥ 3 đơn) → mô hình dùng `by="revenue"` thay vì `by="aov"`; "bao nhiêu gói / lon" → thiếu `base_unit` | log QS10 | **biên dịch rulebook** thành thẻ luật lúc suy luận — đúng thứ private cần (ngành mới, dòng rulebook mới) |
| Ví dụ lỗi từ vựng: "xi ô đi" (COD), "bình sữa" (nhóm hàng) → mô hình chọn sai giá trị | log QS10 | từ điển học được + bí danh giá trị sinh từ dữ liệu DN |
| D2N (câu nhiễu) và D2b (kiểu câu mới) nâng held-out; tự gợi ý nâng public +1.8–2.2 | `docs/12`, `docs/14` | giữ cả hai; thay **nguồn** tự gợi ý (đang là `hints.py`) bằng nhãn rút từ chương trình đúng + luật học được |
| Nhãn từ 55 mẫu luật; 250 câu dev không có chương trình; 6 DN chưa dùng | `docs/15` §5 | lấp độ phủ nhãn; huấn luyện lại đủ 30 DN cho bản cuối |
| Hệ có luật thắng nhờ module gợi ý; luật viết tay chỉnh theo public | `docs/15` §4.2 | luật mới phải **học từ train**, kiểm trên DN chưa thấy |

## 2. Thiết kế: lớp luật học được (LRL — Learned Rule Layer)

### 2.1 Nguyên tắc

1. **Mọi luật do mô hình sinh** (Qwen, đúng mô hình nộp), theo một quy trình tự động, chạy lại được không cần Claude / người — cũng là
   quy trình dùng khi vận hành thật với DN / ngành mới.
2. **Chỉ học từ train** (24 DN dev khi phát triển, 30 DN cho bản cuối). Không luật nào được học hay chỉnh từ public / held-out; public
   chỉ đo ở cuối mỗi vòng.
3. **Kiểm bằng dữ liệu, không bằng niềm tin:** một luật chỉ vào thư viện khi tăng đúng / không tăng sai trên câu của **DN khác** với
   DN sinh ra nó (kiểm chéo theo DN).
4. **Tổng quát bắt buộc:** luật phải khớp ≥ 2 DN; không chứa giá trị chỉ có ở một DN (tên hàng, mã…).
5. Giữ ranh giới user 12/11: A (ingest / automap) + B (toolkit `dm`, `industry`) giữ nguyên; C (`hints`, `qnorm`, `qparse` đọc câu)
   và D (`precheck` / cờ theo luật) **bỏ** khỏi suy luận, thay bằng LRL.

### 2.2 Thẻ luật (một định dạng cho mọi loại luật)

```json
{"id": "R0123", "kind": "period | lexicon | definition | unit | value | check",
 "trigger": {"words": ["xi ô đi", "si ô đi"], "industry": null},
 "content": "'xi ô đi' là cách đọc COD (thu tiền khi giao)",
 "code_hint": "payment='cod'",
 "check": null,
 "source": "induced:train | rulebook:<dòng> | data:<cột> | external:vilexnorm",
 "stats": {"shops": 5, "fires": 41, "fixed": 9, "broke": 0}}
```

`check` (nếu có) là một hàm Python nhỏ **do mô hình viết** trên (câu hỏi, chương trình, kết quả) — ví dụ "câu hỏi số lượng theo đơn
vị gói / lon mà chương trình không truyền `base_unit`".

### 2.3 Bốn nguồn sinh luật

| Nguồn | Khi chạy | Cách làm | Tài liệu nền |
|---|---|---|---|
| **P1 Quy nạp từ lỗi train** | offline (huấn luyện) | chạy hệ thuần trên câu train (k mẫu) → câu sai / bỏ + so sánh mẫu đúng ↔ sai cùng câu → Qwen (chế độ think) đề xuất ≤ 3 thẻ / câu → kiểm (§2.6) → thư viện | HtT (Zhu et al.), AutoManual (NeurIPS 2024), ExpeL (AAAI 2024), LEAP (ICML 2024), ReasoningBank (ICLR 2026) |
| **P2 Biên dịch rulebook** | **lúc suy luận**, mỗi split một lần | mỗi dòng rulebook của split → thẻ luật (thuật ngữ, từ kích hoạt, ánh xạ sang hàm / tham số toolkit, đoạn code mẫu); kiểm bằng chạy thử trên dữ liệu DN (không lỗi, không suy biến) — không cần đáp án | Knowledge-to-SQL / DELLM (ACL Findings 2024), SEED (2025) |
| **P3 Bí danh giá trị từ dữ liệu DN** | lúc suy luận, mỗi DN một lần | với mỗi giá trị trong dữ liệu (nhóm hàng, kênh, cách trả, đơn vị…) mô hình sinh cách gọi / viết tắt / đọc phiên âm có thể có ("COD" → "xi ô đi", "thu hộ") | CHESS (value retrieval, 2024) |
| **P4 Dataset ngoài** | offline | ViLexNorm (EACL 2024, ~10.5k cặp câu mạng xã hội Việt ↔ chuẩn) → luật teencode / viết tắt và mô hình nhiễu cho dữ liệu | ViLexNorm |

P2 và P3 là phần trả lời thẳng ý tưởng "khi vận hành thật không có ai viết luật": chúng chạy trên **đầu vào** của split mới (rulebook,
README, dữ liệu), không cần đáp án, nên ngành mới của private vẫn có luật.

### 2.4 Ba cách dùng luật (đúng ba phương án user nêu)

1. **Ghép vào prompt / schema:** thẻ P2 / P3 của DN đặt trong phần chung đầu prompt (dùng chung prefix cache); thẻ P1 / P4 khớp câu
   hỏi (theo từ kích hoạt + độ giống) đặt ở phần riêng của câu, tối đa ~8 thẻ.
2. **Lớp kiểm tra:** sau khi chạy chương trình, các `check` đã được nhận vào thư viện chạy trên (câu, chương trình, kết quả); cờ →
   lượt sửa (như precheck hiện nay, nhưng hàm kiểm do mô hình viết và được nhận bằng thống kê trên train).
3. **Tool:** `rules(text)` trả về thẻ liên quan để mô hình tự tra trong lượt think (bổ sung, đo xem có hơn cách 1 không).

### 2.5 Nội hoá luật vào mô hình (Qwen tự viết luật cho câu của nó)

Đầu ra mới: `<spec>` → `<plan>` → `<code>`. `<spec>` gồm kỳ (biểu thức toolkit), thực thể → giá trị dữ liệu, đơn vị, đại lượng, kiểu
đáp án, id các thẻ luật đã dùng. **Nhãn `<spec>` rút tự động từ chương trình đúng** (duyệt AST: `month(2026, 1)`, `base_unit="gói"`,
`payment="cod"`…) — **không** từ `hints.py`. Đây là bản tổng quát của "tự gợi ý" (đã chứng minh +1.8–2.2 public), nhưng thầy là
chương trình đúng chứ không phải luật của Claude. Lúc suy luận, bỏ phiếu cả trên `<spec>` (các mẫu đồng ý kỳ / thực thể).

### 2.6 Vòng học luật (offline, P1) và luật nhận thẻ

```
24 DN dev chia 4 phần theo DN
vòng r: chạy hệ (thư viện L_r) trên 3 phần → câu sai / bỏ + cặp mẫu đúng-sai
      → Qwen đề xuất thẻ → gộp / bỏ trùng (curator, cập nhật từng thẻ, không viết lại cả thư viện — tránh "context collapse" của ACE)
      → kiểm trên phần thứ 4: chạy A (không thẻ) / B (có thẻ) trên các câu thẻ kích hoạt, cùng seed
      → nhận nếu đúng tăng ≥ 2, sai không tăng, khớp ≥ 2 DN, không giá trị riêng một DN
      → thẻ check: quét mọi mẫu đã log trên dev; nhận nếu bắt ≥ 5 mẫu sai, độ chính xác ≥ 0.9, chạm ≤ 1 % mẫu đúng
      → L_{r+1} (tối đa ~200 thẻ); xoay phần kiểm, 3 vòng
```

Quy tắc nhận thẻ check chính là cách tôi đã làm tay ở vòng 17 (quét "tín hiệu sạch" cho `count_check`, `size_use_check`) — nay do
Qwen đề xuất và máy thống kê, nên tái lập được không cần Claude.

### 2.7 Một mô hình hay thêm mô hình phụ?

Ngân sách: 9.65B + tối đa ~5B. Ba lựa chọn:

| Lựa chọn | Ưu | Nhược |
|---|---|---|
| **A. Một mô hình đa nhiệm** (cùng Qwen viết luật, `<spec>`, code; khác prompt) | dùng chung KV / prefix cache, không thêm tham số, đã có bằng chứng (tự gợi ý) | huấn luyện đa nhiệm có thể làm giảm code |
| B. Cùng gốc + LoRA thứ hai cho vai viết luật (vLLM multi-LoRA) | tách vai, không đụng LoRA code | +80M tham số, phải đổi đường gộp LoRA trong Docker |
| C. Mô hình nhỏ chuẩn hoá câu (~1.7B) học ViLexNorm + nhiễu tổng hợp | thay `qnorm` bằng bản học được | thêm một bước suy luận, thêm rủi ro container |

**Đề xuất:** A trước; chỉ sang B nếu đo thấy đa nhiệm làm giảm code; C để sau 16/11.

## 3. Dữ liệu v3 (đòn bẩy lớn nhất, CPU)

| Mã | Dữ liệu | Cách làm | Nhắm lỗi |
|---|---|---|---|
| D1+ | lấp độ phủ nhãn | gộp mọi chương trình đúng trong log cũ (`tests/pool_gens.py`) + lấy mẫu mới k = 16 trên 250 câu chưa có (+ câu của 6 DN cho bản cuối) | 250 câu chưa từng có lời giải |
| DS | `<spec>` cho mọi bản ghi | `training/data/spec_from_program.py` (AST) | kỳ / thực thể / đơn vị |
| DP | **cách viết kỳ** | biết kỳ đúng từ chương trình → viết lại phần kỳ trong câu theo nhiều cách (T4/2025, Th4/25, 4-2025, 04/25, tháng tư năm ngoái, quý IV…); thêm bản do Qwen viết lại "kiểu nhắn tin", chỉ giữ bản mô hình đọc ra đúng kỳ | 25.9 % câu public dùng kiểu chưa từng thấy |
| DL | teencode / viết tắt | mô hình nhiễu học từ ViLexNorm (xác suất từ chuẩn → từ nhiễu) áp lên câu train | câu bẩn |
| DT | phiên âm từ mượn | Qwen sinh cách đọc phiên âm của mọi từ tiếng Anh / mượn có trong toolkit, rulebook, giá trị dữ liệu (order, COD, ship, voucher, size, combo, topping…) → thay vào câu | "o đờ", "xi ô đi" |
| DR | **thuật ngữ rulebook mới** | đổi tên / viết lại định nghĩa các dòng rulebook train thành thuật ngữ mới (vd "khách VIP nhất = …") + câu dùng thuật ngữ đó; nhãn = chương trình theo định nghĩa (máy D3) | đọc rulebook để xử lý ngành / thuật ngữ mới của private |
| DC | thẻ luật trong prompt | bản ghi có thẻ P2 / P1 thật + thẻ gây nhiễu (không liên quan) | dùng đúng thẻ, bỏ qua thẻ sai |
| D3+ | câu tổng hợp | tăng D3 (chương trình đã kiểm đặt lại trên DN / kỳ / thực thể khác; thầy viết câu nhiều phong cách) 3–5× | đa dạng |

Phát triển trên 24 DN; **bản cuối huấn luyện lại trên 30 DN** cùng công thức. Các bộ sinh DP / DL / DT chỉ chạy lúc tạo dữ liệu, không
chạy lúc suy luận — mô hình phải tự tổng quát (đúng ranh giới user 08/11 / 12/11).

## 4. Luồng suy luận Q2 (thuần + LRL)

1. A: ingest / automap (giữ nguyên).
2. Mỗi split: P2 biên dịch rulebook → thẻ; mỗi DN: P3 bí danh giá trị → thẻ (cache, nằm trong prefix chung).
3. Mỗi câu: truy hồi ví dụ (theo chữ câu) + thẻ P1 / P4 khớp câu.
4. Qwen viết `<spec>` + `<plan>` + `<code>`, 2 → 8 mẫu như cấu hình Q.
5. Chạy; phản hồi của toolkit; thẻ `check` đã nhận → lượt sửa.
6. Bỏ phiếu theo kết quả (+ đồng thuận `<spec>`), bỏ trả lời < 0.5.

Đo thêm: **P + LRL** (P giữ luật cũ, thêm thẻ P2 biên dịch rulebook) — đường an toàn nếu hệ thuần chưa thắng nhưng thẻ rulebook giúp
ngành mới.

## 5. Đo lường

### 5.1 Sửa đo trước (CPU, 13/11)

- `eval_flow.py`: mỗi DN dùng rulebook của split gốc (sửa lỗi `docs/15` §4.3).
- `pub_ho_clean`: bộ kiểm tương đương tự động (Qwen, 2 prompt độc lập: kỳ / đại lượng / thực thể có giữ nguyên không) lọc `pub_ho`;
  không ai xem từng câu.
- `tr_newterm`: câu dùng thuật ngữ rulebook mới tổng hợp trên 6 DN held-out (đo khả năng đọc rulebook — mô phỏng ngành mới).

### 5.2 Bộ đo và vai trò

| Bộ | Vai trò |
|---|---|
| `pub_full` | đích 1.0; chỉ đo cuối vòng, không học gì từ nó |
| `tr_ho` | held-out sạch cho hệ học luật (thư viện không bao giờ thấy 6 DN này) |
| `pub_ho_clean`, `pub_noind`, `tr_newterm` | tổng quát: câu kiểu public, ngành mới, thuật ngữ mới |

### 5.3 Giữ / bỏ một thay đổi

Như `docs/12_PLAN2.md` §4: giữ khi public tăng ≥ 0.3 và held-out gộp không giảm > 0.3; báo cả hai ở mọi mốc.

### 5.4 Đổi bản nộp từ P sang hệ mới (**cần user chốt**)

Đề xuất: đổi khi hệ mới (thuần + LRL, hoặc P + LRL) có held-out gộp (`tr_ho` + `pub_ho_clean` + `pub_noind` + `tr_newterm`) **≥ P** và
public **≥ 0.985**. Lý do chấp nhận public thấp hơn P một chút: số đo của P lạc quan (`docs/15` §4.2), còn private có ngành mới — nơi
luật học được + biên dịch rulebook có lợi thế.

## 6. Gói việc và lịch

Giờ UTC (GMT+7 = UTC + 7). "acc" = acc Kaggle, mỗi nhánh huấn luyện một acc, FSDP 4 GPU, chuỗi checkpoint khi một acc không đủ giờ.

### H1 — đến hạn nộp (13/11 → 16/11)

| Gói | Việc | Khi | GPU | Nghiệm thu / cổng |
|---|---|---|---|---|
| WP3-00 | sửa đo (§5.1), `spec_from_program.py`, bộ sinh DP / DL / DT / DR / DC, D1+ gộp log, test CPU | 13/11 | — | `tests/run_all.sh` PASS; thống kê dữ liệu (tỉ lệ kiểu kỳ ≈ public) |
| WP3-01 | khung LRL: thẻ luật, truy hồi thẻ, chạy `check` trong sandbox, prompt P1 / P2 / P3, curator, kiểm A/B, quét check; cấu hình `flow_lrl.json` | 13/11 | — | test CPU với mô hình giả |
| WP3-02 | P2 biên dịch rulebook train + public; vòng P1 số 0 từ log có sẵn (`rq5_tr_dev_all`, QS*) | 13/11 chiều | quota còn lại (~45 h) | thẻ chạy được trên mọi DN |
| WP3-03 | **SFT v3** (thuần): V3a = D1+ / DS / DP / DL / DT + D2 / D2N; V3b = V3a + DR + DC; 26M token, 1 acc / nhánh | 13/11 tối (chuỗi 2 acc) hoặc 14/11 00:00 | ~3.5 h tường / nhánh | — |
| WP3-04 | **P refit 30 DN** (công thức WG1 + 6 DN) + `bank_holdout` | 14/11 00:00 | 1 acc | public ≥ 0.993, `pub_ho` không giảm → ứng viên bản nộp |
| WP3-05 | đánh giá V3a / V3b × {Q, Q2 = Q + LRL} trên 4 bộ (chia shard) | 14/11 ~04:00 | 6–8 phiên | §5.3 |
| WP3-06 | vòng P1 số 1–2 với mô hình V3 tốt nhất (4 phần DN, A/B, quét check) | 14/11 ~06:00 → 14:00 | 3–4 phiên / vòng | thư viện L2 |
| WP3-07 | expert iteration: V3 tốt nhất + L2 trả lời câu train / paraphrase / D3+ → giữ chương trình đúng → **SFT v4** (2 seed) | 14/11 ~14:00 → 15/11 00:00 | 2 acc | — |
| WP3-08 | đánh giá v4 + L2 (2 seed) và P + LRL; so theo §5.4 | 15/11 00:00 → 06:00 | 6–8 phiên | quyết định |
| WP3-09 | bản thắng: refit 30 DN (nếu là hệ mới), đường container, `docker/README_SOLUTION.md`, cập nhật `final_submission` | 15/11 06:00 → 12:00 | 1–2 acc | container khớp ±0.03 |
| WP3-10 | user build Docker | 15/11 tối (GMT+7) | — | dự phòng 16/11 |

Ước lượng GPU H1: ~50 giờ tường ≈ 110 giờ quota (trong 420 giờ quota của 14/11) — không cần tiết kiệm; giới hạn là số vòng.

### H2 — sau 16/11 (chỉ khi user muốn, không đổi được điểm private)

- RL GRPO với **thưởng = PA của từng câu** (+1 đúng, 0 bỏ, −1 sai) — học cả việc bỏ trả lời; thưởng đơn giản như Arctic-Text2SQL-R1
  (thực thi + cú pháp). Chi phí: prompt ~6k token → ~4 phút / bước trên 4×L4 (16 câu × 4 mẫu) → cần prompt ngắn hơn hoặc GPU lớn
  hơn; công cụ: TRL `GRPOTrainer` hoặc verl.
- Mô hình chuẩn hoá câu nhỏ (lựa chọn C §2.7); bộ nhớ lúc suy luận kiểu Dynamic Cheatsheet / ReasoningBank (phải giữ tất định).
- Tối ưu prompt / thư viện luật bằng GEPA (ICLR 2026: hơn GRPO trung bình 6 điểm với ít rollout hơn tới 35×).

## 7. Rủi ro và dự phòng

| Rủi ro | Dự phòng |
|---|---|
| Thẻ luật sai làm hỏng câu đang đúng | nhận thẻ bằng A/B trên DN khác; thẻ gây nhiễu trong huấn luyện; giới hạn số thẻ / câu |
| Luật học thuộc DN train | kiểm chéo theo DN; cấm giá trị riêng một DN; đo `tr_ho` |
| Biên dịch rulebook sai ở ngành mới | kiểm chạy thử trên dữ liệu; bỏ thẻ suy biến; luôn đưa cả dòng rulebook gốc vào prompt |
| Đa nhiệm làm giảm code | lựa chọn B (LoRA thứ hai) |
| Không kịp 16/11 | bản nộp vẫn P + WG1 (hoặc P refit 30 DN nếu WP3-04 đạt); mọi gói đều có điểm dừng an toàn |
| Tất định (±0.03) với thẻ sinh lúc suy luận | P2 / P3 dùng temp 0, seed cố định, cache theo hash rulebook / dữ liệu |
| Giấy phép ViLexNorm CC BY-NC-SA 4.0 (phi thương mại) | quy định user 08/11 cho phép dataset nghiên cứu / giáo dục; ghi rõ trong README solution |

## 8. Tài liệu tham khảo (mới so với `docs/11_LITERATURE.md`)

| Nguồn | Nội dung dùng | Code |
|---|---|---|
| Zhu et al., *Large Language Models can Learn Rules* (HtT), arXiv 2310.07064 (Google DeepMind) | quy nạp thư viện luật từ cặp câu – đáp án train, giữ luật theo độ phủ / độ tin; +10–30 % tuyệt đối | [google-deepmind/llms_can_learn_rules](https://github.com/google-deepmind/llms_can_learn_rules) |
| Chen et al., *AutoManual*, NeurIPS 2024 | Planner / Builder / Consolidator: xây, gộp, giới hạn số luật, biên thành sổ tay | [minghchen/automanual](https://github.com/minghchen/automanual) |
| Zhao et al., *ExpeL*, AAAI 2024 | rút "insight" từ trải nghiệm đúng / sai, dùng lúc suy luận, không cập nhật trọng số | [andrewzh112.github.io/expel](https://andrewzh112.github.io/expel) |
| Zhang et al., *In-Context Principle Learning from Mistakes* (LEAP), ICML 2024 | gây lỗi trên ví dụ → rút nguyên tắc → dùng lúc thi | [arXiv 2402.05403](https://arxiv.org/abs/2402.05403) |
| Ouyang et al., *ReasoningBank*, ICLR 2026 | bộ nhớ chiến lược từ cả thành công lẫn thất bại; so sánh nhiều quỹ đạo để lọc luật giả | [google-research/reasoning-bank](https://github.com/google-research/reasoning-bank) |
| Zhang et al., *Agentic Context Engineering* (ACE), ICLR 2026 | playbook tiến hoá: sinh – phản tư – tuyển chọn, cập nhật từng mục (tránh "context collapse") | [ace-agent/ace](https://github.com/ace-agent/ace) |
| Suzgun et al., *Dynamic Cheatsheet*, EACL 2026 | bộ nhớ lúc suy luận gồm chiến lược / đoạn code tái dùng | [suzgunmirac/dynamic-cheatsheet](https://github.com/suzgunmirac/dynamic-cheatsheet) |
| Agrawal et al., *GEPA*, ICLR 2026 (oral) | tiến hoá prompt bằng phản tư ngôn ngữ, hơn GRPO 6 điểm TB, ít rollout tới 35× | [arXiv 2507.19457](https://arxiv.org/abs/2507.19457) |
| Hong et al., *Knowledge-to-SQL* (DELLM), ACL Findings 2024 | **mô hình phụ** sinh tri thức cho text-to-SQL, tinh chỉnh bằng phản hồi của CSDL | [DEEP-PolyU/Knowledge-to-SQL](https://github.com/DEEP-PolyU/Knowledge-to-SQL) |
| Yun & Lee, *SEED*, 2025 | sinh "evidence" tự động từ schema + mô tả + giá trị mẫu; không evidence người viết vẫn bằng / hơn | [arXiv 2506.07423](https://arxiv.org/abs/2506.07423) |
| Wang, Neubig, Fried, *TroVE*, ICML 2024 | tự xây bộ hàm tái dùng, chọn theo đồng thuận thực thi, tỉa định kỳ (gồm table QA) | [zorazrw/trove](https://github.com/zorazrw/trove) |
| Nguyen et al., *ViLexNorm*, EACL 2024 | 10,467 cặp câu mạng xã hội Việt → chuẩn | [ngxtnhi/ViLexNorm](https://github.com/ngxtnhi/ViLexNorm) |
| Yao et al., *Arctic-Text2SQL-R1*, ACL Findings 2026 | GRPO với thưởng đơn giản (thực thi + cú pháp) ổn định hơn thưởng từng phần | [arXiv 2505.20315](https://arxiv.org/abs/2505.20315) |
| SemEval-2025 Task 8 (DataBench, QA trên bảng) — AILS-NTUA và các đội đầu | LLM sinh code pandas + sửa lỗi là hướng thắng | [arXiv 2506.09657](https://arxiv.org/abs/2506.09657) |

Công cụ: TRL (`GRPOTrainer`, `DPOTrainer`), verl (RL quy mô lớn), DSPy (tối ưu prompt / chương trình LM), vLLM (multi-LoRA).

## 9. Nhật ký triển khai (13/11)

**Đo sạch (WP3-00, `src/para_judge.py`, `namhunhh/zc-pj-1`):** Qwen tự chấm tương đương (2 prompt độc lập, không ai xem từng câu) giữ
1171 / 1373 paraphrase public (`pub_ho_clean`) và 1080 / 1195 paraphrase train (`tr_ho_clean` = 631 câu gốc + 1080); tập con
`pub_ho400c` (348), `tr_ho400c` (375). Mốc trên tập sạch:

| Hệ | pub_ho → pub_ho_clean | pub_ho400 → 400c | tr_ho400 → 400c |
|---|---|---|---|
| P + WG1 (seed 0 / 1) | 0.6286 → **0.7267** / 0.6329 → 0.7284 | 0.6225 → 0.6868 / 0.6125 → 0.681 | 0.8775 → 0.92 / 0.8575 → 0.8987 |
| QS7 × 0.35 (thuần) | — | 0.495 → 0.569 | 0.8175 → 0.864 |
| QS10 × 0.70 (thuần) | — | 0.525 → 0.6063 | 0.8625 → 0.9093 |
| QS6 × 0.70 (thuần) | — | 0.5475 → 0.5805 | 0.8775 → 0.912 |

~10 điểm của `pub_ho` cũ là nhãn nhiễu; khoảng cách thật thuần ↔ P ở câu kiểu public ≈ 8 điểm, ở câu train ≈ 1 điểm.
