# 17 — Kế hoạch 4: hệ S–W–V (mô hình giải + mô hình viết luật + bộ chọn học được), không luật viết tay

*Lập 13/11/2026 16:30 UTC. Quyết định user 13/11 tối: (1) **ngừng phát triển luật viết tay** — luật nếu có phải do mô hình viết;
(2) **bản nộp cuối là hệ mới** (không luật viết tay) kể cả khi điểm thấp hơn P; (3) WR30 (P huấn luyện lại 30 DN) chạy xong làm dự phòng.
Thay phần còn lại của kế hoạch 3 (`docs/16`). Hạn: container sẵn cho user build **15/11 12:00 UTC**; nộp trước **16/11 16:59 UTC**.*

## 0. Bằng chứng (đo 13/11, `Process/scores_p3.csv`, `docs/16` §9)

| Hệ | public | pub_ho400c | tr_ho400c | pub_noind* | tr_newterm | held-out gộp |
|---|---|---|---|---|---|---|
| P (luật viết tay, bản cũ) | 0.994 | 0.698 | 0.923 | 0.973 | 0.907 | 0.840 |
| P + thẻ mô hình viết | — | 0.678 | 0.909 | 0.982 | 0.889 | 0.827 |
| Q v3a × 0.70 (thuần) | 0.909 | 0.589 | 0.917 | 0.821 | 0.870 | 0.774 |
| Q2 v3a × 0.70 (thuần + thẻ) | 0.928 | 0.609 | 0.904 | 0.830 | 0.852 | 0.776 |

\* `pub_noind` **chưa từng ẩn hàm ngành** (không job nào bật `--hide-api industry`): mọi số "ngành mới" đến nay đều có toolkit ngành
viết tay giúp. Private có ngành mới **không** có toolkit đó → phép đo đúng là `pub_noind` + `--hide-api industry` (gọi `pub_noindH`).

Phát hiện quyết định thiết kế:
1. **Thẻ văn xuôi** (thẻ biên dịch rulebook / quy nạp từ lỗi train) chỉ +1.9 public cho hệ thuần, ~0 held-out, và làm P giảm → dạng
   văn xuôi lặp lại rulebook đã có trong prompt; thẻ quy nạp từ lỗi train là kiến thức lẽ ra nằm trong trọng số (A/B vòng 0 âm).
2. **Bộ chọn:** câu bỏ trả lời → theo đa số: public −2, held-out +3 (≈ 0). Trần "có mẫu đúng": +1.8 public, +2.0 held-out → chỉ bộ
   chọn **học được** lấy được.
3. Lỗi hệ thuần (docs/15 §6): kỳ 24 câu, thực thể / tham số 32, hàm 17 (chồng nhau); kiểu kỳ "T4/2025" 26 % câu public, 0 % train.
4. Huấn luyện FSDP 4×L4 đo được **1.75k tok/s** (v3b: 19.5M token, 87 bước, 3 h 06); chồng adapter (`--base-adapters`) cho phép học
   tiếp trên v3a × 0.70 với ít token.

## 1. Kiến trúc S–W–V (một mô hình gốc Qwen3.5-9B; tổng tham số lúc chạy ≈ 9.7B)

| Vai | Làm gì | Học từ đâu | Chạy khi nào | Đo |
|---|---|---|---|---|
| **S** mô hình giải | câu + ngữ cảnh → `<spec>` + chương trình trên toolkit / C / T | trọng số (LoRA): kỹ năng ổn định (văn phong, kỳ, nhiễu, API, không trả lời được) + **đọc và dùng đơn vị luật trong prompt** | mỗi câu | S một mình trên mọi bộ |
| **W** mô hình viết luật | dòng rulebook / README / dữ liệu của split mới → **đơn vị luật thực thi được**: `{name, triggers, definition, signature, code (hàm Python trên C dùng primitive toolkit), tests}`; chỉ giữ đơn vị chạy được trên dữ liệu DN, không suy biến | trọng số của S (bản gộp trong image: hai mô hình 9.65B vượt 15B; `"units_model": "base"` = Qwen gốc chỉ để so sánh, G0g) + ví dụ mẫu là cặp (dòng rulebook train/public ↔ hàm toolkit đã cài: `dm.py`, `industry.py`) | mỗi split một lần, trước khi trả lời (chỉ đọc đầu vào, không đáp án) | `pub_noindH`, `tr_newterm`: S+W so S và S + hàm thật (trần) |
| **V** bộ chọn học được | mỗi nhóm đáp án ứng viên của một câu → xác suất đúng → chọn nhóm + quyết định bỏ trả lời | mô hình nhỏ (hồi quy logistic / cây tăng cường) trên đặc trưng chung (tỉ lệ phiếu, đồng thuận `<spec>`, cờ, số lượt sửa, lỗi chạy, biến thể prompt, độ dài) — **học từ log của S trên DN train**, không luật viết tay | sau khi các mẫu của câu xong | AUROC; PA held-out so bỏ phiếu thường |

**Còn giữ (hạ tầng, user cho phép 08/11, không phát triển thêm):** ingest / automap (đọc dữ liệu, schema), toolkit `dm` (định nghĩa chung
của rulebook), sandbox. **Bỏ khỏi hệ nộp:** `hints.py`, `qnorm`, `qparse`, mọi cờ / precheck viết tay của P, thẻ quy nạp văn xuôi.
`industry.py` (toolkit ngành viết tay cho 3 ngành public) **không** dùng khi đo `pub_noindH` và là thứ W phải tự viết lại cho ngành mới.

## 2. Phân luồng lỗi (fine-tune hay luật) — tự động, chạy trên DN train sau mỗi vòng

Với mỗi câu sai / bỏ của S trên DN train (`tr_dev_para_a/b`, 4494 câu đã có log v3a):
1. có mẫu đúng trong K mẫu → **V** (dữ liệu bộ chọn);
2. không có: chạy lại S **greedy** với ngữ cảnh trần (định nghĩa + hàm đúng của toolkit cho thuật ngữ câu dùng) → đúng thì lỗi thiếu
   ngữ cảnh → **W** phải sinh được đơn vị đó (kiểm W); vẫn sai → lỗi năng lực → **dữ liệu S** (RFT: chương trình đúng của câu / paraphrase);
3. đáp án là không trả lời được → dữ liệu "không trả lời được" cho S + nhãn cho V.

## 3. Quy công và nhận thay đổi

- Ma trận bóc tách trên cùng bộ, giải mã tất định ghép cặp (seed cố định, mẫu đầu greedy): S / S+W / S+W_trần / S+V / S+W+V.
- Nhận một thay đổi khi held-out gộp (pub_ho400c + tr_ho400c + pub_noindH + tr_newterm) tăng ≥ 0.5 điểm **và** không bộ nào giảm quá
  1 điểm; bootstrap 1000 lần cho khoảng tin cậy (ghi kèm). Đơn vị luật của W: nhận theo **cụm** (mọi đơn vị của một split cùng lúc),
  không theo từng đơn vị (bài học A/B vòng 0: nhận từng thẻ là nhiễu).

## 4. Gói việc, phân luồng, batch, thời gian (UTC; acc mới đủ 30 h quota từ 14/11 00:00)

| Mã | Việc | Ai / đâu | Batch / tham số | Thời gian | Phụ thuộc |
|---|---|---|---|---|---|
| **V1** | `training/data/selector_data.py`: đặc trưng nhóm ứng viên từ log (c1a/c1b + log cũ của DN train trong `Process/runs`); `src/selector.py` (đã có khung) → mô hình logistic / GBDT; đánh giá offline trên log `q3l70ho` / `q3l70pf` (không chạy GPU) | container CPU | — | 13/11 17:00–20:00 | log có sẵn |
| **W1** | `lrl_gen.py --mode unit`: prompt = API toolkit + card schema DN + dòng rulebook + 6–8 cặp mẫu (dòng train ↔ hàm toolkit); đầu ra JSON đơn vị; `src/units.py`: nạp đơn vị vào namespace chương trình (exec trong sandbox), kiểm chạy trên dữ liệu DN (2 kỳ: tháng cuối, toàn bộ), bỏ đơn vị lỗi / suy biến; khối prompt `# SHOP FUNCTIONS` (chữ ký + định nghĩa) | container CPU (code + test mô hình giả) | — | 13/11 17:00–22:00 | — |
| **S1** | dữ liệu học tiếp v4 (CPU): RFT = chương trình đúng của S trên câu train sai / bỏ ở vòng trước (từ log c1a/c1b, chọn mẫu đúng) + UD = bản ghi "dùng đơn vị luật" (ẩn một hàm toolkit, đưa nó dưới dạng đơn vị trong `# SHOP FUNCTIONS`, nhãn gọi đơn vị) + DR có sẵn + 30 % mẫu v3a (chống quên); ~6M token | container CPU | max_len 4608 | 13/11 20:00–23:30 | W1 (định dạng khối) |
| **E0** | mốc `pub_noindH`: Q2 v3a × 0.70 + `--hide-api industry` (112 câu) | 1 acc còn quota tối 13/11 (caohuyen74 1.8 h) | DP 4 | 0.4 h | — |
| **S2** | **v3b** huấn luyện lại (đã sửa lỗi lưu) | acc A, FSDP 4 | mb 4, eff 64, max_len 4608, r 32, α 64, lr 2e-4, 1 epoch, 19.5M token | 14/11 00:05 → ~03:30 | — |
| **S3** | **v4 seed 1 / seed 2**: `--base-adapters V3A --base-weights 0.70` + LoRA r 32 trên dữ liệu S1 | acc B, C (FSDP 4 mỗi acc) | như S2, ~6M token | 00:05 → ~01:30 | S1 |
| **W2** | W chạy trên rulebook public + train (unit) với Qwen gốc fp8 | acc D | DP 4 (vLLM, max_num_seqs 128) | 00:05 → ~00:45 | W1 |
| **G1** | đánh giá v3a / v4s1 / v4s2 / v3b × {S, S+W} trên `pub_full`, `pub_ho400c`, `tr_ho400c`, `pub_noindH`, `tr_newterm` (mỗi cấu hình 2 shard: `pub_full` / held-out) + log đủ mẫu cho V | 8–16 phiên trên 8 acc (2 phiên / acc) | DP 4, max_num_seqs 128, max_batched_tokens 16384, gpu_util 0.92 | 01:45 → ~03:00 (v3b: ~04:00 → 05:00) | S3, W2 |
| **V2** | huấn luyện lại bộ chọn trên log G1 của DN train (`tr_dev_para` shard thêm vào G1 nếu cần) → đánh giá offline mọi cấu hình | CPU | — | sau G1, ~0.5 h | G1 |
| **R1** | vòng 2: phân luồng lỗi (§2) trên log G1 → sửa W (ví dụ mẫu / kiểm), dữ liệu S (RFT vòng 2) → v5 (học tiếp từ cấu hình thắng) | CPU + 1–2 acc | như S3 | 14/11 06:00 → 12:00 | G1, V2 |
| **G2** | đánh giá vòng 2 + ma trận bóc tách đầy đủ (§3) | 8 phiên | như G1 | 14/11 12:00 → 14:00 | R1 |
| **F1** | cấu hình cuối `configs/flow_s_w_v.json`; huấn luyện lại S cuối trên 30 DN (công thức thắng, thêm 6 DN held-out) nếu kịp | 1 acc / seed | như S2 | 14/11 14:00 → 18:00 | G2 |
| **F2** | đánh giá cuối: `pub_full` × 2 seed (tất định), held-out (trên công thức 24 DN), thời gian 1000 câu; đường container (W chạy lúc suy luận trong `predict_flow`, V nạp từ `refs/`) | 4 phiên | — | 14/11 18:00 → 15/11 00:00 | F1 |
| **D1** | Docker: `docker/` (S hợp nhất + adapter, W gọi lúc suy luận, V trong image), `docker/README_SOLUTION.md`, `final_submission/`; user build | container | — | 15/11 00:00 → 12:00 | F2 |

**Thứ tự cắt khi thiếu thời gian:** R1 / G2 (một vòng thay vì hai) → huấn luyện 30 DN ở F1 → W chỉ đơn vị cho dòng rulebook **mới** (không
có trong toolkit) → V chỉ ngưỡng bỏ trả lời. **Không cắt:** S3 (v4), W1 + W2 (đơn vị cho ngành mới), V1, F2, D1.

## 5. Kiểm trước khi chạy (cổng)

`tests/run_all.sh` (thêm `test_units.py`: W giả sinh đơn vị → nạp → chạy → đơn vị lỗi bị bỏ; `test_selector.py`: bộ chọn tất định, đơn
điệu theo tỉ lệ phiếu, không dùng id DN / câu); smoke CPU của mọi job bằng mô hình giả; `automation/preflight.py` (job `soup_job`: smoke qua
`src/eval_flow.py` với cấu hình thật như 13/11).

## 6. Rủi ro

| Rủi ro | Giảm |
|---|---|
| W sinh hàm sai mà vẫn chạy được | kiểm trên 2 kỳ + so `tests` trong đơn vị; nhận theo cụm bằng A/B ghép cặp; trần (hàm thật) cho biết khoảng cách |
| S không dùng đơn vị dù có | dữ liệu UD trong v4; đo S + W_trần |
| Bộ chọn quá khớp log v3a | huấn luyện lại trên log của chính S cuối (V2); đặc trưng không phụ thuộc DN |
| Thời gian container | W chạy một lần / split (~10–20 phút cho mọi dòng rulebook trên 2×5090), song song với ingest |

## 7. Nhật ký

**13/11 16:40 — V1 xong (CPU, không GPU):** bộ chọn học được (`src/selector.py`, đặc trưng bất biến) huấn luyện trên log của S (v3a × 0.70,
Q2) ở 4494 câu paraphrase train (`c1a`, `c1b`), θ 0.45 chọn trên train; phát lại trên log đánh giá: public 0.928 → 0.923, `pub_ho400c`
0.609 → **0.647**, `tr_ho400c` 0.904 → 0.912, `pub_noind` / `tr_newterm` không đổi → held-out gộp **0.776 → 0.794** (+1.8) → nhận (§3).
Nối vào luồng: cờ `--selector` (agent / `hybrid.finish`), `src/refs/selector/selector_v3a_inv.json`, cấu hình `configs/flow_swv.json`
(= Q2 + V), `tests/test_selector_flow.py`; smoke CPU qua `eval_flow` với mô hình giả OK. V2 sẽ huấn luyện lại trên log của S cuối.

**13/11 17:15 — E0 (`thnliu/zc-e0h-1`, v3a × 0.70, `pub_noind` 112 câu, mọi hàm ngành bị ẩn = `pub_noindH`):** Q **−0.1607** (33 / 51 / 28),
Q2 **−0.0357** (36 / 40 / 36) — so với 0.821 / 0.830 khi còn toolkit ngành viết tay. **Toàn bộ điểm "ngành mới" trước đây là của
`industry.py`**; với ngành mới thật (private) hệ thuần gần như trả lời sai hết → W (hàm do mô hình viết từ rulebook ngành) + V (bỏ trả lời
khi không chắc) là phần quyết định. G0 (S + W + V trên `pub_noindH`) đang chạy.

**13/11 18:20 — G0b (W có suy nghĩ, 5 mẫu, tự nhất quán):** `pub_noindH` **−0.036** (42 / 46 / 24) — kém G0 (0.036): 317 / 389 mẫu suy nghĩ chưa
đóng `</think>` ở trần 4096 token → W chỉ có hàm cho dev_012 / dev_013, không có cho 2 cửa hàng điện thoại. Sửa: bỏ suy nghĩ, giữ 5 mẫu +
tự nhất quán, prompt của W thêm 2 dòng dữ liệu mẫu mỗi bảng (C và T: nhãn, đơn vị, mã thật). **G0c** chạy trên phuongnehe.

**13/11 19:20 — G0c** (W không suy nghĩ, 5 mẫu): `pub_noindH` **−0.107**; chỉ 19 hàm (4 cho DN ngành). Tái hiện trên máy: 33 / 50 đầu ra của W
không có JSON (mô hình vẫn lập luận bằng văn xuôi tới trần 900 token), một số chép nguyên khuôn `def ten_ham(p, ...)` của câu lệnh, `import re`
bị chặn; automap chạy cho mọi DN (15 phút). Sửa: trần 2560 token cho W, câu lệnh không còn khuôn mã, cho phép import thư viện an toàn,
automap chỉ cho DN có dòng cần viết (dùng lại khi kiểm). Ghi nhận: S + W + V v3a trên public 0.919 (W cũ có suy nghĩ, 10 hàm, 1125 s).
**G0d** (`pub_noindH`) chạy trên namhunhh. Lỗi public còn lại của S + W + V v3a: 26 sai không mẫu nào đúng (năng lực S, 24 dùng `month(...)`),
18 có mẫu đúng (V), 2 bỏ.

**13/11 20:15 — G0d** (trần 2560): `pub_noindH` **0.009** (40 / 39 / 33); W vẫn chỉ ra JSON ở 17 / 125 đầu ra (mô hình lập luận bằng văn xuôi tới
trần dù tắt suy nghĩ) → 5 hàm. Sửa gốc: **sinh có ràng buộc JSON** (vLLM structured outputs, schema: `plan` ≤ 700 ký tự cho lập luận ngắn,
`name`, `signature`, `doc`, `triggers`, `code`), `agent.Engine.gen(json_schema=...)`, có đường lùi không ràng buộc nếu vLLM từ chối schema.
**G0e** chạy trên huuhien95.

**13/11 20:50 — G0e** (JSON có ràng buộc): `pub_noindH` **−0.018** (38 / 40 / 34), 12 hàm (480 s). JSON nay parse được nhưng mã kém: một dòng nối
bằng `;` (một `#` nuốt phần còn lại), sai tên bảng, NaN. **Lưu ý thống kê:** với 112 câu, sai số chuẩn của PA ≈ 0.08 → G0 / G0b / G0c / G0d /
G0e (0.036 … −0.107) **chưa khác biệt có ý nghĩa** với không có W (−0.036). Sửa: `code` = mảng dòng (schema), bỏ `import toolkit`,
chỉ nhận hàm khi ≥ 2 mẫu chạy được cho cùng kết quả. Hướng còn lại nếu W vẫn ≈ 0: V bỏ trả lời khi câu ngành không có hàm đáng tin
(học từ log, không luật viết tay) để không mất điểm ở ngành mới.

**13/11 21:40 — W dùng trọng số của S; giới hạn thời gian; đo tách W (đợt 00:05).** Image chỉ chứa trọng số đã gộp của S (một mô hình
9.65B; thêm Qwen gốc thì vượt 15B) → W mặc định = mô hình S (`"units_model": "solver"`, `eval_flow` / `predict_flow` cùng một đường);
`"base"` giữ cho so sánh. `prepare_units` có trần `units_max_s` (mặc định 2700 s): shard nào quá trần bị giết cả nhóm tiến trình (cả
engine vLLM), chạy tiếp S + V không đơn vị (`tests/test_units.py` [4]). Image kiểm import thêm `units`, `lrl`, `lrl_gen`, `selector`.
Đo thêm ở đợt 00:05 (`plan_w14c.json`, v3a × 0.70): **e0hv** S + V không W (`configs/flow_sv_h.json`, `pub_noindH` + `tr_newterm`) = mốc đúng
cho W (E0 chưa có V); **g0hf** W = S; **g0hg** W = Qwen gốc (`flow_swv_hb.json`); **e0hvp / g0hfp** không W / có W trên `pub_ind` (269 câu gốc public của 5 DN có mục ngành) + `pub_hoind` (345 câu
paraphrase của các DN đó, held-out) với hàm ngành ẩn — tập 614 câu để thấy tác dụng W (rulebook train không có mục ngành nào: `tr_ho400c` ẩn hàm
ngành giống hệt bản thường, nên không dùng); **g1v3apf2** S + W + V trên `pub_full` với W mới (điểm public của đường container).

**13/11 22:20 — đo W trực tiếp (offline, chỉ để phân tích) + W tự sửa lỗi.** So 12 hàm G0e giữ lại với hàm ngành tham chiếu (`industry.py`, chỉ dùng
làm thước đo) trên 24 tháng: `tien_giam_voucher` 24 / 24, `khach_thanh_vien_moi_thang` 24 / 24, `phi_ship` 23 / 24 đúng; `thu_may_cu` trả dict
hai đại lượng (đúng một trong hai), `tyle_bao_hanh` ra tỉ lệ 0.034 thay vì số phần trăm 3.39, `so_xe` lệch (431 / 387), hàm điểm lỗi
`TypeError` ở mọi kỳ. Trên mẫu thô G0e: phần lớn mẫu hỏng do cú pháp (mã một dòng — G0f đã sửa bằng mảng dòng) và lỗi API (`p[...]`,
`pd.to_datetime` trên bảng). Sửa (chung, không theo DN): (1) `units.verify` chạy trên 12 tháng cuối + toàn kỳ (13 kỳ, chữ ký tự nhất quán
sắc hơn), lỗi ngoài `Unanswerable` ở > ¼ số kỳ → loại, dict nhiều đại lượng tên biến → loại; (2) **vòng tự sửa** (`--unit-repairs 1`,
cấu hình `units_repairs`): hàng chưa có hàm được chọn → một mẫu hỏng + lỗi thật khi chạy trên dữ liệu DN đưa lại cho W → 3 mẫu mới →
kiểm + chọn lại (Chen et al. 2023, self-debugging); (3) câu lệnh: một giá trị đúng dạng đáp án, phần trăm = số phần trăm (`pct`),
ví dụ mẫu thêm một hàm tỉ lệ. Kiểm lại 12 hàm G0e bằng bộ kiểm mới: loại đúng 2 hàm sai (dict, lỗi 13 / 13 kỳ). `tests/test_units.py`
(mô hình giả): vòng tự sửa 1 → 4 hàm.

**14/11 00:38 — e0hv (S + V, không W, hàm ngành ẩn; v3a × 0.70):** `pub_noindH` **−0.107** (34 / 46 / 32), `tr_newterm` 0.852 (50 / 4 / 0). V
không cứu được ngành mới khi không có W (E0 không V: −0.036; V bỏ trả lời 32 câu nhưng vẫn sai 46). So với mốc đúng này, các bản W đã chạy:
G0 +0.143, G0d +0.116, G0e +0.089, G0b +0.071, G0c 0.000 → W có tác dụng dương ở `pub_noindH` (từng lần vẫn trong nhiễu; chờ tập 614 câu).

**14/11 01:00 — LỖI: V chưa từng chạy trong các lần chạy thật.** g1v3aho (S + W + V v3a, held-out): `pub_ho400c` 0.6034, `tr_ho400c` 0.9173 — log
không có `conf_sel`: đường dẫn `src/refs/selector/...` của cấu hình không có trong kernel Kaggle (code đóng gói thành `<stage>/refs/...`,
không có tầng `src/`) và `hybrid.selector` **lặng lẽ** quay về bỏ phiếu thường. Mọi lần chạy `flow_swv*` / `flow_sv_h` đến nay (G0…G0e,
g1v3apf 0.919, g1v3aho, e0hv và 5 kernel của đợt 00:05) là **S (+ W) không V**; so sánh W / không W vẫn hợp lệ (cả hai đều thiếu V);
e0hv = E0 lặp lại (−0.107 vs −0.036: nhiễu giữa hai lần chạy ~0.07 trên 112 câu). Sửa: thử thêm đường dẫn bỏ `src/`, thiếu file thì
**dừng** (không lùi lặng lẽ); `tests/test_selector_flow.py` kiểm bố cục kernel + trường hợp thiếu file. Phát lại V (cùng mẫu, chỉ đổi
cách chọn) trên log thật: `pub_noindH` −0.107 → **+0.018**, `tr_newterm` 0.852 → 0.852, `pub_ho400c` 0.603 → **0.652**, `tr_ho400c`
0.917 → **0.928** → V có ích ở mọi tập. G1 v4 (sau huấn luyện) sẽ là lần đầu đo S + W + V thật.

**14/11 00:56 — G0f (W = trọng số S, mã = mảng dòng, kiểm 13 kỳ, vòng tự sửa; không V do lỗi trên):** 41 hàm đã kiểm (G0e: 12; 867 s).
`pub_noindH` **+0.018** (42 / 40 / 30) vs không W (e0hv) −0.107 → **+0.125**; `tr_newterm` 0.796 vs 0.852 (−3 câu). Phát lại V: `pub_noindH`
+0.036 vs không W + V +0.018 (+0.018), `tr_newterm` 0.778 vs 0.852. Chờ cặp 614 câu (e0hvp / g0hfp) để kết luận.

**14/11 00:57 — G0g (W = Qwen gốc, còn lại như G0f):** 37 hàm (981 s); `pub_noindH` 0.009 (V phát lại 0.045), `tr_newterm` 0.833 (V 0.815)
— ngang G0f (W = S: 0.018 / 0.036; 0.796 / 0.778) → **giữ W = trọng số S** (một mô hình trong image, đúng giới hạn 15B).

**14/11 00:58 — g1v3apf2 (S + W mới, hàm ngành có sẵn, v3a × 0.70; không V do lỗi trên):** `pub_full` **0.936** (961 / 25 / 14) — cao nhất
của hệ không luật viết tay (Q2 0.928; g1v3apf W cũ 0.919); `pub_ind` 0.900, `pub_noind` 0.830; W 15 hàm (775 s), đánh giá 1994 s. Phát lại V:
`pub_full` 0.929 (bỏ thêm 7 câu đúng; ở held-out V +0.05 / +0.01 — đổi chác chấp nhận được theo quy tắc held-out trước).

**14/11 01:04 — e0hvp (S, không W, không V, hàm ngành ẩn; 614 câu của 5 DN ngành):** `pub_ind` **0.476** (186 / 58 / 25), `pub_hoind`
**0.415** (220 / 77 / 48); V phát lại 0.535 / 0.458. Có hàm ngành viết tay (g1v3apf2) `pub_ind` 0.900 → ngành mới thật mất ~0.42 trên DN
ngành: khoảng W phải lấp. (Tập con `pub_noindH` lần này −0.188: ba lần chạy cùng cấu hình −0.036 / −0.107 / −0.188.)

**14/11 01:05 — g0hfp: W CÓ TÁC DỤNG (kết luận W2).** Cùng 614 câu, hàm ngành ẩn, v3a × 0.70, không V; W = trọng số S, 17 hàm cho 5 DN (629 s):
`pub_ind` 0.476 → **0.595** (+0.119, SE ghép cặp 0.031, z 3.8; 28 câu tốt hơn / 7 kém hơn), `pub_hoind` (held-out) 0.415 → **0.493**
(+0.078, SE 0.023, z 3.3; 27 / 10). Có V (phát lại): 0.535 → 0.595, 0.458 → 0.525. → **giữ W** (W2 xong). W mới lấp ~0.12 / 0.42 khoảng
cách tới hàm viết tay (0.900) → hướng cải tiến tiếp cho W (sau kế hoạch): nhiều mẫu hơn, ví dụ mẫu tốt hơn, vòng tự sửa thứ 2, S học dùng
hàm của W (dữ liệu UD của v4).

**14/11 02:00 — g0hfq (W mạnh hơn: 8 mẫu / hàng, 2 vòng tự sửa; V chạy thật lần đầu — 613 / 614 log có `conf_sel`):** 19 hàm (609 s);
`pub_ind` 0.5725, `pub_hoind` 0.4754 (có V). So ghép cặp với W mặc định g0hfp (cùng V phát lại): `pub_ind` −0.022 (SE 0.022), `pub_hoind`
−0.049 (SE 0.022) → **không nhận**; giữ W mặc định (5 mẫu, 1 vòng tự sửa).

**14/11 02:30 — quyết định user: chỉ tối ưu public, không held-out.** Bỏ R1, G2 và mọi đánh giá held-out mới (các kernel held-out / ẩn ngành
đang chạy để chạy xong, không dùng để quyết định). Phần còn lại: G1 public → chọn S theo public → V2 (bộ chọn học lại trên log train của S)
→ biến thể cấu hình chạy mỗi GPU một biến thể → F1 → F2 → D1 → dừng, PDF 1 / PDF 2, dọn repo.

**14/11 03:03 — G1 v4 seed 2 (V3A × 0.70 + V4S2 × 1.0, S + W + V thật — 1000 / 1000 log có `conf_sel`):** `pub_full` **0.916** (949 / 33 / 18;
bỏ phiếu thường trên cùng log 0.920); `pub_ind` 0.859, `pub_noind` 0.741; W 12 hàm (698 s), đánh giá 2022 s. **Kém v3a** (S + W 0.936,
`pub_noind` 0.830) → v4 (seed 2) không giúp public; chờ seed 1 và lưới biến thể.

**14/11 03:10 — G1 v4 seed 1:** `pub_full` 0.921 (bỏ phiếu thường 0.925) — cả hai seed v4 < v3a (0.936) → **S = v3a × 0.70**.
Phát lại ngưỡng trên public (3 lần chạy v3a / v4s1 / v4s2): bỏ phiếu θ 0 / 0.3 / 0.4 / 0.5 / 0.6 và V θ 0 / 0.2 / 0.3 / 0.45 chỉ đổi ±0.005;
trung bình bỏ phiếu θ 0.5 = 0.927, V θ 0.45 = 0.922 → trên public V không có lợi (lợi ở held-out). Chọn cuối ở F1 sau lưới k16 / soup.

**14/11 03:41 — lưới public (1):** soup V3A × 0.70 + V4S1 × 0.5 + V4S2 × 0.5 (k 8): 0.925 (bỏ phiếu 0.928) < v3a. **k_max 16** (k_fast 4) với
V4S2: 0.916 → **0.926** (bỏ phiếu 0.920 → 0.932), đánh giá 2022 → 2601 s → nhiều mẫu hơn có ích. v3b xong (87 bước, 19.2M token, loss 0.027):
chỉ đánh giá public (k 8 `g1v3bpf`, k 16 `pv3bk16`).

