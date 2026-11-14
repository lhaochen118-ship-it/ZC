# 18 — Kế hoạch 5: hai LoRA huấn luyện riêng (S giải — W viết luật / nối luật / phán xử), hướng tới public = 1

*Lập 14/11/2026 06:30 UTC (= T0). User duyệt 14/11: (1) S đóng băng (v3a × 0.70) + LoRA-W ba vai, huấn luyện riêng (không huấn luyện chung);
(2) được dùng bộ khớp thuật ngữ viết tay của P **chỉ để gán nhãn câu train** cho W2 (lúc suy luận không có luật viết tay); (3) image = 1 base
9.65B + 2 LoRA, gộp adapter lúc nạp theo pha nếu vLLM không chạy LoRA động với fp8; (4) đóng băng **16/11 04:00 UTC** (11:00 giờ VN) để user
build Docker; (5) dùng acc hợp lý, không spam (11 token: 10 acc 4×L4 + `yeshenhi` T4). Chỉ tối ưu public (user 14/11 02:30).*

## 0. Bằng chứng (đáp án chuẩn public, log `pva3k16` phát lại bỏ phiếu thường: 963 / 25 / 12 = 0.938)

| Nhóm lỗi | Câu | Ai sửa |
|---|---|---|
| Có chương trình đúng trong mẫu nhưng thua phiếu (7: nhóm đúng lớn nhất nhưng < 0.5 vì mẫu lỗi chạy; 16: nhóm đúng thiểu số) | 23 | **W3** phán xử |
| Không mẫu nào đúng: 12 đồng thuận 4 / 4 sai (4 phẩm chất chủ quan → `unanswerable`; 5 thuật ngữ rulebook không nối đúng hàm; tỉ trọng theo số lượng; kỳ "1/26"; bỉm size M gộp hãng) + 2 sai sau 16 mẫu | 14 | **W1 + W2** (luật vào schema, nối luật theo câu), W3 (từ chối câu chủ quan) |

Thêm: gộp 11 lần chạy / 4 adapter có mẫu đúng ở 991 / 1000 câu nhưng gộp phiếu chỉ 0.931–0.947 → nút thắt là **chọn** và **nối luật**, không phải
năng lực sinh. Thẻ luật hiện khớp bằng từ khoá ("ai trung thành nhất" nhận 4 thẻ không liên quan; "bình quân 1 phiếu" nhận đúng thẻ nhưng
`code_hint` rỗng). Dữ liệu sẵn cho W3: 8 lần chạy S trên 2247 câu paraphrase của DN train = 18.8k nhóm đáp án đúng, 3.1k nhóm sai, 697 câu lẫn
đúng / sai, ~1.5k câu đồng thuận sai.

## 1. Kiến trúc và luồng suy luận (3 pha tuần tự — thử nghiệm tách được theo pha)

| Pha | Trọng số | Việc | Đầu ra |
|---|---|---|---|
| **A** (một lần / split) | S gộp + LoRA-W | **W1**: mọi dòng rulebook → mục "lớp ngữ nghĩa" `{thuật ngữ, định nghĩa, binding}`; binding = lời gọi toolkit hoặc hàm mới, **bắt buộc chạy được** trên dữ liệu DN (13 kỳ, tự sửa 1 vòng; không còn `code_hint` rỗng). **W2**: mỗi câu → id các mục áp dụng (đầu ra ~20 token) + cờ "dữ liệu không ghi" | `layer.jsonl`, `links.jsonl` |
| **B** | S gộp (v3a × 0.70, không đổi) | S viết chương trình như hiện tại (k_fast 4 / k_max 16); prompt thay khối thẻ khớp từ khoá bằng `# RULES FOR THIS QUESTION` (mục W2 chọn + binding) | `logs.jsonl` (mẫu, kết quả chạy) |
| **C** | S gộp + LoRA-W | **W3**: câu + luật đã nối + các chương trình ứng viên khác nhau + kết quả chạy → chọn một nhóm / `unanswerable` / bỏ. Câu bị bác mà không có ứng viên thay: S thử lại 1 lần với ghi chú của W (chỉ ở vòng 2 nếu kịp) | `submission.jsonl` |

- Pha tuần tự = không phụ thuộc LoRA động của vLLM (rủi ro hạ tầng = 0); LoRA động chỉ là tối ưu thời gian nếu smoke qua (§3 P0).
- W2 không đọc đầu ra của S → pha A làm xong mọi câu trước; W3 chấm được **trên log S cố định** → mỗi biến thể W3 ~35 phút / GPU, quy công sạch.
- Tham số: 9.65B + LoRA-S (đã gộp) + LoRA-W r 32 (~0.08B) < 15B. Adapter W huấn luyện **chồng lên S đã gộp** (`--base-adapters V3A --base-weights 0.70`).

## 2. Dữ liệu huấn luyện LoRA-W (CPU, chỉ từ train; không câu public nào)

| Bộ | Nguồn | Bản ghi | Nhãn |
|---|---|---|---|
| **DW3** phán xử | log 8 lần chạy `c1a/b`, `rq5_tr_dev_para_a/b`, `v2v4s{1,2}{a,b}` | 697 câu lẫn (mọi lần chạy) + ~1.5k đồng thuận sai + ~1.5k đồng thuận đúng (cân bằng) + ~500 mẫu sai do **đột biến chương trình** (đổi `by=`, hàm anh em, lệch kỳ, bỏ bộ lọc) + câu `unanswerable` train (70 gốc + paraphrase) | chỉ số ứng viên đúng / `none` / `unanswerable`, từ đáp án train |
| **DW2** nối luật | 2,119 câu D1 + paraphrase D2 + câu nhiễu D2N (lấy mẫu ~4k) | câu + danh sách mục đánh số của rulebook train (+ mục "thuật ngữ mới" DR để mô hình không học thuộc danh sách) | hợp của: mục có binding xuất hiện trong chương trình đã kiểm, và `rule_terms` của bộ khớp P trên câu train (user duyệt điểm 2) |
| **DW1** biên dịch | dòng rulebook train ↔ hàm toolkit; DR `newterm_in` (thuật ngữ mới ↔ chương trình đã kiểm, tổng quát theo kỳ `p`) | ~0.6–1k | binding chạy đúng trên dữ liệu DN |

Ngân sách token mỗi nhánh ~15M (max_len 4608) ≈ 2.4–2.8 h FSDP 4×L4 (đo 1.75k tok/s). Tiền tố prompt = ngữ cảnh DN như của S (dùng lại prefix cache).

## 3. Lịch chạy (UTC; T0 = 14/11 06:30; đóng băng T+45.5 = 16/11 04:00)

| Mã | Giờ | Việc | Phiên GPU (acc) | Cổng ra |
|---|---|---|---|---|
| **P0** | T+0 → 3.5 (06:30–10:00) | Code CPU: `training/data/w_judge.py`, `w_link.py`, `w_compile.py`; `src/judge.py` (pha C + phát lại trên log), `src/link.py` (pha A-W2), W1 bắt buộc binding; `predict_flow` 3 pha; `soup_job --phase`; test CPU mô hình giả + `preflight` | 1 phiên 20 phút (acc 1): smoke vLLM `enable_lora` + fp8 | `tests/run_all.sh` ALL PASS, `PREFLIGHT OK` |
| **Z** (vòng 0, không huấn luyện) | T+2.5 → 5 | W = trọng số S, zero-shot: **Z-judge** 4 biến thể / 1 phiên (mỗi GPU một biến thể: chọn theo danh sách / chấm từng ứng viên × có / không suy nghĩ) trên log `pva3k16`; **Z-link** và **Z-layer** (W1 mọi dòng + binding bắt buộc) chạy đủ luồng `pub_full` | 3 phiên (acc 1: 2 phiên, acc 2: 1 phiên) | mốc zero-shot cho từng vai; mẫu số phiếu = mẫu hợp lệ (phát lại 0.942) nhận ngay nếu Z xác nhận |
| **T1** huấn luyện W vòng 1 | T+3.5 → 7 | 4 nhánh song song, mỗi nhánh 1 phiên FSDP 4 GPU: **W-mix** (DW1 + DW2 + DW3), **W3-list**, **W3-point**, **W2-only**; mb 4 × tích luỹ 16 (eff 64), bf16, r 32, α 64, lr 2e-4, 1 epoch, checkpoint theo hạn 12 h | 4 phiên (acc 2: 1 phiên, acc 3: 2, acc 4: 1) | loss giảm, adapter đăng dataset trên acc đánh giá |
| **E1** đánh giá vòng 1 | T+7 → 9.5 | Phát lại W3 trên log `pva3k16` + `g1v3apf2` (2 biến thể adapter × 2 ngưỡng / phiên); đủ luồng `pub_full` cho W2-only và W-mix (2 seed) | 5 phiên (acc 4: 1, acc 5: 2, acc 6: 2) | **G1 (T+10):** chọn dạng W3, có / không W2; mỗi vai giữ khi public tăng ghép cặp ≥ +0.005 ở cả 2 seed / 2 log |
| **R2** vòng 2 | T+10 → 20 | (a) khai thác lỗi của **chính W** trên DN train: S + W chạy 2 shard `tr_dev_para` → mẫu khó → W v2 (học tiếp từ v1, ~6M token, ~1.5 h, 2–3 biến thể); (b) vòng thử lại của S với ghi chú W; (c) phân tích lỗi public gốc còn lại → chỉ sửa **dạng dữ liệu / prompt chung**, không vá theo câu | 2 + 3 + 5 phiên (acc 6–8) | **G2 (T+20):** cấu hình ứng viên cuối |
| **R3** xác nhận + quy công | T+20 → 30 | 3 seed cấu hình cuối; ma trận 2 × 2 trên cùng seed: S / S + W1W2 / S + W3 / đủ → biết điểm tăng do vai nào | 8 phiên (acc 8–10) | PA trung bình 3 seed, bảng quy công |
| **F** đóng gói | T+30 → 38 | Đường container (`predict_flow` 3 pha) × 2 trên public + đo thời gian 1000 câu; đăng adapter W cuối (`zc-final-adapter` thêm thư mục W + `info/`), sha256, `docker/`, `README_SOLUTION`, `final_submission/` | 3 phiên (acc 9–10) | container = eval ± 0.01; thời gian ≤ 1.5 × hiện tại |
| **D** tài liệu | T+38 → 42 | STATUS, PDF 1 / 2 / 3, dọn repo | — | bàn giao |
| Dự trữ | T+42 → 45.5 | chạy lại khi phiên chết / hàng đợi | — | đóng băng 16/11 04:00 |

**Thứ tự cắt khi trễ:** vòng thử lại của S (R2b) → W v2 (R2a) → seed thứ 3 → LoRA động. **Không cắt:** Z, T1, E1, quy công 2 × 2, đường container.
**Lùi an toàn:** vai nào không qua G1 / G2 thì tắt bằng cấu hình; mọi vai trượt → nộp hệ hiện tại + mẫu số phiếu hợp lệ.

## 4. Acc, quota, GPU

- Tổng ước ~56 giờ tường phiên 4×L4 ≈ 125 h quota (hệ số 2.2) trên ~275 h của 10 acc L4 → **10 acc hiện có là đủ**, không cần thêm token;
  đỉnh đồng thời 5–8 phiên = 3–4 acc, mỗi acc dùng đủ 2 phiên rồi mới sang acc kế (playbook mục 9). `yeshenhi` (T4): chỉ kernel CPU / dự phòng.
- Trước mỗi lần đẩy: `automation/quota.py` + `est_h` trong plan của `launch_many.py` (từ chối acc không đủ giờ); phiên ≤ 4 h, xa trần 12 h;
  huấn luyện vẫn bật checkpoint (`ZC_SESSION_S`, `--resume-from auto`).
- Mỗi phiên dùng cả 4 GPU: huấn luyện = FSDP 4 GPU; phát lại W3 = mỗi GPU một biến thể (`--parallel-gpus 4`); đủ luồng = DP 4
  (`max_num_seqs 128`, `gpu_util 0.92`, fp8). Không có L4 → 2×T4 chỉ cho smoke.
- Theo dõi bằng `watch_done.py` nền (timeout 7200000 ms), không poll.

## 5. Đo và nhận thay đổi

- So **ghép cặp trên cùng câu**, ≥ 2 seed (nhiễu một lần chạy public ±0.004–0.01); W3 đo trên log cố định nên không có nhiễu sinh.
- Chỉ số theo dõi: PA `pub_full`; số câu của hai nhóm §0 được sửa / bị làm hỏng; thời gian 1000 câu.
- Kỳ vọng: 0.96–0.975 khả dĩ, ~0.98 tốt nhất (ước lượng từ số câu mỗi nhóm, chưa đo); 1.0 không thực tế trong 45 h.

## 6. Rủi ro

| Rủi ro | Giảm |
|---|---|
| W3 học trên log DN train (S đã thuộc các DN này) → phân phối lỗi khác public | mẫu đột biến chương trình + câu paraphrase / nhiễu; mốc zero-shot (Z) để biết huấn luyện có giúp không; vòng 2 khai thác lỗi của chính W |
| Paraphrase train đổi nghĩa (nhãn nhiễu) | chỉ dùng paraphrase mà ≥ 1 lần chạy có mẫu đúng, hoặc câu gốc |
| W2 chọn thiếu mục → S mất định nghĩa | prompt S giữ nguyên rulebook đầy đủ; khối W2 chỉ **thêm** |
| W3 bác nhầm câu đúng | ngưỡng: chỉ đổi đáp án đồng thuận khi W3 chọn `unanswerable` / `none` với xác suất cao; đo số câu bị làm hỏng ở G1 |
| Thời gian container tăng (3 lần nạp mô hình + 2 lượt W mỗi câu) | đầu ra W ngắn (≤ 60 token), prefix cache; đo ở F; LoRA động nếu smoke qua |
| Lỗi nạp lặng lẽ (bài học V 14/11) | thiếu adapter W / `layer.jsonl` / `links.jsonl` → **dừng**, log ghi dấu `w2`, `w3` từng câu |

## 7. Nhật ký

*(ghi tại đây từng lần chạy: mã, acc, giờ, kết quả)*

**14/11 06:35 — T1 + Z đẩy.** wjl1 / wjp1 (quynhle07 / quynhtran35) chết "STALE CODE": hai fanout trên cùng acc, kernel đầu bị Kaggle giữ
hàng đợi > 8 phút rồi được đẩy lại sau khi lần đẩy thứ hai đã thay dataset `zc-code` → **quy tắc: mỗi acc chỉ một lần đẩy cho tới khi kernel
trước RUNNING**. Đẩy lại wjl1b (yuan1103) / wjp1b (anderson1202). zjs / zjb lần 1 lỗi: Kaggle tự giải nén `.gz` trong dataset code
(`judge.find_file` nay thử cả hai tên; log gửi dạng không nén). Container khởi động lại 1 lần (~08:05), `automation/resume.py` gắn lại kernel.

**14/11 09:20 — Z-judge (zero-shot, log `pva3k16`, đáp án public, phát lại chính sách `tests/replay_judge.py`):** bỏ phiếu 0.938 (mẫu số = mẫu
hợp lệ: 0.942). Trọng số S không suy nghĩ: chỉ 5 % đầu ra có `<choice>` (S viết chương trình) → vô dụng; S có suy nghĩ: phán mọi câu 0.941
(sửa 13 / hỏng 8), **chỉ phán câu có mẫu bất đồng: 0.947**. Qwen gốc không suy nghĩ: 0.947 (chỉ câu bất đồng, τ 1.0). → `judge.py
--only-split 1` mặc định (~90 / 1000 câu, nhanh ~10 lần). W1 lần 1 (`w1b`): 98 / 125 dòng lập luận văn xuôi quá trần → JSON có ràng buộc (`w1b2`).

**14/11 09:30 — WJB1 xong** (list + point, 3025 bản ghi, 10.1M token, 143 bước, 2 h 32, loss 0.27 → 0.002, 1.72k tok/s, đỉnh 16.4 GB / GPU).

**14/11 10:10 — E1: WJB1 (W3 huấn luyện, list + point) phán trên 2 lần chạy S public, chỉ câu bất đồng:** list **0.950** (pva3k16; bỏ phiếu 0.938) /
**0.946** (g1v3apf2; bỏ phiếu 0.936); list+think 0.949 / 0.948; point 0.947 / 0.939 → W3 có ích ổn định +0.010–0.012 (zero-shot tốt nhất 0.947).
Lỗi còn lại trên pva3k16 (27): 13 câu 4 / 4 mẫu đồng thuận sai (không câu nào có mẫu đúng: phẩm chất chủ quan 4, thuật ngữ rulebook 5, kỳ / tỉ
trọng / bỉm 4) → việc của W1 / W2 (zlk, zlb); 14 câu bất đồng (phán xử chia phiếu hoặc chọn sai). W1 v2 (`w1b2`, JSON có ràng buộc): 56 / 125
dòng rulebook public có lời gọi toolkit đã chạy trên dữ liệu (`src/refs/lrl/compile_public_b.jsonl`).

**14/11 11:50 — E1 xong ba adapter W3 + Z-link (public, chỉ phán câu bất đồng, mẫu số hợp lệ):**

| Phán xử | pva3k16 (vote 0.938) | g1v3apf2 (vote 0.936) | gộp pva3k16 + zlk |
|---|---|---|---|
| zero-shot S (think) / Qwen gốc | 0.947 / 0.947 | — | — |
| **WJB1** (list + point) | **0.950** (kết hợp biến thể 0.953) | 0.946 / 0.948 | 0.945–0.950 (kết hợp 0.953) |
| WJL1 (list) | 0.949–0.952 | — | 0.942 |
| WJP1 (point) | 0.941–0.945 | — | 0.935–0.940 |

S + link zero-shot (zlk) 0.921, + W1 bind (zlb) 0.925 < 0.938: link sửa đúng câu thuật ngữ (trung thành, ưa chuộng, khách quen…) nhưng dòng
chung chung kèm hàm ("Đơn tính vào doanh thu → revenue(p)" ở 27 % câu) kéo câu khác sai → `link.py --max-df 0.05`. Linker huấn luyện **WL1**
(1800 bản ghi, nhãn bộ khớp P trên câu train): độ chính xác so với bộ khớp P trên public 0.78 (zero-shot 0.24). Gộp mẫu hai lần chạy S nâng trần
có-mẫu-đúng 986 → 991 nhưng phán xử chưa tận dụng được. Lỗi còn lại sau phán xử chủ yếu là **kiến thức rulebook** (prompt phán xử không có dòng
rulebook) → `judge.py --links` (dòng W2 trong prompt), WJR1 = phán xử huấn luyện kèm dòng rulebook (nhãn bộ khớp P trên câu train).

**14/11 12:30 — kết quả tốt nhất đến nay (offline, log pva3k16):** W3 WJB1 + dòng rulebook do WL1 nối trong prompt phán xử: **0.954** (kết hợp biến thể
0.955; bỏ phiếu 0.938). S + link WL1 (zlw): 0.940 (ngang mốc). LoRA động vLLM fp8 với adapter thô: không tác dụng → thử `merge.py vllm` (ejb1lora2);
container mặc định gộp adapter W theo pha (`w_mode: merge`). Lỗi vận hành và quy tắc tránh: PDF 2 §9.
