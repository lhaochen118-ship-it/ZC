# 15 — Kiểm tra đề bài, chia dữ liệu, huấn luyện, rò rỉ (13/11)

Trả lời câu hỏi của user 13/11: đã làm đúng đề bài, code, chia dữ liệu, huấn luyện đúng cách chưa; có rò rỉ dữ liệu không; có
tận dụng hết dữ liệu train không; huấn luyện trên những DN nào, suy luận trên tập nào. Mọi số dưới đây đo lại trên repo ngày 13/11
(lệnh ở cuối tài liệu).

## 1. Kết luận nhanh

| Câu hỏi | Kết luận | Mức độ |
|---|---|---|
| Ràng buộc đề bài (≤ 15B, offline, 6 h, tất định, không hard-code id, không train bằng test) | **đạt** (§2) | — |
| Rò rỉ dữ liệu test vào **trọng số** mô hình | **không có**: không bản ghi huấn luyện nào chứa id, DN hay đáp án public; không bản ghi nào của 6 DN held-out (§4.1) | — |
| Rò rỉ vào **phép đo** (số đo bị lạc quan) | **có, với hệ có luật P**: luật viết tay được rút ra từ lỗi public (dùng `public_gt.jsonl`), và các module luật được viết trước khi tách 6 DN held-out → public 0.994 và held-out 0.775 của P đều **lạc quan** so với private (§4.2) | cao |
| Tận dụng hết dữ liệu train | **chưa**: mô hình nộp chỉ học 2119 / 3000 câu (70.6 %): bỏ 6 DN held-out (631 câu) và 250 câu không có chương trình đã kiểm; mô hình nộp chưa huấn luyện lại trên đủ 30 DN (§5) | cao |
| Huấn luyện đúng cách | đúng quy trình (FSDP, loss chỉ trên đầu ra, 1 epoch, seed cố định), nhưng **nhãn sinh từ 55 mẫu chương trình của luật** → học sinh chỉ bắt chước được độ phủ của luật, loss 0.03 (học thuộc), phải trộn 0.35 với mô hình gốc (§6) | trung bình |
| Lỗi đo nhỏ | `tr_ho` chạy với rulebook **public**, thiếu 3 dòng có trong rulebook train; `pub_ho` có nhãn nhiễu (paraphrase đổi nghĩa) (§4.3) | thấp |

## 2. Đề bài và ràng buộc (`docs/00_TASK.md`)

| Ràng buộc | Trạng thái |
|---|---|
| Tổng tham số lúc suy luận ≤ 15B | Qwen3.5-9B (9.65B gồm vision + MTP), LoRA đã gộp vào trọng số (không thêm tham số) |
| Offline, không dịch vụ ngoài | container `--network none` (`docker/README_SOLUTION.md`) |
| ≤ 6 h trên 2×RTX 5090, ghi đáp án tăng dần | `--budget 20400 s`, `--hard-deadline`, ghi append + flush |
| Tất định ±0.03 | 3 seed public 0.993 / 0.994 / 0.993; container 0.994 |
| Không hard-code đáp án / logic theo id câu / id DN | không có id câu nào trong `src/`; id DN (`dev_0xx`, `biz_1xx`) chỉ xuất hiện trong **chú thích** ghi nguồn gốc một sửa lỗi; `retrieval.TR_HO` chỉ dùng để loại 6 DN held-out khỏi bộ ví dụ |
| Không dùng public / private test để train | đúng cho trọng số (§4.1). Lưu ý: luật viết tay (hints / qnorm / precheck / canon) được chỉnh theo lỗi trên public có đáp án (`data/raw/public_gt.jsonl`, chỉ để chấm) — không phải "train", nhưng là chỉnh theo test (§4.2). Kế hoạch mới (`docs/16`) bỏ hẳn việc này |
| Image không chứa đáp án | `.dockerignore` loại `data/`, `goldkey`, `handoff`, `final_submission`, `tests`, `training/data`; `public_gt.jsonl` không có trong git |

## 3. Chia dữ liệu — huấn luyện trên DN nào, suy luận trên tập nào

| Tập | DN | Câu | Dùng để |
|---|---|---|---|
| Train (BTC) | 30 (`biz_101…130`) | 3000 | nguồn duy nhất của dữ liệu huấn luyện |
| — dev (huấn luyện) | 24 | 2369 | D1 / D2 / D2b / D2N / RFT / bộ ví dụ truy hồi |
| — `tr_ho` (held-out) | 6 (`biz_105, 109, 114, 123, 127, 130`) | 631 câu gốc + 1195 paraphrase = 1826 | chỉ đo |
| Public test | 14 (`dev_001…014`), **không trùng DN nào với train** | 1000 | suy luận **toàn bộ** 1000 câu; chấm bằng `public_gt.jsonl` |
| — `pub_ho` | 14 | 1373 paraphrase (k = 1, 2) của 779 câu public | đo held-out |
| — `pub_noind` | 4 (`dev_011–014`) | 112 | mô phỏng ngành mới (ẩn hàm ngành) |

Trả lời trực tiếp: mô hình **không** huấn luyện trên toàn bộ 30 DN, cũng không chỉ vài DN: huấn luyện trên **24 DN train** (giữ 6 DN
để đo), rồi suy luận trên **toàn bộ 1000 câu của 14 DN public** — các DN này mô hình chưa từng thấy. 15 câu public trùng **chữ**
với câu train (câu chung chung như "doanh thu hôm qua"), nhưng khác DN nên khác đáp án — không phải rò rỉ.

## 4. Rò rỉ

### 4.1 Vào trọng số mô hình: không có

| Kiểm tra (13/11) | Kết quả |
|---|---|
| 13 file dữ liệu huấn luyện (`training/data/out/{wg,pure}/*.jsonl`: D1, D2, D2b, D2N, D2mix, RFT, bản `p` / `ph`) — id public, DN public | 0 / 0 |
| — bản ghi của 6 DN held-out | 0 |
| Bộ ví dụ truy hồi `src/refs/bank/examples.jsonl` (2119 ví dụ, 24 DN) | không có DN held-out, không có public |
| Chương trình đọc dữ liệu tham chiếu `src/refs/compile/` | 30 file, đều là DN train |
| Khi đo `tr_ho`, DN held-out có dùng chương trình đọc dữ liệu viết riêng cho chính nó không | không: cả 6 DN chọn `fmt:automap` (tổng quát) — `compile_logs.jsonl` của P |
| Dữ liệu tổng hợp D3 / RFT | chỉ 24 DN dev (943 câu D3 trong RFT) |

### 4.2 Vào phép đo: có (hệ P)

1. **Luật được chỉnh theo lỗi public.** Quy định cũ cho phép Claude phân tích lỗi trên public gốc, nên `qnorm.ABBR` có các mục
   như `"o đờ" → đơn`, `"xi ô đi" → COD`; `canon.py` ghi 22 chỗ sửa theo DN public (`dev_008`, `dev_014`…); `qparse.py`, `dm.py`,
   `industry.py` cũng có. Public 0.994 của P vì thế **không** ước lượng được private. `pub_ho` là paraphrase của **chính** các câu
   public đó (cùng DN, cùng ý) nên cũng lạc quan với P.
2. **Luật viết trước khi tách held-out.** `qparse.py`, `qnorm.py`, `answer_examples.jsonl` (55 mẫu chương trình) và
   `question_cleaning/` có từ 02/11 (commit `f6b5232`, `8bc0c44`), còn `tests/make_splits.py` tách 6 DN held-out ngày 08/11
   (`590f943`). Luật đã "thấy" câu của 6 DN này → `tr_ho` của P cũng lạc quan. Mô hình thuần thì không (trọng số chưa thấy `tr_ho`).
3. **Chọn cấu hình / α / mô hình theo public** — thiên lệch chọn nhẹ (bình thường, nhưng cộng thêm vào 1 và 2).
4. Hệ quả: private có **loại DN chưa xuất hiện** ở train lẫn public (`data/raw/data.txt`) — đúng chỗ luật viết tay yếu nhất. Đây là
   lý do nghiêm túc để ủng hộ ý tưởng của user: luật phải do một quy trình tự động sinh ra và kiểm trên train, không do người /
   Claude viết theo test.

### 4.3 Lỗi đo nhỏ

- `src/eval_flow.py` chép rulebook **public** cho mọi tập con (chú thích "superset of the training one" sai): rulebook public
  **bỏ** 3 dòng của rulebook train ("Giá nhập bình quân một mặt hàng trong tháng", "Tồn theo đơn vị lớn", "Giá bán X ngày D") →
  câu `tr_ho` dùng 3 định nghĩa này bị đo thiếu định nghĩa (bi quan; ước lượng thô theo từ khoá ≈ 50 / 631 câu có thể chạm, có thể đếm thừa). Sửa: mỗi DN dùng rulebook
  của split gốc của nó.
- `pub_ho`: một phần paraphrase đổi nghĩa (đổi kỳ / đại lượng) mà nhãn vẫn là đáp án câu gốc (`handoff/01_SCIENTIFIC_FINDINGS.md`)
  → trần dưới 1.0. Kế hoạch mới thêm `pub_ho_clean` (lọc paraphrase tự động bằng bộ kiểm tương đương, không ai xem từng câu).

## 5. Có tận dụng hết dữ liệu train không — chưa

| Phần | Số | Ghi chú |
|---|---|---|
| Câu train có chương trình đã kiểm (`answer_examples.jsonl`) | 2681 / 3000 | sinh từ 55 mẫu (`tpl`) của luật `qparse` |
| D1 (24 DN dev) | 2119 / 2369 | **250 câu dev không có chương trình** (number 165, percent 59, list 19, string 7) — mô hình nộp chưa từng thấy lời giải đúng của loại câu này; nhánh thuần lấy lại được 154 qua RFT |
| 6 DN held-out | 631 câu, 562 có chương trình | đúng thiết kế để đo, nhưng **mô hình nộp và bộ ví dụ chưa bao giờ dùng** (`bank_holdout` = 0 trong `flow_final.json`) |
| Dữ liệu giao dịch của 30 DN | 12–30 tháng / DN | sinh được vô hạn câu mới có đáp án (D3); mới dùng 943 câu |
| Paraphrase train | 5689 | D2 dùng 3818 (2037 câu gốc) |

Đề xuất ngay (rủi ro thấp, chuẩn trong học máy): sau khi chốt cấu hình, **huấn luyện lại trên đủ 30 DN** (cùng siêu tham số) và bật
`bank_holdout` cho bản nộp; chỉ giữ nếu public ≥ 0.993 (2 seed) và `pub_ho` không giảm (không còn `tr_ho` để đo).

## 6. Huấn luyện đúng cách chưa

Đúng quy trình: LoRA r 32 α 64 lr 2e-4, 1 epoch, batch 64, max-len 4608, loss chỉ trên đầu ra của trợ lý, FSDP 4 GPU, seed cố định,
không dữ liệu test. Vấn đề về **nội dung** nhãn:

- Nhãn D1 / D2 là chương trình của **55 mẫu luật** → mô hình học sinh chỉ học được những gì luật đã phủ; loss 0.03–0.06 (gần như học
  thuộc); dùng nguyên LoRA (× 1.0) kém hơn trộn × 0.35 với gốc → dữ liệu hẹp, mất khả năng tổng quát.
- Nhãn "tự gợi ý" của nhánh thuần (QS7, QS10) lấy từ `hints.py` (luật viết tay) → vẫn là chưng cất luật của Claude.
- **Lệch phong cách câu train ↔ test** (đo 13/11 trên chữ câu hỏi): kiểu viết kỳ "T4/2025", "Th12/25" có ở **25.9 %** câu public
  nhưng **0 %** câu train (D1 / D2 / D2b: 0 %, D2N: 7 %); năm 2 chữ số "12/25": 7.6 % vs 0 %. Mô hình thuần QS7 đạt PA **0.845** trên
  329 câu public có kiểu viết này so với 0.934 trên phần còn lại (P: 0.997 — vì `qparse` đã được viết cho đúng kiểu này).

## 7. Lệnh kiểm tra (tái lập)

- Phân bố DN / tập: đoạn Python đọc `data/raw/*/questions.jsonl` và `tests/splits/*.txt` (13/11, trong phiên).
- Rò rỉ dữ liệu huấn luyện: quét `training/data/out/{wg,pure}/*.jsonl` theo id / DN public và `TR_HO`.
- Kiểu câu: regex kỳ viết tắt trên câu train / public / dữ liệu huấn luyện.
- Lỗi mô hình thuần theo phần chương trình (kỳ / thực thể / hàm) so với P: `<scratchpad>/tools/cmp_pure_p.py`, `period_share.py`.
