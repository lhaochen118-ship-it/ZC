# PDF 1 — Scientific Findings (DataMate, đến 11/11/2026)

*Phạm vi: đóng góp, ý nghĩa khoa học, kết quả tích cực và tiêu cực, bằng chứng, điều kiện kiểm chứng. Kinh nghiệm kỹ thuật /
vận hành ở PDF 2; tiến độ, vị trí file và việc tiếp theo ở PDF 3. Số liệu: `EXPERIMENT_LOG.md` (kế hoạch 1),
`Process/history/STATUS_full_until_2026-11-11.md` (từng vòng kế hoạch 2), `Process/scores.csv` (mọi lần chạy).*

## 1. Bài toán và thước đo

- Hỏi đáp tiếng Việt trên dữ liệu bảng bẩn (CSV / Excel / SQLite), mỗi DN một cấu trúc, rulebook nghiệp vụ.
  **PA = (đúng − sai) / N**, bỏ trả lời = 0. Train 30 DN / 3,000 câu; public 14 DN / 1,000 câu; private: DN và cách hỏi mới.
- Thước đo quyết định (giữ suốt kế hoạch 2): **public gốc** (dev, được phân tích lỗi) và **held-out** — `pub_ho` (1,373
  paraphrase câu public, k = 1, 2), `tr_ho` (1,826 câu của 6 DN train giữ ngoài huấn luyện); gộp = trung bình theo số câu.
  Nhiễu một lần chạy: public ±0.4, held-out gộp ±0.3 → mọi quyết định dùng ≥ 2 seed.
- Quy tắc giữ / bỏ (`docs/12_PLAN2.md` §4): giữ khi public tăng ≥ 0.3 (một lần hoặc trung bình) và held-out gộp không giảm > 0.3;
  sửa lỗi hiển nhiên được giữ nếu không kém đi; cải tiến tăng public mà giảm held-out → loại.

## 2. Kiến trúc cuối: hybrid framework (Agent sinh code + luật hỗ trợ)

Mọi câu: gợi ý tổng quát (kỳ, thực thể, kiểu đáp án, từ rulebook, câu đã làm sạch) + ví dụ truy hồi → **mô hình viết chương
trình** trên toolkit nghiệp vụ đã kiểm (`dm`, `industry`) → kiểm tra trước khi chạy (precheck: lỗi → lượt sửa) → chạy sandbox →
cờ sau khi chạy → vòng 2 (thêm mẫu, nửa "think") → **bỏ phiếu**, bỏ trả lời khi phiếu < 0.5. Không luật nào tự viết đáp án.

| Mốc | Public gốc | Held-out gộp |
|---|---|---|
| Kế hoạch 1, M0 (súp WF1–4 × 0.35, không luật P2) | 0.934 (tb 5 lần) | 0.752 |
| Hướng R vòng 8 (+ luật tổng quát P2, súp cũ) | 0.9815 | 0.761 |
| Bản nộp 11/11: O + WG1 × 0.35 | **0.9855** (container 0.986) | **0.778** |

## 3. Kết quả tích cực

### 3.1 Hướng R — luật tổng quát làm ngữ cảnh / kiểm tra (16 vòng, 0.934 → 0.9815 với cùng mô hình)

- **Sửa gợi ý / làm sạch câu là đòn bẩy lớn nhất** (vòng 1: +3.1 public): lỗi "máy / mấy" ở 27 câu, từ rulebook khớp có dấu, mẫu
  "N ngày tới", "the" → thế / thẻ khi câu không dấu, khôi phục dấu từng từ có bigram hỗ trợ.
- **Trần token 512 → 1536** (10.7 % mẫu bị cắt ở 512): +0.2 public, held-out không giảm (với luồng R).
- **Kiểm tra lỗi có "tín hiệu sạch"** (chỉ chạm mẫu sai trên mọi log): `size` (54 + 114 mẫu, 0 đúng), `unit` (71 + 104), `len_mask`
  (`len()` của mặt nạ True / False; 69 mẫu cuối trên 20 lần chạy, 0 đúng). Áp dụng như lỗi → lượt sửa. Phương pháp chọn kiểm tra này
  (đo trên log có sẵn, chỉ nhận khi không chạm mẫu đúng) là đóng góp có thể tái dùng.
- Cờ cảnh báo (period, named-periods, threshold) giảm trọng số phiếu thay vì chặn: +0.2–0.5.

### 3.2 Hướng M — cải tiến phía mô hình (không thêm luật; từ M0 = 0.934 / 0.752)

| Gói | Ý tưởng | Kết quả | Quyết định |
|---|---|---|---|
| **M7** | mỗi thành viên súp chạy như bộ sinh riêng | WF3 (D1 + mẫu D2) × 0.35: **0.951 / 0.759**; WF1 / WF2 (chỉ D1): 0.924 / 0.931 | **thành viên có paraphrase > súp 4 thành viên** |
| M6 | bộ chọn đáp án học được (logistic, 13 đặc trưng tỉ lệ bất biến số mẫu; train trên log câu train / paraphrase / D3) | M0: public 0.934 → **0.944**, held-out 0.752 → **0.767** | giữ cho M; trên luồng R: public −0.1…−0.8 → không dùng |
| M2 | RFT / STaR: SFT trên chương trình đúng của chính mô hình (710 câu khó + câu dễ), LoRA **học trên** súp | M0: 0.939 / 0.768 (+0.5 / +1.6); trên WF3 + R: +0.05 / +0.3; trên WG1: −0.6 public | giúp mô hình yếu, không giúp mô hình đã mạnh |
| M4 | DPO theo phản hồi thực thi (871 cặp đúng / sai của chính súp) | 0.935 / 0.749 | loại (khớp cặp, margin 9.2, không tổng quát) |
| — | trần 1536 trên luồng M0 | public +0.55, held-out −1.0 | loại (khác hướng R) |

### 3.3 Mô hình nộp: dữ liệu paraphrase là đòn bẩy (vòng 10–16, cùng luồng R)

| Mô hình × 0.35 (luồng J / O) | Public tb | Held-out gộp tb | Seed |
|---|---|---|---|
| súp WF1–4 (cũ) | 0.9815 | 0.761 | 2 |
| WF3 (D1 + 2,119 paraphrase) | 0.979 | 0.773 | 3 |
| WF3 + RFT trên WF3 | 0.9805 | 0.776 | 2 |
| súp WF3 + WF4 | 0.977 | 0.769 | 3 |
| **WG1 (D1 + 3,818 paraphrase)** | **0.985** (J), **0.9855** (O) | 0.776 / 0.778 | 3 / 2 |
| WG2 (D1 + D2 + D2N) | 0.979 | **0.782** | 1 |
| WG3 (= WG1, seed khác) | 0.983 | 0.768 | 1 |
| WG1 × 0.40 | 0.986 | 0.780 | 2 |
| WG1 × 0.40 + k 12 | 0.9825 | 0.784 | 2 |

Kết luận: (1) nhiều paraphrase hơn → tổng quát tốt hơn ở cả public lẫn held-out; (2) **súp không hơn thành viên tốt nhất** (4 lần
thử); (3) chất lượng phụ thuộc seed huấn luyện (WG1 vs WG3: 0.8 điểm public cùng dữ liệu); (4) α 0.35–0.40 ổn định; (5) trộn câu
RFT vào SFT từ mô hình gốc (WG4) làm held-out giảm (0.754), RFT chỉ có ích khi học chồng lên mô hình đã fine-tune và còn yếu.

## 4. Kết quả tiêu cực và hạn chế

1. **Chưa đạt điều kiện dừng 100 %**: còn 3 câu sai ổn định (chọn sai hàm: bỉm size M theo bịch, bình quân trên phiếu thu, nhập về
   N viên) + ~10 câu phiếu tách đổi theo seed → public trần thực tế ~0.985–0.99 với hệ hiện tại.
2. **Held-out paraphrase public còn thấp (0.64)**: một phần là paraphrase đổi nghĩa (đổi kỳ / đại lượng) nên số này dưới-ước lượng;
   tr_ho (DN chưa thấy, câu gốc) 0.88 là ước lượng gần private nhất.
3. **Bộ chọn học được không chuyển được** giữa luồng / mô hình (trên luồng R public giảm mọi lần) → cần train lại trên log của đúng
   cấu hình, và cần log câu train ngoài mẫu (D3) để không bỏ trả lời quá tay.
4. **Sàng lọc trên tập con 400 câu nhiễu ±0.6**: ×0.40 và k12 hơn +1.2 trên tập con nhưng chỉ +0.2 / +0.6 trên đủ tập → tập con
   chỉ dùng để loại.
5. Điểm 1.000 của đường luật kế hoạch 1 là trong mẫu (paraphrase chỉ 0.51–0.67) → lý do chuyển sang hybrid.

### Đã thử và không giữ (kế hoạch 2)

prefill `<plan>` (0.898), θ 0.375 (public +, pub_ho −), review 3 / k 12 (trung tính, chậm hơn), cờ raw-pandas / base-unit / hàm nhập
(không sạch), DPO, RFT trên WG1, súp WF3 + WF4 / WG1 + WG3 / WG1 + WG2, WG1 × 0.30, bộ chọn trên luồng R.

## 5. Ý nghĩa khoa học

- Với mô hình ≤ 10B, **ngữ nghĩa trong code kiểm chứng + LLM viết chương trình ngắn** đạt 0.986 public / 0.88 trên DN chưa thấy;
  luật không trả lời thẳng mà nâng chất lượng ngữ cảnh (gợi ý, làm sạch) và kiểm tra lỗi → +5 điểm.
- **Đa dạng cách hỏi trong dữ liệu fine-tune** (paraphrase do chính mô hình viết, kiểm giữ kỳ) quan trọng hơn kỹ thuật huấn luyện
  phức tạp (DPO, RFT, súp) cho tổng quát hoá cách hỏi mới.
- **Chọn kiểm tra bằng "tín hiệu sạch" trên log** cho phép thêm luật tổng quát mà không làm hại held-out.
- Đánh giá có nhiễu lớn giữa các lần chạy (±0.4 public) → mọi kết luận cần ≥ 2 seed và thước đo held-out cố định.

## 6. Điều kiện kiểm chứng

| Kết luận | Cách kiểm |
|---|---|
| Điểm mọi lần chạy | `Process/scores.csv` (tính lại: script ở PDF 2 §4 hoặc `tests/pool_gens.py <log dirs>`) |
| Bản nộp O + WG1 | `Process/runs/p2r15o*`, `p2r16pred` (logs + submission nén); chấm `python3 src/score.py data/raw/public_gt.jsonl <submission>` |
| Phát lại bỏ phiếu / bộ chọn không GPU | `python3 tests/pool_gens.py <run>[,<run>] --selector src/refs/selector/selector_m0_inv.json` |
| Tín hiệu sạch của một kiểm tra | quét log `Process/runs/*` như PDF 2 §4 (đếm mẫu bị chạm theo đúng / sai) |
| Huấn luyện lại WG1 | `automation/jobs/wg.jobs` (dữ liệu `Process/train_data/wg/*.gz`), ~5 h trên 4×L4 |
| Toàn bộ test CPU | `bash tests/run_all.sh` → `ALL PASS` |
