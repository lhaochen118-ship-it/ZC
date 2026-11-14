# PDF 3 — Handoff (DataMate, 14/11/2026)

*Mục tiêu, quyết định, trạng thái, vị trí, việc tiếp theo. Phát hiện khoa học: PDF 1; kinh nghiệm kỹ thuật / vận hành: PDF 2; trạng thái sống:
`handoff/STATUS.md`.*

## 1. Bài toán và ràng buộc

Zalo AI Challenge 2026, DataMate: hỏi đáp tiếng Việt trên dữ liệu bảng bẩn của cửa hàng (CSV / Excel / SQLite, rulebook nghiệp vụ). Điểm
**PA = (đúng − sai) / N**, bỏ trả lời = 0. Mô hình ≤ 15B tham số, offline, ≤ 6 h trên 2×RTX 5090, Docker. Train 30 DN / 3000 câu, public 14 DN /
1000 câu (không trùng DN), private có loại DN mới. Hạn: container ~15/11 12:00 UTC, nộp **16/11 23:59 GMT+7**. Đã có đội đạt 96 % (14/11).

## 2. Quyết định của user còn hiệu lực

- Cấm hard code đáp án; không train trên public / private test; phân tích lỗi chỉ trên public gốc / train.
- 13/11 tối: **ngừng luật viết tay** (luật nếu có phải do mô hình viết); **bản nộp = hệ không luật viết tay**, kể cả thấp hơn P.
- 14/11 02:30: **chỉ tối ưu public** (không held-out); xong kế hoạch 4 thì dừng, viết PDF 1 / 2, dọn repo; user đổi mô hình để nghĩ chiến lược
  mới hướng tới **public = 1**. GPU: dùng hết 4 L4 / phiên (FSDP, hoặc mỗi GPU một biến thể), không tiết kiệm acc.

## 3. Trạng thái (14/11 ~05:00 UTC)

| Hệ | Cấu hình | Public gốc |
|---|---|---|
| **Hệ nộp (S–W, không luật viết tay)** | `configs/flow_final.json`: Qwen3.5-9B + v3a × 0.70 (gộp), W viết hàm luật từ rulebook, k_max 16, bỏ phiếu thường; adapter `nhatlong1103/zc-final-adapter` (bản mới nhất), sha256 `src/adapter_hashes.json` | **0.938** |
| P (luật viết tay hỗ trợ; dự phòng) | `configs/flow_p_final.json` + WG1 × 0.35 (bản cũ của dataset, `src/adapter_hashes_p.json`) | 0.994 |

Lỗi còn lại của hệ nộp (39 câu không đúng): 17 có mẫu đúng nhưng bỏ / chọn sai, 22 không mẫu nào đúng (11 ở DN ngành). Các nhánh đã thử
không hơn v3a trên public: v4 (dùng hàm W + RFT), v3b (thẻ luật trong prompt), soup, bộ chọn học được V (lợi ở held-out, không lợi public).
W (hàm do mô hình viết) có ích có ý nghĩa ở ngành mới (+0.12 / +0.08 trên 614 câu khi ẩn hàm ngành viết tay).

## 4. Vị trí

| Thứ | Ở đâu |
|---|---|
| Mã | `src/` (luồng), `training/` (huấn luyện / gộp / đánh giá), `automation/` (Kaggle), `tests/` (`run_all.sh`), `docker/` |
| Điểm mọi lần chạy | `Process/scores.csv` (kế hoạch 1–2), `Process/scores_p3.csv` (3–4) |
| Log đã lưu | `Process/runs/` (P, kế hoạch 3–4); dữ liệu huấn luyện `Process/train_data/` |
| Kernel / adapter | Kaggle (sổ `Process/kaggle/kernels.csv`); adapter v3a / v4 / v3b: `<acc>/zc-{v3a,v4s1,v4s2,v3b}-adapter` |
| Kế hoạch 4 (nhật ký) | `docs/17_PLAN4_SWV.md` §7 |

## 5. Việc tiếp theo

1. User duyệt chiến lược mới (public = 1) → lên kế hoạch chi tiết (chia luồng, batch, thời gian) → chạy.
2. Trước hạn: user build Docker (`docker/SETUP_DOCKER.md`: `fetch_assets.sh` → `build.sh` → `run_public_test.sh`), nộp portal.
3. Muốn quay về P: `CONFIG=/app/configs/flow_p_final.json` và adapter WG1 × 0.35 (`automation/set_final_model.py --members WG1 --weights 0.35 …
   --publish`, hoặc dùng bản cũ của dataset + `src/adapter_hashes_p.json`).
