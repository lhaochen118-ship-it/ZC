# PDF 3 — Handoff (khung chat kế tiếp: hybrid framework: Agent sinh code + luật hỗ trợ)

*Cập nhật 08/11/2026. Phạm vi: mục tiêu, tiến độ, quyết định đã thống nhất, vị trí mã nguồn / dữ liệu / kết quả, các bước
tiếp theo. Kết quả khoa học ở PDF 1, kinh nghiệm kỹ thuật / vận hành ở PDF 2, cấu hình máy ở `environment_summary.pdf`.*

## 1. Mục tiêu

- Zalo AI Challenge 2026, track **DataMate**: trả lời câu hỏi tiếng Việt trên dữ liệu bảng bẩn của hộ kinh doanh (mỗi DN
  một cấu trúc, rulebook nghiệp vụ). Điểm **PA = (đúng − sai) / N**, abstain = 0.
- Ràng buộc BTC: tổng tham số ≤ 15B, offline, ≤ 6 giờ trên 2×RTX 5090, Docker. **Hạn nộp: 16/11/2026 23:59 GMT+7**
  (user build + nộp portal).
- Mục tiêu user: PA cao nhất trên **private test** (DN và cách hỏi chưa thấy), nhanh nhất có thể.

## 2. Quyết định đã thống nhất (không hỏi lại)

### 2.1 (08/11, mới nhất) Hybrid framework: AI Agent sinh code + luật tổng quát hỗ trợ

**Ranh giới user đã làm rõ (08/11):**

| Được phép: hard code LOGIC | Bị cấm: hard code ANSWER |
|---|---|
| Quy tắc tổng quát nhận diện ý định câu hỏi (loại câu, kỳ thời gian, thực thể, đại lượng) → đưa cho Agent làm gợi ý | Luật sinh thẳng code Python / đáp án 100 % mà không qua mô hình |
| Đọc file data sẵn có, viết schema bằng code (`ingest`, `automap`); Agent viết thêm mô tả schema khi cần | Đáp án luật là đáp án cuối, mô hình không tham gia (như stage 1 hiện nay) |
| Toolkit nghiệp vụ đã kiểm (`dm`, `industry`) = rule engine để Agent gọi | Mẫu / nhánh theo id DN, id câu |
| Làm sạch câu (`qnorm`), ví dụ đã kiểm truy hồi (few-shot), kiểm tra hợp lệ bằng code | |

Nghĩa là: luật được dùng để **chuẩn bị ngữ cảnh** (schema đã ánh xạ, ý định, kỳ thời gian, thực thể, ví dụ) giúp Agent sinh
code **nhanh hơn hoặc chính xác hơn**; **code trả lời phải do mô hình sinh** cho mọi câu. Chương trình do luật dựng
(`qparse`) chỉ được đưa vào prompt như gợi ý / dùng để kiểm chéo, không tự thành đáp án.

Khung (user, 08/11):

| Thành phần | Thực hiện bởi |
|---|---|
| Hiểu câu hỏi tiếng Việt (đa dạng, viết tắt, mơ hồ) | LLM |
| Phân tích schema (thích nghi dữ liệu mới) | LLM + code |
| Diễn giải quy tắc nghiệp vụ (rulebook đổi giữa các split) | LLM + rule engine |
| Sinh SQL / Python | LLM |
| Thực hiện tính toán | SQL / Python (LLM không tự tính) |
| Kiểm tra tính hợp lệ | Code + LLM khi cần |
| Quyết định abstain | Logic + mô hình đánh giá độ tin cậy |

Luồng: câu hỏi → mô hình → code truy vấn → chạy → mô hình nhận kết quả → trả lời / abstain. Chấp nhận public giảm từ
1.000 xuống ~0.92–0.95 để tổng quát hơn.

### 2.2 Các quyết định khác

- Một mô hình duy nhất **Qwen3.5-9B** (9.65B, FP8 trên vLLM 0.32). Không train trên test.
- Không fix theo id DN / id câu; không đưa đáp án public (`data/raw/public_gt.jsonl` = `score_1.jsonl` của user,
  `goldkey/devset.py`) vào `src/` hay prompt; chỉ dùng để chấm.
- Không đẩy kernel khi chưa `PREFLIGHT OK`; dồn đủ 2 phiên GPU + 5 CPU của một acc rồi mới sang acc khác.
- Một thay đổi chỉ được giữ khi cải thiện **held-out** (paraphrase k=1,2 public, paraphrase train); chỉ phân tích lỗi trên
  k=0 public và train.

## 3. Tiến độ (đến 08/11)

| Hạng mục | Trạng thái |
|---|---|
| Đường LLM (`src/agent.py` + `src/pipeline.py` + `src/consensus.py`) | Có sẵn, đã chạy trên Kaggle: v28 = **0.921** public (921 / 69 / 10, 8 lần chạy + đồng thuận ánh xạ), train 0.9897 |
| Đường luật (`src/rules_first.py`: automap + qnorm + qparse) | Train 3000 / 3000, public 1000 / 1000 — **trong mẫu**; paraphrase train held-out chỉ ~0.67 (luật một mình) |
| Luồng nộp hiện tại (`src/predict_flow.py`) | Rule-first: luật → stage 1.5 (LLM viết lại câu, luật đọc lại) → stage 2 LLM. **Trái §2.1 (luật trả lời thẳng) → phải thay** |
| Docker (`docker/`) | Script sẵn; **chưa build** (container dev không có Docker / GPU) |
| File nộp public | `submissions/v30/` (luật, 1000 / 1000), `submissions/v28/` (LLM, 0.921) |

## 4. Vị trí

| Thứ | Ở đâu |
|---|---|
| Repo / branch | `lhaochen118-ship-it/ZC`, `claude/datamate-v3-0dny90` |
| Dữ liệu BTC | Kaggle `nhatlong1103/zc-2026` → `data/raw/train/train/`, `data/raw/public_test/public_test/` (không commit); đáp án public: `data/raw/public_gt.jsonl` (user gửi lại nếu mất: `score_1.jsonl`) |
| Bộ kiểm tổng quát hoá | Kaggle private datasets `<acc>/zc-para` (`para_pub.jsonl` 2065 câu, `para_train.jsonl` 5689 câu; mỗi acc L4 một bản), `<acc>/zc-pend`; tái tạo: `src/paraphrase_llm.py`, `src/perturb.py` |
| Trọng số | Kaggle Model `danbth/qwen3-5-9b/transformers/default/1`, sha256 kiểm với `src/model_hashes.json` |
| Wheels | `nhatlong1103/zc-assets` |
| Job chuẩn | `automation/jobs/v28.jobs` (đường LLM, train + 5 public), `v32–v34.jobs` (luồng nộp trên paraphrase) |
| Nhật ký số liệu | `EXPERIMENT_LOG.md` (entry 0–27; LLM: 0–19, luật: 20–27) |
| Trạng thái sống | `handoff/STATUS.md` |

## 5. Các bước tiếp theo (user đã duyệt 08/11)

**Chạy tự động** đến khi thực thi xong toàn bộ kế hoạch: không dừng, không hỏi user; hết quota thì tạm dừng phần GPU và tự chạy
tiếp khi reset; không chạy lại bước đã xong (bảng "Tiến độ WP" trong `handoff/STATUS.md`); tạo routine tự tiếp tục mỗi giờ ngay
đầu khung chat mới.

**Việc của khung chat kế tiếp: triển khai code đúng theo `docs/08_MASTER_PLAN.md`** — gói việc §8 theo thứ tự, đặc tả §5,
quy tắc quyết định §10, lịch / quota §9. Tóm tắt thứ tự:

| WP | Nội dung | Ngày |
|---|---|---|
| WP0 | chuẩn bị: token, dữ liệu, cache, `tests/splits/`, dataset wheels huấn luyện | 09/11 |
| WP1 | chạy thử GPU: bộ nhớ / tốc độ LoRA, gộp vs adapter, nạp thầy FP8, ngữ pháp → QĐ1–QĐ3, QĐ12 | 09/11 |
| WP2 (P0) | luồng hybrid v1: `api_spec`, `grammar`, `hints`, `tools` (find / period / check / help), `precheck`, `prompting`, `predict_flow` mới | 09–10/11 |
| WP3 | mốc E0, chọn mô hình chính (Qwen3.5-9B vs Gemma 4 12B-it), A/B prompt / ngữ pháp, `pub_noind`, `pub_noacc` | 10/11 |
| WP4 | nhà máy dữ liệu D1, D2, D2N, D3, D4T, DV | 09–11/11 |
| WP5 | dò thầy (Qwen3.6-27B / 35B-A3B, Gemma 4 31B, gpt-oss) | 11/11 |
| WP6 | nhánh fine-tune song song (vòng 1 dữ liệu, vòng 2 thầy) + gộp nhánh / multi-LoRA | 11–12/11 |
| WP7 | huấn luyện đầy đủ + tích hợp | 12–13/11 |
| WP8 (P1) | data card, đọc kết quả, tool còn lại, rulebook, bộ kiểm 0.8B, truy hồi embedding 0.6B, tin cậy + ngưỡng | 12–13/11 |
| WP9 | vòng sửa lỗi tới public = 1.0 (chỉ cơ chế chung) | 10–14/11 |
| WP10 | đánh giá cuối (quota mới 14/11) → `configs/flow_final.json` | 14/11 |
| WP11 | Docker + `submissions/vNN`; user build ngày 15/11, nộp trước 16/11 23:59 GMT+7 | 15–16/11 |
| WP12 (P1) | nhánh thoát DuckDB SQL | 11–13/11 |

Thứ tự cắt khi thiếu thời gian và những gói không được cắt: `docs/08_MASTER_PLAN.md` §8 (cuối). Giờ GPU, số acc (cần thêm 4 acc
L4 mới, lý tưởng 5) và phân bổ theo đợt: §9; bố trí GPU / batch: §5.12.

## 6. Rủi ro cần nhớ

- Thời gian: LLM cho mọi câu trên 2×5090 chưa đo. 4×L4: ~55–75 phút / lần chạy public (1000 câu). Private có thể nhiều
  câu hơn → giữ cơ chế ngân sách thời gian + gộp kết quả mỗi 20 s (`merge_sub.py`).
- Quota: 30 giờ GPU / tuần / acc, tính theo GPU (một phiên 4×L4 = 4 h quota / giờ). Tuần W46 bắt đầu 07/11; yuan1103 đã
  dùng ~15 h.
- Ánh xạ LLM không tất định (80–89 % câu giống nhau giữa các lần chạy) → cần đồng thuận ánh xạ + nhiều lần chạy.
