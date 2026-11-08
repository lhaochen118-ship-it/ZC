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

## 5. Các bước tiếp theo (theo thứ tự)

Kế hoạch đầy đủ và lộ trình theo ngày (08–16/11), phân bổ quota, nhánh fine-tune song song: **`docs/08_MASTER_PLAN.md`**
(thay cho phần lộ trình dưới đây khi khác nhau).

**Bước 1.** **Xin token Kaggle** (acc L4: quynhtran35, yuan1103, phatle1578; T4: yeshenhi, mxuanquy) → `~/.kaggle/tok_<acc>` (600).
Tải dữ liệu, ingest cache (lệnh: `environment_summary.pdf` §1).

**Bước 2.** **Đo nền trên held-out trước khi sửa** (chưa từng đo): `agent.py` với `--prefer-map fmt:automap`, `--qw 0`,
không `--confirm` (mô hình sinh code cho mọi câu, schema từ automap) trên `para_pub.jsonl` + public gốc + train; ghi PA,
thời gian / câu, tỉ lệ abstain. So với luồng hiện tại (held-out public 0.602) và v28 (public 0.921).

**Bước 3.** **Viết `src/predict_flow.py` mới (hybrid)** theo bảng §2.1; thiết kế chi tiết, tham khảo paper và kế hoạch
thí nghiệm E0–E5: `docs/06_HYBRID_FRAMEWORK.md`. Tóm tắt:

- Ánh xạ schema (được phép dùng luật, §2.1): `automap` (code) dựng schema chuẩn trước, kiểm bằng `canon.secondary_checks`;
  LLM viết ánh xạ / mô tả schema khi automap cảnh báo nặng hoặc thiếu; đồng thuận ánh xạ (`consensus.py`).
- Hiểu câu (luật hỗ trợ + LLM): `qnorm` làm sạch; trích ý định / kỳ thời gian / thực thể bằng luật tổng quát (tách từ
  `qparse` thành phần "hint", không sinh đáp án) → đưa vào prompt cùng câu gốc.
- Sinh code (LLM, bắt buộc): Agent viết chương trình trên toolkit `dm` / `industry` cho **mọi** câu; few-shot truy hồi
  các cặp (câu train, chương trình đã kiểm) (`src/fewshot.py`); chương trình `qparse` (nếu có) chỉ là gợi ý trong prompt.
  Bỏ đường stage 1 trả lời thẳng và `--confirm` dựa trên luật; luật không được là phiếu tự quyết (`--qw` chỉ dùng để kiểm
  chéo, không vượt phiếu mô hình — đo cả hai biến thể).
- Rulebook: đưa rulebook của split vào prompt (đã có); toolkit là rule engine cho định nghĩa đã kiểm.
- Kiểm tra: bỏ phiếu K chương trình + kiểm kiểu đáp án + kiểm grounding (tên / mã trong câu); câu bất đồng → LLM
  xem lại kết quả (bước "mô hình nhận kết quả").
- Abstain: tỉ lệ phiếu < ngưỡng (0.75 hiện tại) + điểm tin cậy; chọn ngưỡng trên k=0, kiểm trên held-out.

**Bước 4.** Cập nhật tests (`tests/test_rules_first.py`, `test_canon_flow.py` gắn với luồng cũ), preflight, chạy Kaggle:
public gốc + paraphrase public + train. Giữ bản mới khi held-out ≥ luồng cũ và thời gian ước lượng ≤ 6 h trên 2×5090.

**Bước 5.** Tạo file nộp mới (`automation/make_submission.py` → `submissions/vNN/`), push, cập nhật `docker/predict.sh`,
`docker/README_SOLUTION.md`.

**Bước 6.** **User**: build Docker (`docker/SETUP_DOCKER.md`), chạy public, gửi dòng cuối `run.log` + thời gian; nộp trước 16/11.

## 6. Rủi ro cần nhớ

- Thời gian: LLM cho mọi câu trên 2×5090 chưa đo. 4×L4: ~55–75 phút / lần chạy public (1000 câu). Private có thể nhiều
  câu hơn → giữ cơ chế ngân sách thời gian + gộp kết quả mỗi 20 s (`merge_sub.py`).
- Quota: 30 giờ GPU / tuần / acc, tính theo GPU (một phiên 4×L4 = 4 h quota / giờ). Tuần W46 bắt đầu 07/11; yuan1103 đã
  dùng ~15 h.
- Ánh xạ LLM không tất định (80–89 % câu giống nhau giữa các lần chạy) → cần đồng thuận ánh xạ + nhiều lần chạy.
