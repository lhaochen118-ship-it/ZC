# PDF 3 — Handoff (khung chat kế tiếp: chuyển sang kiến trúc LLM-first)

*Cập nhật 08/11/2026. Phạm vi: mục tiêu, tiến độ, quyết định đã thống nhất, vị trí mã nguồn / dữ liệu / kết quả, các bước
tiếp theo. Kết quả khoa học ở PDF 1, kinh nghiệm kỹ thuật / vận hành ở PDF 2, cấu hình máy ở `environment_summary.pdf`.*

## 1. Mục tiêu

- Zalo AI Challenge 2026, track **DataMate**: trả lời câu hỏi tiếng Việt trên dữ liệu bảng bẩn của hộ kinh doanh (mỗi DN
  một cấu trúc, rulebook nghiệp vụ). Điểm **PA = (đúng − sai) / N**, abstain = 0.
- Ràng buộc BTC: tổng tham số ≤ 15B, offline, ≤ 6 giờ trên 2×RTX 5090, Docker. **Hạn nộp: 16/11/2026 23:59 GMT+7**
  (user build + nộp portal).
- Mục tiêu user: PA cao nhất trên **private test** (DN và cách hỏi chưa thấy), nhanh nhất có thể.

## 2. Quyết định đã thống nhất (không hỏi lại)

### 2.1 (08/11, mới nhất) Chuyển sang LLM-first

User cấm luật if / else viết sẵn sinh code mà không dùng mô hình (chỉ khớp public, mù trên private). Khung bắt buộc:

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
| Luồng nộp hiện tại (`src/predict_flow.py`) | Rule-first: luật → stage 1.5 (LLM viết lại câu, luật đọc lại) → stage 2 LLM. **Trái §2.1 → phải thay** |
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

**Bước 1.** **Xin token Kaggle** (acc L4: quynhtran35, yuan1103, phatle1578; T4: yeshenhi, mxuanquy) → `~/.kaggle/tok_<acc>` (600).
Tải dữ liệu, ingest cache (lệnh: `environment_summary.pdf` §1).

**Bước 2.** **Đo nền LLM-only trên held-out trước khi sửa** (chưa từng đo): đường LLM thuần (`agent.py`, `--qw 0`, không
`--prefer-map fmt:automap`) trên `para_pub.jsonl` + public gốc + train; ghi PA, thời gian / câu, tỉ lệ abstain.
So với luồng hiện tại (held-out public 0.602).

**Bước 3.** **Viết `src/predict_flow.py` mới (LLM-first)** theo bảng §2.1:

- Ánh xạ schema: LLM viết chương trình ánh xạ (đã có), code kiểm (`canon.secondary_checks`), đồng thuận ánh xạ
  (`consensus.py`). `automap` (luật) chỉ còn là một ứng viên được kiểm như mọi ứng viên — **hỏi user** có được giữ
  không; mặc định: tắt.
- Hiểu câu + sinh code: LLM viết chương trình trên toolkit `dm` / `industry` (đã có). `qparse` không sinh đáp án nữa;
  các cặp (câu train, chương trình đã kiểm) dùng làm **few-shot truy hồi** (`src/fewshot.py`). `qnorm` (làm sạch chính
  tả) chỉ được đưa làm gợi ý trong prompt, câu gốc là chuẩn.
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
