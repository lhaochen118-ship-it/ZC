# Nhiệm vụ tiếp theo sau khi xong kế hoạch 1 (user, 09/11/2026 — nguyên văn yêu cầu, không tự ý thay đổi)

## 0. Bối cảnh và điều kiện dừng
- Kế hoạch 1 = docs/08_MASTER_PLAN.md (WP0 → WP12). Phải chạy xong toàn bộ trước, theo đúng chế độ chạy tự động
  trong CLAUDE.md (routine tự đánh thức mỗi giờ, cập nhật bảng "Tiến độ WP" trong handoff/STATUS.md sau mỗi bước).
- **Điều kiện dừng cuối cùng:** framework đạt **100 % accuracy trên public test**, chấm bằng file đáp án
  `score 1.jsonl` user đã gửi (trong repo: `data/raw/public_gt.jsonl`, chỉ dùng để chấm).
  Chưa đạt thì KHÔNG dừng: lặp lại vòng Phân tích → Kế hoạch → Triển khai → Đo (mục 2–5).
- Ràng buộc giữ nguyên: tổng tham số ≤ 15B, offline, ≤ 6 giờ trên 2×RTX 5090; code trả lời do mô hình sinh
  (cấm hard code đáp án); được hard code logic tổng quát; không vá theo từng câu / DN của test; không dùng
  public / private test để train.
- **Lưu ý chống overfit:** 100 % public phải đạt bằng cải tiến tổng quát. Mỗi cải tiến phải đồng thời
  không làm giảm held-out (paraphrase public k=1,2, paraphrase train, 6 DN train giữ ngoài). Cải tiến chỉ tăng
  public mà giảm held-out thì loại. Báo cáo cả hai số ở mọi mốc.

## 1. Bước 1: Hoàn tất kế hoạch 1
- Làm tiếp WP chưa xong đầu tiên trong STATUS.md, đến WP12. Ghi kết quả từng WP (public gốc, held-out,
  thời gian chạy, số liệu GPU từ gpu_summary.json).
- Xong WP12: chấm với `score 1.jsonl`, liệt kê từng câu sai (id, câu hỏi, đáp án đúng, đáp án mô hình,
  code sinh ra, log). Nếu đã 100 % và held-out không giảm thì báo user và dừng.

## 2. Bước 2: Tự phân tích nguyên nhân lỗi
Phân loại MỌI câu sai trên public gốc / train (không xem từng câu held-out) theo taxonomy, kèm số lượng và tỉ lệ:
1. Hiểu sai ý định câu hỏi (intent, phép tổng hợp, điều kiện lọc, khoảng thời gian, đơn vị).
2. Schema linking sai (chọn sai bảng / cột, join sai, nhầm tên cột tiếng Việt).
3. Code đúng ý nhưng sai cú pháp / lỗi runtime / hết giờ.
4. Sai logic nghiệp vụ (công thức ngành, toolkit `dm` / `industry` thiếu hoặc sai).
5. Định dạng đáp án (làm tròn, kiểu dữ liệu, thứ tự, null) / lỗi chấm.
6. Dữ liệu bẩn / ingest sai.
7. Chọn sai giữa nhiều ứng viên (self-consistency, điểm tin cậy kém).
Với mỗi nhóm: tìm nguyên nhân gốc ở mức mô hình (năng lực model nhỏ, prompt, ngữ cảnh quá dài, thiếu ví dụ,
decoding) hay ở mức hệ thống (thiếu tool, thiếu kiểm tra, thiếu sinh lại). Lưu kết quả vào `docs/10_ERROR_ANALYSIS.md`.

## 3. Bước 3: Nghiên cứu tài liệu ngoài
Đọc paper / báo cáo (không đọc code repo người khác) về Text-to-SQL / Text-to-Code trên bảng, ưu tiên nguồn
nhiều citation / stars hoặc giải cao: BIRD, Spider 2.0 leaderboard (CHESS, CHASE-SQL, XiYan-SQL, OmniSQL,
DIN-SQL, MAC-SQL, DAIL-SQL, Distillery, SQL-R1 / Reasoning-SQL (RL), Alpha-SQL…), các bài về schema linking,
candidate generation + selection, self-correction, execution-guided decoding, fine-tune / distillation /
RL (GRPO) cho model ≤ 15B, synthetic data (SynSQL…). Với mỗi kỹ thuật ghi: ý tưởng, mức tăng báo cáo,
chi phí, áp dụng vào nhóm lỗi nào ở mục 2. Ghi vào `docs/11_LITERATURE.md`.

## 4. Bước 4: Viết kế hoạch 2 → `docs/12_PLAN2.md`
Kết hợp phân tích lỗi (mục 2) với tài liệu (mục 3). Kế hoạch phải có:
- Ma trận "nhóm lỗi → giải pháp → mức tăng kỳ vọng → chi phí GPU / giờ → rủi ro".
- Các hướng cần cân nhắc: thêm module / tool cho Agent (schema linker, value retriever, unit tester,
  verifier); sinh nhiều ứng viên + bộ chọn (selector / reward model); fine-tune (SFT, LoRA) và RL theo
  execution reward; distillation từ teacher mạnh hơn (sinh dữ liệu từ train + dataset public uy tín,
  chỉ giữ mẫu được kiểm bằng thực thi).
- **Thí nghiệm song song để chọn nhanh:** khi hệ thống đã cao, fine-tune / distil nhiều biến thể cùng lúc
  (khác student model, khác teacher, khác dataset, khác hỗn hợp dữ liệu), mỗi biến thể một phiên GPU,
  đánh giá chung một bộ held-out, giữ biến thể tốt nhất. Ghi rõ lưới thí nghiệm và tiêu chí loại sớm.
- Gói việc đánh số (WP-P2-xx) theo thứ tự, đặc tả triển khai, quy tắc quyết định giữ / bỏ, lộ trình thời gian.
- Ưu tiên tối thiểu thời gian ra kết quả: việc nào rẻ và tăng nhiều thì làm trước; việc GPU dài thì chạy
  nền song song với việc CPU.
Commit + push file kế hoạch, cập nhật CLAUDE.md và STATUS.md để trỏ sang kế hoạch 2 (thêm bảng "Tiến độ P2").

## 5. Bước 5: Triển khai kế hoạch 2
Chạy tự động như kế hoạch 1, không hỏi user. Sau mỗi gói việc: chấm public (`score 1.jsonl`) và held-out.
Nếu chưa 100 % public thì quay lại mục 2 với lỗi còn lại, viết bổ sung kế hoạch (docs/12_PLAN2.md mục
"Vòng n") và tiếp tục.

## 6. Quy tắc dùng tài nguyên Kaggle (không cần tiết kiệm, cần nhanh nhất)
- Mỗi acc chạy đồng thời tối đa **2 phiên GPU + 5 phiên CPU**. Dùng hết slot của một acc rồi mới sang acc
  khác. Phần không cần GPU (eval, sinh dữ liệu bằng luật, kiểm tra) đưa lên phiên CPU.
- Acc có **4×L4** (4 GPU 24 GB độc lập, không NVLink): mô hình vừa 1 GPU thì data parallel `--dp 4`;
  train thì DDP / mỗi GPU một biến thể; batch size / `--max-num-seqs` lớn nhất mà VRAM cho phép
  (`--gpu-util 0.92`, đọc KV cache thật).
- Acc chỉ có **2×T4** (2×16 GB, không bf16 / FP8): `--dp 2`, lượng tử hoá AWQ / fp16, batch lớn nhất vừa VRAM.
- Theo đúng Playbook tốc độ GPU và Cổng kiểm tra trước khi chạy trong CLAUDE.md (preflight OK, watcher nền,
  không poll, QUEUED > 8 phút thì huỷ và đẩy lại).
- Chỉ xin user những thứ chỉ user làm được: token / acc Kaggle mới, accept rules, nộp portal, file `score 1.jsonl`
  nếu không có.

## 7. Báo cáo
Mỗi mốc lớn, ghi vào handoff/STATUS.md: public %, held-out %, thời gian chạy, câu còn sai theo nhóm, bước tiếp theo.
