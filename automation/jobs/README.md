# automation/jobs — job chuẩn + công cụ

- `v28.jobs`: đường LLM chuẩn (train + 5 public, seed khác nhau; jobs cách nhau `;`) — điểm xuất phát cho LLM-first.
- `v30*.jobs`, `v31–v34.jobs`: luồng nộp rule-first trên câu nhiễu / paraphrase (`stress_flow.py --modes …`); `v33.jobs`:
  stage 1.5 (`canon_llm.py`) riêng.
- `prepreflight.py "<jobs>"`: chạy `automation/preflight.py`, ghi mark cache để `fanout.py` khởi chạy ngay.
- `mksub5.sh vNN [vCONS]` (env `S` = scratch có `runs/`, `cache_pub/`): bỏ phiếu 5 lần chạy public của vNN + (tuỳ chọn)
  consensus → `submissions/`.
