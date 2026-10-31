# Log các run của khung chat 2 (train / public), nén gzip

`<run>__<kernel>/logs.jsonl.gz`: mỗi dòng một câu: id, câu hỏi, DN, final, conf, votes, **samples** (mọi lần sinh: code, lỗi,
đáp án, stdout). `compile_logs.jsonl.gz`: các ứng viên compile theo DN. `submission.jsonl.gz`, `score.json.gz`.
Cấu hình từng run: `EXPERIMENT_LOG.md` entry 3–5 và `handoff/03_HANDOFF.md`.

Dùng lại (không cần GPU):

```bash
python3 src/compare.py data/raw/train/train/answers.jsonl e8=<(zcat results/chat2_runs/e8__zc-e8-1/logs.jsonl.gz)
python3 src/build_shots.py data/raw/train/train out.jsonl <(zcat results/chat2_runs/e9__zc-e9-1/logs.jsonl.gz) ...
```
