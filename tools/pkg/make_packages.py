#!/usr/bin/env python3
"""Build the three self-contained solution folders Docker_C4 / Docker_C6 / Docker_PG from the repository."""
import json, os, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKIP = {"run_kaggle.py", "stage_hf.py", "eval_flow.py", "para_judge.py", "stress_flow.py", "perturb.py", "ZC_CODE_MARKER", "__pycache__", "adapter_hashes.json", "adapter_hashes_p.json", "adapter_hashes_w.json"}
KEEP_DOC = "dm.py,industry.py,tools.py,api_spec.py,merge.py"

V = {
    "Docker_C4": {"config": "configs/flow_p5c4.json", "adapter": "src/adapter_hashes.json", "judge": "src/adapter_hashes_w.json",
                  "sub": "Process/runs/p5_results/merged/C4_submission.jsonl", "pa": "0.939", "counts": "963 / 24 / 13", "solver": "V3A", "w": "0.70"},
    "Docker_C6": {"config": "configs/flow_p5c6.json", "adapter": "src/adapter_hashes.json", "judge": "src/adapter_hashes_w.json",
                  "sub": "Process/runs/p5_results/merged/p5c6h.jsonl", "pa": "0.834", "counts": "891 / 57 / 52", "solver": "V3A", "w": "0.70"},
    "Docker_PG": {"config": "configs/flow_pg.json", "adapter": "0.994_Docker_submission/adapter_hashes.json", "judge": None,
                  "sub": "Process/runs/p5_results/merged/pgq.jsonl", "pa": "0.880", "counts": "920 / 40 / 40", "solver": "WG1", "w": "0.35"},
}


def ign(d, names):
    out = {n for n in names if n in SKIP}
    if os.path.basename(d) == "refs": out.add("public_rulebook.md")
    return out


def build(name, v):
    dst = os.path.join(ROOT, name)
    if os.path.isdir(dst): shutil.rmtree(dst)
    app = os.path.join(dst, "app")
    shutil.copytree(os.path.join(ROOT, "src"), os.path.join(app, "src"), ignore=ign)
    os.makedirs(os.path.join(app, "training")); shutil.copy(os.path.join(ROOT, "training", "merge.py"), os.path.join(app, "training", "merge.py"))
    os.makedirs(os.path.join(app, "configs"))
    c = json.load(open(os.path.join(ROOT, v["config"]), encoding="utf-8")); c.pop("_doc", None)
    json.dump(c, open(os.path.join(app, "configs", "solution.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    pys = [os.path.join(app, "src", f) for f in os.listdir(os.path.join(app, "src")) if f.endswith(".py")] + [os.path.join(app, "training", "merge.py")]
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "pkg", "strip_comments.py")] + pys + ["--keep-doc", KEEP_DOC], capture_output=True, text=True)
    print(name, r.stdout.strip(), r.stderr.strip()[-300:])
    if r.returncode: sys.exit(1)
    for f, a, b in (("agent.py", " (pub_noind)", ""), ("api_spec.py", " (the pub_noind evaluation set)", ""),
                    ("shopctx.py", "PUBLIC_RULEBOOK", "REF_RULEBOOK"), ("shopctx.py", "public_rulebook.md", "reference_rulebook.md"),
                    ("lrl_gen.py", "PUBLIC_RULEBOOK", "REF_RULEBOOK"), ("rules_first.py", "public_rulebook.md", "reference_rulebook.md")):
        p = os.path.join(app, "src", f); t = open(p, encoding="utf-8").read(); open(p, "w", encoding="utf-8").write(t.replace(a, b))
    os.makedirs(os.path.join(dst, "hashes"))
    shutil.copy(os.path.join(ROOT, v["adapter"]), os.path.join(dst, "hashes", "adapter.json"))
    if v["judge"]: shutil.copy(os.path.join(ROOT, v["judge"]), os.path.join(dst, "hashes", "judge.json"))
    shutil.copy(os.path.join(ROOT, "src", "model_hashes.json"), os.path.join(dst, "hashes", "model.json"))
    shutil.copy(os.path.join(ROOT, v["sub"]), os.path.join(dst, "submission.jsonl"))
    for f, body in files(name, v).items():
        p = os.path.join(dst, f); os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w", encoding="utf-8").write(body)
        if f.endswith(".sh"): os.chmod(p, 0o755)


def files(name, v):
    judge = bool(v["judge"])
    out = {}
    out["Dockerfile"] = f"""FROM vllm/vllm-openai:v0.32.0 AS merge
COPY models/Qwen3.5-9B/ /w/base/
COPY models/adapter/ /w/adapter/
COPY app/training/merge.py /w/merge.py
COPY docker/merge_assets.py /w/merge_assets.py
RUN python3 /w/merge_assets.py

FROM vllm/vllm-openai:v0.32.0
RUN pip install --no-cache-dir "pandas==2.3.3" "numpy==2.1.3" "openpyxl>=3.1" "python-calamine>=0.3" "duckdb==1.3.2"
WORKDIR /app
COPY app/src/ /app/src/
COPY app/configs/ /app/configs/
COPY app/training/merge.py /app/training/merge.py
""" + ("COPY models/adapters/ /models/adapters/\n" if judge else "") + """COPY docker/predict.sh /app/predict.sh
COPY --from=merge /w/model/ /models/solver/
RUN chmod +x /app/predict.sh && python3 -c "import sys; sys.path.insert(0, '/app/src'); import predict_flow, hybrid, judge, lrl_gen, units, pipeline, agent; print('import OK')"
ENV MODEL=/models/solver ZC_ADAPTER_ROOT=/models/adapters HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONHASHSEED=0
ENTRYPOINT ["/app/predict.sh"]
"""
    out["docker/predict.sh"] = """#!/bin/bash
set -euo pipefail
DATA=${DATA:-/data}; RESULT=${RESULT:-/result}
SPLIT=$(dirname "$(find "$DATA" -name questions.jsonl | head -1)")
[ -f "$SPLIT/questions.jsonl" ] || { echo "no questions.jsonl under $DATA"; exit 1; }
mkdir -p "$RESULT/logs" /tmp/zc
exec python3 /app/src/predict_flow.py "$SPLIT" "$RESULT" --model "${MODEL:-/models/solver}" --budget "${BUDGET:-20400}" \\
    --runs 1 --dp "${DP:-0}" --work /tmp/zc --config "${CONFIG:-/app/configs/solution.json}"
"""
    out["docker/merge_assets.py"] = open(os.path.join(ROOT, "Docker_submission", "merge_assets.py"), encoding="utf-8").read().split('"""', 2)[2].lstrip("\n").replace('                                 # tests: another root', '')
    out["scripts/fetch_assets.sh"] = """#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pip install -q -U "huggingface_hub>=0.25" kaggle
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-9B", local_dir="models/Qwen3.5-9B", allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "LICENSE", "README.md"])
PY
rm -rf models/Qwen3.5-9B/.cache
python3 - <<'PY'
import json, hashlib, os, sys
h = json.load(open("hashes/model.json"))
bad = []
for f, ref in h["files"].items():
    d = hashlib.sha256()
    with open(os.path.join("models/Qwen3.5-9B", f), "rb") as fi:
        for chunk in iter(lambda: fi.read(16 << 20), b""): d.update(chunk)
    if d.hexdigest() != ref["sha256"]: bad.append(f)
print("base model sha256:", "OK" if not bad else f"MISMATCH {bad}"); sys.exit(1 if bad else 0)
PY
fetch() {
  local ds=$1 ref=$2 out=$3
  rm -rf "$out" models/_dl; mkdir -p "$out"
  kaggle datasets download -d "$ds" -p models/_dl --unzip -q
  python3 - "$ref" "$out" <<'PY'
import json, hashlib, os, shutil, glob, sys
ref, out = json.load(open(sys.argv[1]))["files"], sys.argv[2]
bad = []
for rel, h in ref.items():
    c = glob.glob("models/_dl/**/" + rel, recursive=True) or glob.glob("models/_dl/" + rel)
    if not c or hashlib.sha256(open(c[0], "rb").read()).hexdigest() != h: bad.append(rel); continue
    dst = os.path.join(out, rel); os.makedirs(os.path.dirname(dst) or ".", exist_ok=True); shutil.copy(c[0], dst)
shutil.rmtree("models/_dl")
print(out, "sha256:", "OK" if not bad else f"MISMATCH {bad}"); sys.exit(1 if bad else 0)
PY
}
fetch "$(python3 -c "import json; print(json.load(open('hashes/adapter.json'))['dataset'])")" hashes/adapter.json models/adapter
""" + ("""fetch "$(python3 -c "import json; print(json.load(open('hashes/judge.json'))['dataset'])")" hashes/judge.json models/adapters
""" if judge else "mkdir -p models/adapters\n") + "du -sh models/*\n"
    out["scripts/build.sh"] = """#!/bin/bash
set -euo pipefail
TAG=${1:-datamate:latest}
cd "$(dirname "$0")/.."
[ -f models/Qwen3.5-9B/config.json ] && [ -f models/adapter/soup.json ] || { echo "models missing: run scripts/fetch_assets.sh first"; exit 1; }
docker build -t "$TAG" .
docker images "$TAG"
"""
    out["scripts/run_public_test.sh"] = """#!/bin/bash
set -euo pipefail
DATA=$(realpath "$1"); RESULT=$(realpath -m "${2:-./result_public}"); TAG=${3:-datamate:latest}
mkdir -p "$RESULT"
time docker run --rm --gpus all --network none --shm-size 16g --ipc host ${DOCKER_ARGS:-} -v "$DATA":/data:ro -v "$RESULT":/result "$TAG"
echo "answers: $(wc -l < "$RESULT/submission.jsonl") -> $RESULT/submission.jsonl"
"""
    out["scripts/export_image.sh"] = """#!/bin/bash
set -euo pipefail
TAG=${1:-datamate:latest}; OUT=${2:-datamate_solution.tar.gz}
docker save "$TAG" | gzip -1 > "$OUT"
md5sum "$OUT" | tee "$OUT.md5"
ls -lh "$OUT"
"""
    out[".dockerignore"] = "models/_dl\nresult_*\n*.tar.gz\n*.md5\n**/__pycache__\n"
    out["README_SOLUTION.md"] = solution_readme(name, v)
    out["README.md"] = guide(name, v)
    return out


def solution_readme(name, v):
    judge = bool(v["judge"])
    pg = name == "Docker_PG"
    steps = ["**Ingest and schema** (`src/ingest.py`, `src/automap.py`, `src/canon.py`): CSV / Excel / SQLite files are cleaned and mapped to canonical tables; the model writes the mapping program when no known layout fits.",
             "**Toolkit** (`src/dm.py`, `src/industry.py`): periods, valid orders, aggregation, ranking with ties, rounding and formatting." + (
                 " Industry toolkits are shown only to shops whose rulebook has that industry section." if name == "Docker_C4" else
                 " Industry-specific definitions are not built in: only general text tools (product line, brand, vehicle plate, service task) are given; every business definition comes from the split's rulebook."),
             "**Rule cards and shop functions written by the model at run time** (`src/lrl_gen.py`, `src/units.py`): the split's rulebook rows become rule cards, the shop's data gives value aliases, and each shop-specific row becomes a Python function kept only when it runs on the shop's data and its samples agree.",
             "**Vietnamese glossary** (`src/refs/lrl/glossary_vi.jsonl`): meanings of common abbreviations, chat spellings, money and unit words, shown when the question uses the word.",
             ("**Question reading** (`src/hints.py`, `src/qparse.py`, `src/qnorm.py`): the period(s), thresholds, entities of the shop's data, expected answer form and the rulebook rows the question mentions are added to the prompt." if pg else
              "**Consistency check** (`src/consistency.py`, `src/precheck.py`): each program is compared with what the question states (periods, thresholds, answer form); a program that ignores them counts half in the vote."),
             "**Generation** (`src/hybrid.py`): retrieved examples of training shops; the model writes a program, which is executed in a sandbox with repair turns; several samples are voted with the official matching rules." + (" A review turn re-checks suspicious results." if pg else "")]
    if judge: steps.append("**Judge** (`src/judge.py`): on questions whose samples disagree, the judge adapter chooses among the candidates (list / point / point with reasoning, ensembled); it abstains when no candidate is reliable.")
    body = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    adapters = f"- **Solver** {v['solver']}, merged at build time with weight {v['w']} (served weights `/models/solver`)."
    if judge: adapters += "\n- **Judge** WJR1, merged at run time into a copy of the solver weights."
    return f"""# DataMate solution

## Models (total ≤ 15B parameters)

| Model | Parameters | Licence | Released |
|---|---|---|---|
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) + LoRA fine-tunes merged into it | 9.65B | Apache-2.0 | 27/02/2026 |

LoRA adapters of the same backbone, trained on the competition's training split only:
{adapters}

sha256 of every file: `hashes/`. Inference is fully offline (`--network none`).

## Method (`app/src/predict_flow.py`, configuration `app/configs/solution.json`)

The model writes every answer program. No logic is keyed on a question or business identifier and no answer is hard-coded;
every rule card and shop function of a test split is written at run time from that split's own rulebook and data.

{body}

## Input / output

Input: the test split mounted read-only at `/data` (`questions.jsonl`, `rulebook.md`, `calendar.csv`, `data/<business_id>/`).
Output: `/result/submission.jsonl` (one line per question, appended while running) and `/result/logs/` (`run.log` and one JSON
per question with the programs written, execution output, votes and final answer).

## Public test result

PA **{v['pa']}** (right / wrong / abstained: {v['counts']}), file `submission.jsonl` of this folder (Kaggle 4 × L4).
"""


def guide(name, v):
    return f"""# {name} — hướng dẫn

Lời giải DataMate (Zalo AI Challenge 2026) đóng gói thành Docker image. Điểm public: **{v['pa']}**.

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `Dockerfile` | 2 stage: gộp LoRA vào Qwen3.5-9B, rồi image chạy (vLLM 0.32.0, CUDA ≥ 12.8) |
| `app/src/` | mã nguồn: `predict_flow.py` (điểm vào), `ingest.py` / `automap.py` / `canon.py` (đọc dữ liệu), `dm.py` / `industry.py` (bộ hàm), `lrl_gen.py` / `units.py` (thẻ luật, hàm cửa hàng), `hybrid.py` / `pipeline.py` (sinh chương trình), `judge.py` (phán xử), `refs/` (ví dụ, từ điển) |
| `app/configs/solution.json` | cấu hình chạy |
| `app/training/merge.py` | gộp LoRA vào trọng số |
| `docker/predict.sh` | entrypoint: đọc `/data`, ghi `/result/submission.jsonl` + `/result/logs/` |
| `docker/merge_assets.py` | bước gộp lúc build |
| `scripts/` | `fetch_assets.sh` (tải trọng số), `build.sh`, `run_public_test.sh`, `export_image.sh` |
| `hashes/` | sha256 của trọng số gốc và adapter |
| `submission.jsonl` | kết quả trên public test |
| `README_SOLUTION.md` | mô tả mô hình và phương pháp cho BTC |

## 1. Môi trường

- Linux (hoặc Windows + Docker Desktop + WSL2), Docker ≥ 24, ổ trống ≥ 80 GB.
- Chạy thử cần GPU NVIDIA ≥ 24 GB (máy chấm 2 × RTX 5090), driver ≥ 570, NVIDIA Container Toolkit. Kiểm tra:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

- Python 3.10+ và token Kaggle có quyền đọc dataset adapter (`hashes/adapter.json`, `hashes/judge.json`: trường `dataset`) ở `~/.kaggle/kaggle.json`.

## 2. Tải trọng số (~20 GB) và build

Chạy trong thư mục này:

```bash
scripts/fetch_assets.sh
```

```bash
scripts/build.sh datamate:latest
```

Cuối build phải có dòng `import OK`.

## 3. Chạy thử trên public test (giống BTC: không mạng, `/data` chỉ đọc)

```bash
scripts/run_public_test.sh ~/zc_data/public_test ./result_public
```

Thư mục truyền vào phải chứa `questions.jsonl`, `rulebook.md`, `calendar.csv`, `data/`. Kết quả: `result_public/submission.jsonl`,
log ở `result_public/logs/`. Thời gian tham chiếu ~70–110 phút cho 1000 câu trên 4 × L4; giới hạn mặc định 5 h 40 phút (`BUDGET`).

## 4. Xuất file nộp

```bash
scripts/export_image.sh datamate:latest datamate_solution.tar.gz
```

Upload `datamate_solution.tar.gz`, gửi link cùng **checksum MD5** (nội dung file `datamate_solution.tar.gz.md5`, tạo sẵn bởi script) theo hướng dẫn nộp của BTC. Tự kiểm lại: `md5sum datamate_solution.tar.gz`.
"""


if __name__ == "__main__":
    for n, v in V.items(): build(n, v)
