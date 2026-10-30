#!/usr/bin/env python3
"""Build a Kaggle script kernel that runs jobs from src/ on Kaggle (all heavy compute runs on Kaggle, not in the container).

1. Uploads (or versions) the code dataset `<owner>/zc-code` with every file of src/ (recursively).
2. Writes run.py + kernel-metadata.json: mounts the code, the PUBLIC data dataset and any extra datasets
   (model weights, wheels, caches), runs the jobs in parallel and saves results/ and logs/ to /kaggle/working.
Then: python3 automation/run_kernel.py <kernel_dir> <out_dir>   (fanout.py does both steps for every account/slot).

Usage: make_kernel.py <kernel_dir> <slug> "<script> <args>;<script> <args>;..." "" [--gpu|--l4] [--owner <user>]
                      [--ds a/b,c/d] [--ks owner/kernel,...] [--env K=V,K2=V2] [--internet]
  Tokens inside job args: DATA -> root of the mounted competition data dataset, INPUT -> /kaggle/input
                          (scripts look up model weights themselves), OUT -> results dir (also env ZC_OUT).
  --gpu : default T4 x2.  --l4 : 4x L4 via the gemma-4-developer-agent grant (rules accepted on the web, internet off).
  Several jobs on a GPU session are spread round-robin over the GPUs (CUDA_VISIBLE_DEVICES); a single job keeps all GPUs.
"""
import sys, os, json, shutil, subprocess

OWNER = sys.argv[sys.argv.index("--owner") + 1] if "--owner" in sys.argv else "nhatlong1103"
CODE_DS = f"{OWNER}/zc-code"
_tok = os.path.expanduser(f"~/.kaggle/tok_{OWNER}")                     # per-account token file
if os.path.isfile(_tok): os.environ["KAGGLE_API_TOKEN"] = open(_tok).read().strip()
DATA_DS = "nhatlong1103/zc-2026"                                         # PUBLIC copy of the organisers' data (docs/02 §4)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L4 = "--l4" in sys.argv; GPU = "--gpu" in sys.argv or L4
EXTRA_DS = sys.argv[sys.argv.index("--ds") + 1].split(",") if "--ds" in sys.argv else []
KS = sys.argv[sys.argv.index("--ks") + 1].split(",") if "--ks" in sys.argv else []   # kernel outputs to mount (e.g. vLLM wheels)
MS = sys.argv[sys.argv.index("--ms") + 1].split(",") if "--ms" in sys.argv else []   # Kaggle Models, e.g. qwen-lm/qwen-3/transformers/14b/1
NET = "--internet" in sys.argv and not L4                                  # L4 grant sessions refuse internet
ENV = dict(kv.split("=", 1) for kv in sys.argv[sys.argv.index("--env") + 1].split(",")) if "--env" in sys.argv else {}
VALUED = ("--owner", "--ds", "--env", "--ks", "--ms")
argv = [a for i, a in enumerate(sys.argv) if a not in ("--gpu", "--l4", "--internet") + VALUED and sys.argv[i - 1] not in VALUED]
kdir, slug, jobs = argv[1], argv[2], [j.split() for j in argv[3].split(";") if j.strip()]

# --- 1. code dataset
stage = os.path.join(kdir, "_code"); shutil.rmtree(stage, ignore_errors=True)
shutil.copytree(os.path.join(ROOT, "src"), stage, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "cache"))
json.dump({"title": "zc-code", "id": CODE_DS, "licenses": [{"name": "CC0-1.0"}]}, open(os.path.join(stage, "dataset-metadata.json"), "w"))
exists = subprocess.run(["kaggle", "datasets", "status", CODE_DS], capture_output=True, text=True).stdout.strip().lower() == "ready"
cmd = (["kaggle", "datasets", "version", "-p", stage, "-m", "update"] if exists else ["kaggle", "datasets", "create", "-p", stage]) + ["-r", "zip", "-q"]
r = subprocess.run(cmd, capture_output=True, text=True); print("code dataset:", (r.stdout + r.stderr).strip().splitlines()[-1:])

# --- 2. kernel
runner = r'''
import os, sys, json, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor
JOBS = %s
ENV = %s
t0 = time.time()
W = "/kaggle/working"; C = os.path.join(W, "code")
code = [d for d, _, fs in os.walk("/kaggle/input") if "ZC_CODE_MARKER" in fs][0]
shutil.copytree(code, C, dirs_exist_ok=True)
R = os.path.join(W, "results"); L = os.path.join(W, "logs"); os.makedirs(R, exist_ok=True); os.makedirs(L, exist_ok=True)
DATA = next((d for d, _, _ in os.walk("/kaggle/input") if os.path.basename(d) == "%s"), "")
try:
    import torch; NGPU = torch.cuda.device_count(); names = [torch.cuda.get_device_name(i) for i in range(NGPU)]
except Exception: NGPU, names = 0, []
print("data:", DATA, "| cpus", os.cpu_count(), "| gpus", names, flush=True)
NW = max(1, min(len(JOBS), NGPU if NGPU else (os.cpu_count() or 4) - 1))
TH = str(max(1, (os.cpu_count() or 4) // NW))                        # BLAS threads per job (no oversubscription)
SUB = {"DATA": DATA, "INPUT": "/kaggle/input", "OUT": R}
def run(ij):
    i, j = ij
    env = dict(os.environ, OMP_NUM_THREADS=TH, MKL_NUM_THREADS=TH, OPENBLAS_NUM_THREADS=TH, ZC_OUT=R, ZC_DATA=DATA, **ENV)
    if NGPU and len(JOBS) > 1: env["CUDA_VISIBLE_DEVICES"] = str(i %% NGPU)
    cmd = [sys.executable, j[0]] + [SUB.get(a, a) for a in j[1:]]
    t = time.time(); log = os.path.join(L, "_".join(j).replace(".py", "").replace("/", "-")[:120] + ".log")
    r = subprocess.run(cmd, cwd=C, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
    print(f"done {' '.join(j)} rc={r.returncode} {time.time()-t:.0f}s", flush=True); return r.returncode
print("parallel jobs", NW, "| threads/job", TH, flush=True)
with ThreadPoolExecutor(NW) as ex: rcs = list(ex.map(run, enumerate(JOBS)))
shutil.rmtree(C, ignore_errors=True)
print("EVAL_DONE", rcs, f"{time.time()-t0:.0f}s", flush=True)
''' % (json.dumps(jobs), json.dumps(ENV), DATA_DS.split("/")[1])
os.makedirs(kdir, exist_ok=True); open(os.path.join(kdir, "run.py"), "w").write(runner)
meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": GPU, "enable_tpu": False, "enable_internet": NET,
        "dataset_sources": [CODE_DS, DATA_DS] + EXTRA_DS, "competition_sources": [], "kernel_sources": KS, "model_sources": MS}
if L4: meta.update(machine_shape="NvidiaL4", competition_sources=["gemma-4-developer-agent"])
json.dump(meta, open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=2)
shutil.rmtree(stage, ignore_errors=True)
print("kernel dir:", kdir, "| jobs:", len(jobs), "| datasets:", meta["dataset_sources"])
