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
VALUED = ("--owner", "--ds", "--env", "--ks", "--ms", "--train-data")
argv = [a for i, a in enumerate(sys.argv) if a not in ("--gpu", "--l4", "--internet", "--no-upload") + VALUED and sys.argv[i - 1] not in VALUED]
kdir, slug, jobs = argv[1], argv[2], [j.split() for j in argv[3].split(";") if j.strip()]

# --- 1. code dataset
stage = os.path.join(kdir, "_code"); shutil.rmtree(stage, ignore_errors=True)
shutil.copytree(os.path.join(ROOT, "src"), stage, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "cache"))
# training code (+ configs) go with it (docs/08 §5.1); training data only for the subfolders of training/data/out named by --train-data
TRAIN_DATA = sys.argv[sys.argv.index("--train-data") + 1].split(",") if "--train-data" in sys.argv else []
if os.path.isdir(os.path.join(ROOT, "training")):
    shutil.copytree(os.path.join(ROOT, "training"), os.path.join(stage, "training"), ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "out"))
    for d in TRAIN_DATA:
        src_d = os.path.join(ROOT, "training", "data", "out", d)
        def _link(s, t):                                                 # 16/11: hard links (a 3.4 GB adapter dir copied per
            try: os.link(s, t)                                           # launch filled the disk 3 times); copy across filesystems
            except OSError: shutil.copy2(s, t)
        if os.path.isdir(src_d): shutil.copytree(src_d, os.path.join(stage, "training", "data", "out", d), copy_function=_link)
        else: sys.exit(f"--train-data {d}: no directory {src_d}")
if os.path.isdir(os.path.join(ROOT, "configs")): shutil.copytree(os.path.join(ROOT, "configs"), os.path.join(stage, "configs"))
if os.path.isdir(os.path.join(ROOT, "tests", "splits")):               # evaluation set ids (src/eval_flow.py reads refs/splits)
    shutil.copytree(os.path.join(ROOT, "tests", "splits"), os.path.join(stage, "refs", "splits"), dirs_exist_ok=True)
# version token: the runner refuses a mounted code dataset whose marker differs (a kernel pushed before the new dataset version
# was live ran the previous code once: huyenco/zc-pfwc05thinks1-1 did not find a config added in between)
import hashlib                                                        # 16/11: token = hash of the staged CONTENT (was a random uuid):
_h = hashlib.sha1()                                                    # consecutive launches of the same code on one account no longer
for _root, _dirs, _files in sorted(os.walk(stage)):                    # refuse an older version with identical content (STALE CODE x5)
    _dirs.sort()
    for _f in sorted(_files):
        _p = os.path.join(_root, _f); _s = os.path.getsize(_p); _h.update(f"{os.path.relpath(_p, stage)}\0{_s}\0".encode())
        with open(_p, "rb") as _fh:
            _h.update(_fh.read(65536))
            if _s > 131072: _fh.seek(-65536, 2); _h.update(_fh.read(65536))
CODE_TOKEN = _h.hexdigest()[:16]
open(os.path.join(stage, "ZC_CODE_MARKER"), "w").write(CODE_TOKEN + "\n")
print("code version:", CODE_TOKEN, flush=True)
json.dump({"title": "zc-code", "id": CODE_DS, "licenses": [{"name": "CC0-1.0"}]}, open(os.path.join(stage, "dataset-metadata.json"), "w"))
if "--no-upload" not in sys.argv:                                       # preflight.py builds the kernel files offline
    exists = subprocess.run(["kaggle", "datasets", "status", CODE_DS], capture_output=True, text=True).stdout.strip().lower() == "ready"
    cmd = (["kaggle", "datasets", "version", "-p", stage, "-m", "update"] if exists else ["kaggle", "datasets", "create", "-p", stage]) + ["-r", "zip", "-q"]
    r = subprocess.run(cmd, capture_output=True, text=True); print("code dataset:", (r.stdout + r.stderr).strip().splitlines()[-1:])
    # 16/11: a kernel pushed while the new version was still being processed mounted the previous one (STALE CODE, 5 merges lost,
    # worst with a 3 GB --train-data): wait until Kaggle reports the dataset ready again, then a short grace period
    import time as _t
    _t0 = _t.time(); _t.sleep(20)
    while _t.time() - _t0 < 900:
        st = subprocess.run(["kaggle", "datasets", "status", CODE_DS], capture_output=True, text=True).stdout.strip().lower()
        if st == "ready": break
        _t.sleep(15)
    _t.sleep(45); print(f"code dataset ready after {_t.time() - _t0:.0f}s", flush=True)

# --- 2. kernel
runner = r'''
import os, sys, json, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor
JOBS = %s
ENV = %s
t0 = time.time()
W = "/kaggle/working"; C = os.path.join(W, "code")
code = [d for d, _, fs in os.walk("/kaggle/input") if "ZC_CODE_MARKER" in fs][0]
_tok = open(os.path.join(code, "ZC_CODE_MARKER")).read().strip()
if _tok != "%s":
    print("STALE CODE: the mounted code dataset is version", _tok, "but this kernel was built for %s; relaunch", flush=True); sys.exit(3)
shutil.copytree(code, C, dirs_exist_ok=True)
R = os.path.join(W, "results"); L = os.path.join(W, "logs"); os.makedirs(R, exist_ok=True); os.makedirs(L, exist_ok=True)
DATA = next((d for d, _, _ in os.walk("/kaggle/input") if os.path.basename(d) == "%s"), "")
try:
    import torch; NGPU = torch.cuda.device_count(); names = [torch.cuda.get_device_name(i) for i in range(NGPU)]
except Exception: NGPU, names = 0, []
print("data:", DATA, "| cpus", os.cpu_count(), "| gpus", names, flush=True)
def gpus_of(j):                                                     # GPUs one job uses: --dp N, or every GPU for the
    if any(x in " ".join(j) for x in ("arms_job", "full_job", "soup_job", "d3_job", "training/smoke")): return max(1, NGPU)   # training
    return int(j[j.index("--dp") + 1]) if "--dp" in j else 1
MULTI = NGPU and sum(gpus_of(j) for j in JOBS) > NGPU                  # jobs that each need several GPUs run one after another
NW = 1 if MULTI else max(1, min(len(JOBS), NGPU if NGPU else (os.cpu_count() or 4) - 1))
TH = str(max(1, (os.cpu_count() or 4) // NW))                        # BLAS threads per job (no oversubscription)
SUB = {"DATA": DATA, "INPUT": "/kaggle/input", "OUT": R}
# Kaggle stops a commit session at 12 h and its outputs are then lost (user 10/11): every job gets ZC_DEADLINE (training
# checkpoints and stops, the agent's --budget ends with abstains), a sequential job that would not finish in time is not
# started (listed in results/_skipped_jobs.json for a new kernel), a job still running at the deadline is killed so the
# session ends cleanly and the finished outputs are saved.
SESSION_S = int(ENV.get("ZC_SESSION_S", str(12 * 3600))); MARGIN_S = int(ENV.get("ZC_SESSION_MARGIN", "2400"))
DEADLINE = t0 + SESSION_S - MARGIN_S; DURS = []; SKIPPED = []
def run(ij):
    i, j = ij
    est = max(DURS) if DURS else 0                                     # a sequential job: as long as the longest one so far
    if DURS and time.time() + est > DEADLINE or time.time() > DEADLINE:
        SKIPPED.append(" ".join(j)); print(f"SKIPPED (12 h session limit, {(DEADLINE - time.time()) / 3600:.1f} h left, longest job {est / 3600:.1f} h): {' '.join(j)}", flush=True)
        return "skipped"
    env = dict(os.environ, OMP_NUM_THREADS=TH, MKL_NUM_THREADS=TH, OPENBLAS_NUM_THREADS=TH, ZC_OUT=R, ZC_DATA=DATA, ZC_DEADLINE=str(int(DEADLINE)), **ENV)
    if NGPU and len(JOBS) > 1 and not MULTI: env["CUDA_VISIBLE_DEVICES"] = str(i %% NGPU)
    def sub(x):                                                        # OUT, OUT/train, DATA/..., INPUT/... (prefix tokens)
        for k, v in SUB.items():
            if x == k or x.startswith(k + "/"): return v + x[len(k):]
        return x
    cmd = [sys.executable, j[0]] + [sub(a) for a in j[1:]]
    t = time.time(); log = os.path.join(L, "_".join(j).replace(".py", "").replace("/", "-")[:120] + ".log")
    p = subprocess.Popen(cmd, cwd=C, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try: rc = p.wait(timeout=max(60, DEADLINE + MARGIN_S // 2 - time.time()))
    except subprocess.TimeoutExpired:                                  # past the deadline: the whole process group goes
        import signal
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: p.kill()
        p.wait(); rc = "killed_at_deadline"
    DURS.append(time.time() - t)
    print(f"done {' '.join(j)} rc={rc} {time.time()-t:.0f}s", flush=True); return rc
print("parallel jobs", NW, "| threads/job", TH, "| sequential (multi-GPU jobs)" if MULTI else "", flush=True)
with ThreadPoolExecutor(NW) as ex: rcs = list(ex.map(run, enumerate(JOBS)))
shutil.rmtree(C, ignore_errors=True)
if SKIPPED:
    json.dump(SKIPPED, open(os.path.join(R, "_skipped_jobs.json"), "w"), indent=1)
    print(f"JOBS_SKIPPED_12H {len(SKIPPED)}: run them in a new kernel", flush=True)
print("EVAL_DONE", rcs, f"{time.time()-t0:.0f}s", flush=True)
''' % (json.dumps(jobs), json.dumps(ENV), CODE_TOKEN, CODE_TOKEN, DATA_DS.split("/")[1])
os.makedirs(kdir, exist_ok=True); open(os.path.join(kdir, "run.py"), "w").write(runner)
meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": GPU, "enable_tpu": False, "enable_internet": NET,
        "dataset_sources": [CODE_DS, DATA_DS] + EXTRA_DS, "competition_sources": [], "kernel_sources": KS, "model_sources": MS}
if L4: meta.update(machine_shape="NvidiaL4", competition_sources=["gemma-4-developer-agent"])
json.dump(meta, open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=2)
shutil.rmtree(stage, ignore_errors=True)
print("kernel dir:", kdir, "| jobs:", len(jobs), "| datasets:", meta["dataset_sources"])
