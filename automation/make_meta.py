#!/usr/bin/env python3
"""Generate kernel-metadata.json for a Kaggle kernel run.
Three tiers (policy: iterate on T4, reserve RTX 6000 / L4x4 for large-scale/large-model runs):
  t4  (default): default GPU (T4 x2) -> allocated instantly, no competition/machine_shape/docker pin.
  cpu          : no accelerator (light analysis / scoring jobs).
  rtx          : RTX PRO 6000 (96GB) via the ARC-Prize-2026 competition grant + byod docker image.
  l4           : 4x NVIDIA L4 (96GB total) via the gemma-4-developer-agent competition grant.

Competition-gated tiers (rtx, l4) only grant the accelerator once the account has accepted that
competition's rules on kaggle.com/competitions/<slug>/rules -- there is no API/CLI call that does this,
it's a one-time manual step per account. Attaching competition_sources in the metadata alone is not
enough; if rules were never accepted the push still succeeds but the session falls back to the default
GPU (no error).
Usage: make_meta.py <dir> <id> <code_file> <comma-datasets> [tier] [comma-competitions] [docker_image]
"""
import sys, json, os

d, kid, code = sys.argv[1], sys.argv[2], sys.argv[3]
datasets = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] else []
tier = (sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else "t4").lower()

meta = {
    "id": kid, "title": kid.split("/")[-1], "code_file": code,
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "dataset_sources": datasets,
    "competition_sources": [], "kernel_sources": [], "model_sources": [],
}

if tier == "rtx":
    meta["machine_shape"] = "NvidiaRtxPro6000"
    comps = sys.argv[6].split(",") if len(sys.argv) > 6 and sys.argv[6] else ["arc-prize-2026-arc-agi-3"]
    meta["competition_sources"] = comps
    docker = sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] else \
        "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
    if docker.lower() != "none":
        meta["docker_image"] = docker
if tier == "l4":
    meta["machine_shape"] = "NvidiaL4"   # verified 2026-10-30 (4x L4)
    comps = sys.argv[6].split(",") if len(sys.argv) > 6 and sys.argv[6] else ["gemma-4-developer-agent"]
    meta["competition_sources"] = comps
if tier == "cpu":
    meta["enable_gpu"] = False
# tier == "t4": leave machine_shape/competition/docker unset -> default GPU (instant, no queue)

os.makedirs(d, exist_ok=True)
json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
label = {"rtx": " (RTX PRO 6000)", "l4": " (4x L4)", "cpu": " (CPU only)"}.get(tier, " (default T4 x2, instant)")
print(f"wrote {os.path.join(d,'kernel-metadata.json')} | tier={tier.upper()}{label}")
if tier in ("rtx", "l4"):
    for c in meta["competition_sources"]:
        print(f"  NOTE: requires rules accepted at https://www.kaggle.com/competitions/{c}/rules "
              f"on account '{kid.split('/')[0]}' -- otherwise this silently falls back to the default GPU.")
