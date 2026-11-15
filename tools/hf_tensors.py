#!/usr/bin/env python3
"""Read single tensors of a Hugging Face safetensors repo over HTTP range requests (no full download).
    get(repo, names) -> {name: np.ndarray (float32 for bf16)}; used to derive the REAP-64 expert keep-list (P5 plan P01)."""
import json, struct, urllib.request, numpy as np
H = "https://huggingface.co/{repo}/resolve/main/{f}"
_hdr = {}


def _range(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    with urllib.request.urlopen(req, timeout=120) as r: return r.read()


def header(repo, f):
    k = (repo, f)
    if k not in _hdr:
        url = H.format(repo=repo, f=f); n = struct.unpack("<Q", _range(url, 0, 7))[0]
        _hdr[k] = (8 + n, json.loads(_range(url, 8, 8 + n - 1)))
    return _hdr[k]


def weight_map(repo, index="model.safetensors.index.json"):
    with urllib.request.urlopen(H.format(repo=repo, f=index), timeout=120) as r: return json.load(r)["weight_map"]


def get(repo, name, f):
    base, h = header(repo, f); m = h[name]; a, b = m["data_offsets"]
    raw = _range(H.format(repo=repo, f=f), base + a, base + b - 1)
    if m["dtype"] == "BF16":
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(m["shape"])
    return np.frombuffer(raw, dtype={"F32": np.float32, "F16": np.float16}[m["dtype"]]).reshape(m["shape"])
