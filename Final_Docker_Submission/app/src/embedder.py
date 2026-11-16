#!/usr/bin/env python3
import numpy as np

TASK = "Given a shop owner's question about the shop's sales data, retrieve questions that are answered by the same kind of computation"


class Embedder:
    def __init__(self, path, device=None, max_len=128, batch=64):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        self.m = AutoModel.from_pretrained(path, dtype=torch.float16 if self.dev == "cuda" else torch.float32).to(self.dev).eval()
        self.max_len, self.batch = max_len, batch

    def encode(self, texts, query=False):
        torch = self.torch; out = []
        texts = [f"Instruct: {TASK}\nQuery: {t}" if query else t for t in texts]
        with torch.no_grad():
            for i in range(0, len(texts), self.batch):
                b = self.tok(texts[i:i + self.batch], padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.dev)
                h = self.m(**b).last_hidden_state[:, -1]
                out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1))
