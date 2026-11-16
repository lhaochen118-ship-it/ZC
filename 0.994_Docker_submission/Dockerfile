# DataMate solution image (Zalo AI Challenge 2026) — final configuration configs/flow_p_final.json (system P, public 0.994).
# Base: official vLLM image, CUDA >= 12.8 (Blackwell / RTX 5090), the vLLM version of every scored Kaggle run (0.32.0).
# If the organisers' Docker guideline names another base image, change both FROM lines and keep the rest.
# Build context = repo root (0.994_Docker_submission/build.sh); weights in ./models/ (0.994_Docker_submission/fetch_assets.sh).
ARG MODEL_NAME=Qwen3.5-9B

# stage 1: S = base checkpoint + LoRA WG1 x 0.35 (models/adapter_p/soup.json) merged with the image's own torch (training/merge.py:
# same tensor names as the base, served with FP8 weights computed at load time). Only the merged weights reach the final image.
FROM vllm/vllm-openai:v0.32.0 AS merge
ARG MODEL_NAME
COPY models/${MODEL_NAME}/ /w/base/
COPY models/adapter_p/ /w/adapter/
COPY training/merge.py /w/merge.py
COPY ["0.994_Docker_submission/merge_assets.py", "/w/merge_assets.py"]
RUN python3 /w/merge_assets.py

FROM vllm/vllm-openai:v0.32.0
# pandas / numpy as in the scored Kaggle runs (pandas 2.3.3, numpy 2.1.3): pandas 3 maps NaN dictionary keys differently
RUN pip install --no-cache-dir "pandas==2.3.3" "numpy==2.1.3" "openpyxl>=3.1" "python-calamine>=0.3" "duckdb==1.3.2"
WORKDIR /app
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY ["0.994_Docker_submission/predict.sh", "/app/predict.sh"]
# system P has no W phase (no linker / judge adapter)
# The served directory name must NOT contain "qwen3": the Qwen3.5 chat template reasons first when enable_thinking is not passed,
# and src/agent.py passes it only for paths containing "qwen3" (think_kw "legacy"). Every scored run served a merged directory
# without that word (/tmp/soup_*/merged), i.e. with reasoning on; /models/solver reproduces that behaviour exactly.
COPY --from=merge /w/model/ /models/solver/
RUN chmod +x /app/predict.sh && python3 -c "import sys; sys.path.insert(0, '/app/src'); import ingest, dm, industry, automap, agent, pipeline, merge_sub, predict_flow, api_spec, hints, prompting, retrieval, shopctx, hybrid, units, lrl, lrl_gen, link, judge, duckdb; print('agent import OK')"
ENV MODEL=/models/solver HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONHASHSEED=0
ENTRYPOINT ["/app/predict.sh"]
