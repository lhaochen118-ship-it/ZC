# DataMate v3 solution image. Base: official vLLM image (CUDA >= 12.8, Blackwell / RTX 5090), same vLLM version as the
# Kaggle experiments (0.32.0). If the organisers' Docker guideline names another base image, change FROM, keep the rest.
# Build context = repo root (docker/build.sh); model weights must already be in ./models/<name>/ (docker/fetch_assets.sh).
ARG MODEL_NAME=Qwen3.5-9B

# stage 1: the fine-tuned LoRA adapter or soup of adapters (models/adapter/, docker/fetch_assets.sh; docker/merge_assets.py) merged into the base checkpoint with the vLLM
# image's own torch (training/merge.py: same tensor names as the base, served with --quantization fp8); without an adapter
# (models/adapter/NONE) the base weights are used as they are. Only the merged weights reach the final image.
FROM vllm/vllm-openai:v0.32.0 AS merge
ARG MODEL_NAME
COPY models/${MODEL_NAME}/ /w/base/
COPY models/adapter/ /w/adapter/
COPY training/merge.py /w/merge.py
COPY docker/merge_assets.py /w/merge_assets.py
RUN python3 /w/merge_assets.py

FROM vllm/vllm-openai:v0.32.0
ARG MODEL_NAME
# pandas / numpy as in the Kaggle runs that were scored (run_kaggle.py logs them: pandas 2.3.3, numpy 2.1.3, Python 3.13):
# pandas 3 maps NaN dictionary keys differently (one public question changed between the two), so the version is pinned
RUN pip install --no-cache-dir "pandas==2.3.3" "numpy==2.1.3" "openpyxl>=3.1" "python-calamine>=0.3" "duckdb==1.3.2"
WORKDIR /app
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY docker/predict.sh /app/predict.sh
# v2: the W phases (tool writer, linker, judge) merge their LoRA adapters into a copy of the weights at run time (training/merge.py)
COPY training/merge.py /app/training/merge.py
COPY models/adapters/ /models/adapters/
COPY --from=merge /w/model/ /models/${MODEL_NAME}/
RUN chmod +x /app/predict.sh && python3 -c "import sys; sys.path.insert(0, '/app/src'); import ingest, dm, qparse, qnorm, automap, industry, canon, fewshot, score, agent, pipeline, merge_sub, predict_flow, vote, api_spec, grammar, hints, prompting, retrieval, shopctx, tools, precheck, hybrid, confidence, sqlesc, units, lrl, lrl_gen, selector, link, judge, duckdb; print('agent import OK')"
ENV MODEL=/models/${MODEL_NAME} ZC_ADAPTER_ROOT=/models/adapters HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONHASHSEED=0
ENTRYPOINT ["/app/predict.sh"]
