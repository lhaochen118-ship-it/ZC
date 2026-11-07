# DataMate v3 solution image. Base: official vLLM image (CUDA >= 12.8, Blackwell / RTX 5090), same vLLM version as the
# Kaggle experiments (0.32.0). If the organisers' Docker guideline names another base image, change FROM, keep the rest.
# Build context = repo root (docker/build.sh); model weights must already be in ./models/<name>/ (docker/fetch_assets.sh).
FROM vllm/vllm-openai:v0.32.0

ARG MODEL_NAME=Qwen3.5-9B
# pandas / numpy as in the Kaggle runs that were scored (run_kaggle.py logs them: pandas 2.3.3, numpy 2.1.3, Python 3.13):
# pandas 3 maps NaN dictionary keys differently (one public question changed between the two), so the version is pinned
RUN pip install --no-cache-dir "pandas==2.3.3" "numpy==2.1.3" "openpyxl>=3.1" "python-calamine>=0.3"
WORKDIR /app
COPY src/ /app/src/
COPY docker/predict.sh /app/predict.sh
COPY models/${MODEL_NAME}/ /models/${MODEL_NAME}/
RUN chmod +x /app/predict.sh && python3 -c "import sys; sys.path.insert(0, '/app/src'); import ingest, dm, qparse, qnorm, automap, industry, canon, fewshot, score, agent, pipeline, rules_first, merge_sub, predict_flow, vote, canon_llm; print('agent import OK')"
ENV MODEL=/models/${MODEL_NAME} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONHASHSEED=0
ENTRYPOINT ["/app/predict.sh"]
