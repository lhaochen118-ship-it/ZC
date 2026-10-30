# DataMate solution image. Base: official vLLM image (CUDA >= 12.8, Blackwell / RTX 5090 support).
# If the organisers' Docker guideline names another base image, change FROM and keep the rest.
# Build context = repo root (see docker/build.sh); model weights must already be in ./models/<name>/ (docker/fetch_assets.sh).
FROM vllm/vllm-openai:v0.32.0

ARG MODEL_NAME=Qwen3-14B-FP8
# python deps of the agent (installed at build time; the container runs fully offline)
RUN pip install --no-cache-dir "pandas>=2.2,<4" "openpyxl>=3.1" "python-calamine>=0.3" "numpy<3"
WORKDIR /app
COPY src/ /app/src/
COPY docker/predict.sh /app/predict.sh
COPY models/${MODEL_NAME}/ /models/${MODEL_NAME}/
RUN chmod +x /app/predict.sh && python3 -c "import sys; sys.path.insert(0, '/app/src'); import ingest, agent, score, semantic, vnnorm; print('agent import OK')"
ENV MODEL=/models/${MODEL_NAME} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONHASHSEED=0
ENTRYPOINT ["/app/predict.sh"]
