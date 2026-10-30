# DataMate solution image (DRAFT: switch the base image to the one named in the organisers' Docker guideline).
# Requirements: offline (no network at run time), CUDA >= 12.8 for RTX 5090 (Blackwell), weights inside the image.
# Build context = repo root, with the model weights downloaded beforehand into ./models/<name>/ and the extra wheels
# into ./wheels/ (e.g. from the Kaggle assets kernel nhatlong1103/zc-assets).
FROM vllm/vllm-openai:v0.32.0

WORKDIR /app
COPY wheels/ /tmp/wheels/
RUN pip install --no-index --find-links /tmp/wheels python-calamine && rm -rf /tmp/wheels
COPY src/ /app/src/
COPY docker/predict.sh /app/predict.sh
COPY models/Qwen3-14B-FP8/ /models/Qwen3-14B-FP8/
RUN chmod +x /app/predict.sh
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 PYTHONHASHSEED=0
ENTRYPOINT ["/app/predict.sh"]
