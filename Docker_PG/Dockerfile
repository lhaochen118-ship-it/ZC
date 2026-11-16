FROM vllm/vllm-openai:v0.32.0 AS merge
COPY models/Qwen3.5-9B/ /w/base/
COPY models/adapter/ /w/adapter/
COPY app/training/merge.py /w/merge.py
COPY docker/merge_assets.py /w/merge_assets.py
RUN python3 /w/merge_assets.py

FROM vllm/vllm-openai:v0.32.0
RUN pip install --no-cache-dir "pandas==2.3.3" "numpy==2.1.3" "openpyxl>=3.1" "python-calamine>=0.3" "duckdb==1.3.2"
WORKDIR /app
COPY app/src/ /app/src/
COPY app/configs/ /app/configs/
COPY app/training/merge.py /app/training/merge.py
COPY docker/predict.sh /app/predict.sh
COPY --from=merge /w/model/ /models/solver/
RUN chmod +x /app/predict.sh && python3 -c "import sys; sys.path.insert(0, '/app/src'); import predict_flow, hybrid, judge, lrl_gen, units, pipeline, agent; print('import OK')"
ENV MODEL=/models/solver ZC_ADAPTER_ROOT=/models/adapters HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONHASHSEED=0
ENTRYPOINT ["/app/predict.sh"]
