# vLLM 0.32.0 (released 2026-10-18), Python 3.12, CUDA 12.9 (the guide allows CUDA 12.x and 13.x up to 13.2). Qwen3_5ForCausalLM is in its registry.
FROM vllm/vllm-openai:v0.32.0-cu129

ENV PIP_BREAK_SYSTEM_PACKAGES=1
WORKDIR /code
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
# The base image's torchcodec needs libnvrtc.so.13 (CUDA 13), which this CUDA 12.9 image lacks, so `from vllm import LLM` died with OSError.
# vLLM treats a missing torchcodec (ImportError) as "no audio input", which we never use. The import check fails the build if this breaks again.
RUN python3 -m pip uninstall -y torchcodec && python3 -c "from vllm import LLM, SamplingParams; import sherpa_onnx; print('vllm imports ok')"

# Offline at scoring. One engine process: vLLM stays in-process, so no IPC through the container's small /dev/shm, in predict.sh and in Jupyter alike
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONHASHSEED=42 PYTHONUNBUFFERED=1

COPY . .
RUN chmod +x predict.sh start_jupyter.sh && ln -s /result /code/result

ENTRYPOINT []
CMD ["bash", "/code/predict.sh"]
