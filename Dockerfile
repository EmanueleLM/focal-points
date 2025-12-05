# GPU base (CUDA 12.1 runtime, Ubuntu 22.04)
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    TRANSFORMERS_NO_TORCHVISION=1 \
    TORCH_COMPILE_DISABLE=1 \
    HF_HOME=/cache \
    HF_DATASETS_CACHE=/cache \
    HUGGINGFACE_HUB_CACHE=/cache \
    TORCH_HOME=/cache \
    TRITON_CACHE_DIR=/cache \
    TORCHINDUCTOR_CACHE_DIR=/cache \
    XDG_CACHE_HOME=/cache \
    MPLCONFIGDIR=/cache

# System deps + Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl ca-certificates git build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Virtualenv
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN python -m pip install --upgrade pip setuptools wheel

# Workdirs
RUN mkdir -p /workspace /cache /logs /results && chmod -R 777 /workspace /cache /logs /results
WORKDIR /workspace

# Install Python deps
COPY requirements.txt /workspace/requirements.txt

# PyTorch CUDA 12.1 wheels (official index)
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch  

# Project dependencies
RUN python -m pip install --no-cache-dir -r /workspace/requirements.txt

# Copy project files
COPY . /workspace

# Set HOME
ENV HOME=/workspace

# Entrypoint 
ENTRYPOINT ["/bin/bash", "/workspace/experiments.sh"]
CMD ["-m", "meta-llama/Llama-3.3-70B-Instruct", "-n", "30", "-q", "8bit", "-p", "true"]