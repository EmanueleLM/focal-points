# Start from a base image with GPU support
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Create a new folder, grant rwx to everyone (inside container)
RUN mkdir -p /workspace && chmod -R 777 /workspace

# Switch into /workspace
WORKDIR /workspace

# Insert the huggingface-cli token
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Insert the huggingface-cli token
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Point all relevant env‐vars to that folder
RUN mkdir -p /cache /logs /results && chmod -R 777 /cache /logs /results

# Build essential tools for compiling
RUN apt-get update && \
  apt-get install -y --no-install-recommends build-essential && \
  rm -rf /var/lib/apt/lists/*

# Build essential tools for compiling
RUN apt-get update && \
  apt-get install -y --no-install-recommends build-essential && \
  rm -rf /var/lib/apt/lists/*

ENV AUDIOCRAFT_CACHE_DIR=/models \
  TORCH_HOME=/models \
  HF_HOME=/models \
  HF_DATASETS_CACHE=/models/datasets \
  HF_METRICS_CACHE=/models/metrics \
  HF_MODULES_CACHE=/models/modules \
  HUGGINGFACE_HUB_CACHE=/models/huggingface \
  TRITON_CACHE_DIR=/models/triton_cache \
  XDG_CACHE_HOME=/models/.cache \
  TORCHINDUCTOR_CACHE_DIR=/models/torch_inductor_cache \
  PYTHONUNBUFFERED=1 \
  NCCL_SHM_DISABLE=1 \
  OMP_NUM_THREADS=1

# Install the Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the code
COPY . .

# Make sure everything under /workspace is writable
RUN chmod -R a+rw /workspace

# Launch the script
ENTRYPOINT [ \
  "python", \
  "main.py", \
  "--model", \
  "meta-llama/Llama-3.3-70B-Instruct", \
  "--quantization", \
  "8bit" \
  ]
