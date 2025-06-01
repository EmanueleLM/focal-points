# Start from a base image with GPU support
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Create a writable cache directory for AudioCraft / HuggingFace, and the output directories
RUN mkdir -p /models /logs /results && chmod -R 777 /models /logs /results

# Point all relevant env‐vars to that folder
ENV AUDIOCRAFT_CACHE_DIR=/models \
    TORCH_HOME=/models \
    HF_HOME=/models \
    HF_DATASETS_CACHE=/models/datasets \
    HF_METRICS_CACHE=/models/metrics \
    HF_MODULES_CACHE=/models/modules  \
    HUGGINGFACE_HUB_CACHE=/models/huggingface \
    PYTHONUNBUFFERED=1 \
    NCCL_SHM_DISABLE=1 \
    OMP_NUM_THREADS=1

# Install the Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the code
COPY . .

# Launch the script
ENTRYPOINT ["/bin/bash", "-c", "jupyter nbconvert --to script schelling.ipynb && python schelling.py"]
