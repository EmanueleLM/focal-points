#!/bin/bash
#SBATCH --job-name=lefocal
#SBATCH --output=/private/kraus-lab/idoa/slurm/lefocal//lefocal_%j.out
#SBATCH --error=/private/kraus-lab/idoa/slurm/lefocal//lefocal_%j.err
#SBATCH --partition=p_b200_kraus
#SBATCH --account=ug_kraus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --mail-user=idoah106@gmail.com
#SBATCH --mail-type=ALL

set -euo pipefail

# Argument forwarding
DEFAULT_EXPERIMENT_ARGS=(-m all -q 8bit -x 4096)
EXPERIMENT_ARGS=("${DEFAULT_EXPERIMENT_ARGS[@]}")

if [[ $# -gt 0 ]]; then
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: sbatch run.sh [experiments.sh args]

Examples:
  sbatch run.sh -m gpt-5.4 -r low -q None -d "amsterdam amsterdam-instruct-all-features"
  sbatch run.sh -m all -q 8bit -x 4096

If no args are provided, defaults are used:
  -m all -q 8bit -x 4096
EOF
    exit 0
  fi

  EXPERIMENT_ARGS=("$@")
fi

# Timestamp for per-run log files (e.g. 2711:1810)
TS=$(date +"%d%m:%H%M")

# DYNAMIC
OUT_FOLDER_NAME=out_lefocal

# STATIC
DockerName=slurm-job-$SLURM_JOB_ID
IMAGE=ido106/lefocal:latest
CACHEDIR=/private/kraus-lab/idoa/cache
HF_HUB_DIR=/private/kraus-lab/shared/cache
WORK=/private/kraus-lab/idoa/lefocal
HOST_USER_NAME="$(id -un)"

CODE="$WORK/workspace"
OUT="$WORK/$OUT_FOLDER_NAME"

# GPU log (live nvidia-smi text, always latest snapshot)
GPU_LOG="$OUT/gpu/gpu_${SLURM_JOB_ID}.txt"

# Pull the latest image tag
docker pull "$IMAGE"

# Make sure host directories exist (safe if they already exist)
mkdir -p "$OUT" "$OUT/results" "$OUT/logs" "$OUT/images" "$OUT/plots" "$OUT/gpu" "$OUT/stdout" "$OUT/plots"

# ensure and update code to latest main
if [ -d "$CODE/.git" ]; then
  git -C "$CODE" fetch origin
  git -C "$CODE" reset --hard origin/main
else
  rm -rf "$CODE"
  git clone --depth=1 https://github.com/EmanueleLM/focal-points.git "$CODE"
fi

# Start live GPU monitor (overwrites file each time)
(
  while true; do
    {
      echo "===== $(date '+%Y-%m-%d %H:%M:%S') ====="
      nvidia-smi
    } > "$GPU_LOG"
    sleep 1
  done
) & NVIDIA_PID=$!

# Run
docker run --rm \
  --name "$DockerName" \
  --env-file "$CODE/.env" \
  --gpus "device=${SLURM_JOB_GPUS}" \
  -e USER="$HOST_USER_NAME" \
  -e LOGNAME="$HOST_USER_NAME" \
  -v "$CODE":/workspace \
  -v "$HF_HUB_DIR":/hf \
  -v "$CACHEDIR":/cache \
  -v "$OUT/results":/workspace/results \
  -v "$OUT/logs":/workspace/logs \
  -v "$OUT/images":/workspace/images \
  -v "$OUT/plots":/workspace/plots \
  "$IMAGE" \
  "${EXPERIMENT_ARGS[@]}" \
  > >(tee -a "$OUT/stdout/lefocal_${SLURM_JOB_ID}_$TS.out") \
  2> >(tee -a "$OUT/stdout/lefocal_${SLURM_JOB_ID}_$TS.err" >&2)

# Kill the monitor once Docker finishes
kill "$NVIDIA_PID" 2>/dev/null