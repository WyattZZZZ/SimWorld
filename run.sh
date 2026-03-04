#!/bin/bash

cleanup_ue() {
  echo "[$(date +%T)] Cleaning UE and Python Processes..."
  pkill -9 -f "SimWorld"
  fuser -k 9000/tcp >/dev/null 2>&1
  sleep 2
}

# ==========================================================
# 1. Environment & Conda Setup
# ==========================================================
# Define the path to your conda initialization script
CONDA_PATH="$HOME/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="simworld"

# Initialize conda for this shell session
if [ -f "$CONDA_PATH" ]; then
  source "$CONDA_PATH"
  conda activate "$ENV_NAME"
  echo "[$(date +'%H:%M:%S')] Conda environment '$ENV_NAME' activated."
else
  echo "ERROR: Conda initialization script not found at $CONDA_PATH"
  exit 1
fi

# Set your Python executable path (as backup/verification)
PYTHON_EXEC=$(which python)
echo "[$(date +'%H:%M:%S')] Using Python at: $PYTHON_EXEC"

# Global Configs
export PROJECT_ROOT=$(pwd)
mkdir -p logs
mkdir -p records

TOTAL_ROUNDS=1
UE_PORT=9000
VLLM_PORT=8000
UE_GPU_ID=1

echo "=========================================================="
echo "EXPERIMENTAL RUN STARTED AT: $(date)"
echo "=========================================================="

# ==========================================================
# 2. Main Loop
# ==========================================================
for ((i = 1; i <= $TOTAL_ROUNDS; i++)); do
  cleanup_ue
  echo "[$(date +'%H:%M:%S')] --- Starting ROUND $i ---"

  # A. Launch UE5 Backend
  echo "[$(date +'%H:%M:%S')] Launching UE5 Engine on GPU $UE_GPU_ID..."
  CUDA_VISIBLE_DEVICES=$UE_GPU_ID ../simworld-server/SimWorld.sh -RenderOffScreen >logs/ue_round_$i.log 2>&1 &
  UE_PID=$!

  # B. Wait for Initialization
  # Increased to 60s to ensure the UnrealCV plugin starts listening
  echo "[$(date +'%H:%M:%S')] UE5 PID: $UE_PID. Waiting 40s for map and plugin..."
  sleep 40

  # C. Port Connection Check
  # Verify port 9000 is open before wasting 40 minutes
  if ! ss -tuln | grep -q ":$UE_PORT "; then
    echo "[$(date +'%H:%M:%S')] ERROR: UnrealCV port $UE_PORT is not open. Skipping Round $i."
    kill -9 $UE_PID >/dev/null 2>&1
    pkill -f "SimWorld"
    continue
  fi

  # D. Execute Python Experiment
  # This will run for approximately 40 minutes
  echo "[$(date +'%H:%M:%S')] Python started. Running 40-minute logic..."
  $PYTHON_EXEC epoch_test.py --round $i --ue_port $UE_PORT --vllm_port $VLLM_PORT

  PY_EXIT_CODE=$?
  echo "[$(date +'%H:%M:%S')] Python process exited with code $PY_EXIT_CODE."

  # E. Hard Cleanup
  # Kill UE5 to reset world state and flush VRAM for next round
  echo "[$(date +'%H:%M:%S')] Cleaning up processes..."
  kill -9 $UE_PID >/dev/null 2>&1
  pkill -f "SimWorld"

  sleep 5
  echo "----------------------------------------------------------"
done

echo "=========================================================="
echo "ALL EXPERIMENTS FINISHED AT: $(date)"
echo "=========================================================="
