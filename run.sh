#!/bin/bash

# ==========================================================
# 配置
# ==========================================================
NUM_PARALLEL=2
AVAILABLE_GPUS=(3) # 显卡 ID
BASE_PORT=9000     # 起始端口
VLLM_PORT=8000
UE_EXEC="../simworld-server/SimWorld.sh" # 更新后的相对路径
CONDA_ENV="simworld"

# 初始化 Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# 创建日志目录
mkdir -p logs

echo "=========================================================="
echo "Starting Parallel Instances: Pure Host Mode"
echo "UE Path: $UE_EXEC"
echo "=========================================================="

# 预清理：杀死旧的进程
echo "Cleaning up old SimWorld processes..."
pkill -f "SimWorld-Linux-Shipping" || true
sleep 2

for ((i = 0; i < NUM_PARALLEL; i++)); do
  CURRENT_PORT=$((BASE_PORT + i))
  GPU_ID=${AVAILABLE_GPUS[$((i % ${#AVAILABLE_GPUS[@]}))]}

  echo "[Instance $i] GPU $GPU_ID -> Port $CURRENT_PORT"

  # 直接使用相对路径启动 UE
  # -RemoteControlHttpPort: 覆盖默认 9000 端口
  CUDA_VISIBLE_DEVICES=$GPU_ID $UE_EXEC \
    -RenderOffScreen \
    -Adapter=0 \
    -Vulkan \
    -RemoteControlHttpPort=$CURRENT_PORT \
    -nosound \
    -messaging >"logs/ue_inst_$i.log" 2>&1 &

  echo "Waiting for UE instance $i (Port $CURRENT_PORT) to initialize..."
  sleep 25

  # 启动 Python 进程
  python epoch_test.py \
    --round "$i" \
    --ue_port "$CURRENT_PORT" \
    --vllm_port "$VLLM_PORT" >"logs/py_inst_$i.log" 2>&1 &

  echo "Instance $i started."
  sleep 5
done

echo "----------------------------------------------------------"
echo "All processes started. Use 'jobs' to see background tasks."
echo "UE logs: tail -f logs/ue_inst_0.log"
echo "----------------------------------------------------------"

wait
