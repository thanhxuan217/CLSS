#!/bin/bash
#SBATCH --job-name=clss-train                # Job name
#SBATCH --output=logs/slurm_%j.out           # Output file
#SBATCH --error=logs/slurm_%j.err            # Error file
#SBATCH --partition=batch                    # Partition name
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --cpus-per-task=16                   # CPUs per task
#SBATCH --gres=gpu:1                         # Number of GPUs
#SBATCH --mem=16G                            # Memory per node
#SBATCH --time=48:00:00                      # Time limit

set -euo pipefail

# ============================================================================
# LOAD CONFIGURATION
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "${SCRIPT_DIR}/config_train.slurm" ]]; then
    echo "ERROR: config_train.slurm not found in ${SCRIPT_DIR}"
    exit 1
fi

source "${SCRIPT_DIR}/config_train.slurm"

# Ensure directories exist
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

# ============================================================================
# SETUP ENVIRONMENT
# ============================================================================

echo "=========================================="
echo "CLSS Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Activate conda environment
source "${CONDA_PATH}"
conda activate "${CONDA_ENV}"

echo "Python: $(which python) ($(python --version 2>&1))"

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Print GPU info
if command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    echo "GPU:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
fi

echo ""
echo "Config: ${TRAIN_CONFIG}"
echo "Transformer: ${TRANSFORMER_MODEL}"
echo "Output: ${OUT_DIR}"
echo "=========================================="

# ============================================================================
# RUN TRAINING
# ============================================================================

cd "${WORKING_DIR}" || { echo "ERROR: Cannot cd to ${WORKING_DIR}"; exit 1; }

echo ""
echo "[Step 1/2] Training..."
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config "${TRAIN_CONFIG}" \
    --transformer_model "${TRANSFORMER_MODEL}" \
    --out_dir "${OUT_DIR}"

# ============================================================================
# RUN EVALUATION
# ============================================================================

echo ""
echo "[Step 2/2] Evaluating on test set..."
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config "${TRAIN_CONFIG}" \
    --test \
    --transformer_model "${TRANSFORMER_MODEL}" \
    --out_dir "${OUT_DIR}"

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "Results saved to: ${OUT_DIR}"
echo "=========================================="
