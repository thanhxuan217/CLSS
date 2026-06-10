#!/bin/bash
#SBATCH --job-name=clss-train                # Job name
#SBATCH --output=logs/slurm_%j.out           # Output file
#SBATCH --error=logs/slurm_%j.err            # Error file
#SBATCH --partition=batch                    # Partition name
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --cpus-per-task=16                   # CPUs per task
#SBATCH --gres=gpu:1                         # Number of GPUs
#SBATCH --mem=32G                            # Memory per node
#SBATCH --time=48:00:00                      # Time limit

set -euo pipefail

# ============================================================================
# LOAD CONFIGURATION
# ============================================================================
source "config_train.slurm"

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

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ============================================================================
# RUN TRAINING
# ============================================================================
echo ""
echo "[Step 1/2] Training..."
python train.py \
    --config "${TRAIN_CONFIG}" \
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
