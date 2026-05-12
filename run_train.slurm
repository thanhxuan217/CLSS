#!/bin/bash
#SBATCH --job-name=clss-train                # Job name
#SBATCH --output=logs/slurm_%j.out           # Output file
#SBATCH --error=logs/slurm_%j.err            # Error file
#SBATCH --partition=batch                    # Partition name
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --cpus-per-task=16                   # CPUs per task
#SBATCH --gres=gpu:1                         # Number of GPUs
#SBATCH --mem=16G                            # Memory per node (max 64GB per server rules)
#SBATCH --time=48:00:00                      # Time limit (max 48h per server rules)

set -euo pipefail

# ============================================================================
# LOAD CONFIGURATION
# ============================================================================
# Source the configuration file for parameters
if [[ ! -f config_train.slurm ]]; then
    echo "ERROR: config_train.slurm not found in $(pwd)"
    exit 1
fi

source config_train.slurm

# Set working directory
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# ============================================================================
# SETUP ENVIRONMENT
# ============================================================================

echo "=========================================="
echo "CLSS Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: ${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}"
echo "Start Time: $(date)"
echo "=========================================="

# Load required modules (adjust based on your cluster)
# module load python/3.11
# module load cuda/11.8
# module load cudnn/8.6

# Activate conda environment
echo "Activating conda environment..."
source "${CONDA_PATH}"
conda activate "${CONDA_ENV}"

echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================

# Print GPU information
echo ""
echo "GPU Information:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
else
    echo "nvidia-smi not found"
fi
echo ""

echo ""
echo "=========================================="
echo "TRAINING CONFIGURATION"
echo "=========================================="
echo "Config File: $TRAIN_CONFIG"
echo "Transformer: $TRANSFORMER_MODEL"
echo ""
echo "Learning Rate: $LEARNING_RATE"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "Effective Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "Max Epochs: $MAX_EPOCHS"
echo "Seed: $SEED"
echo ""
echo "Use CRF: $USE_CRF"
echo "Hidden Size: $HIDDEN_SIZE"
echo "Word Dropout: $WORD_DROPOUT"
echo "Temperature: $TEMPERATURE"
echo ""
echo "Multi-view Training: $MULTI_VIEW"
echo "Distill Posterior: $DISTILL_POSTERIOR"
echo "Interpolation: $INTERPOLATION"
echo ""
echo "Target Dir: $TARGET_DIR"
echo "Model Name: $MODEL_NAME"
echo "=========================================="
echo ""

# ============================================================================
# RUN TRAINING
# ============================================================================

cd "${WORKING_DIR}" || { echo "ERROR: Cannot cd to ${WORKING_DIR}"; exit 1; }

echo "Starting training with config: ${TRAIN_CONFIG}"
echo ""

CUDA_VISIBLE_DEVICES=0 python train.py --config "${TRAIN_CONFIG}"

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "Model saved to: ${TARGET_DIR}/${MODEL_NAME}"
echo "=========================================="
