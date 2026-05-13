# Configuration file for CLSS Training
# This file is sourced by run_train.slurm for parameter management
# Modify this file to change training parameters without editing the SLURM script

# ============================================================================
# WORKING DIRECTORY & ENVIRONMENT
# ============================================================================
export WORKING_DIR="/media02/ddien02/thanhxuan217/CLSS"
export CONDA_PATH="/media02/ddien02/datnd/miniconda3/etc/profile.d/conda.sh"
export CONDA_ENV="/media02/ddien02/thanhxuan217/envs/sikubert"

# ============================================================================
# TRAINING CONFIG
# ============================================================================
# YAML config file (relative to WORKING_DIR)
export TRAIN_CONFIG="config/sino_nom_doc_cl.yaml"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
export TRANSFORMER_MODEL="SIKU-BERT/sikubert"
# Other options:
#   bert-base-chinese
#   hfl/chinese-roberta-wwm-ext
#   hfl/chinese-roberta-wwm-ext-large

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
# These are logged for reference — actual values are set in the YAML config
export LEARNING_RATE="5e-5"
export BATCH_SIZE=4
export GRADIENT_ACCUMULATION=4
export MAX_EPOCHS=10
export SEED=42

# ============================================================================
# CRF & MODEL ARCHITECTURE
# ============================================================================
export USE_CRF="true"
export HIDDEN_SIZE=256
export WORD_DROPOUT=0.1
export TEMPERATURE=0.3

# ============================================================================
# COOPERATIVE LEARNING
# ============================================================================
export MULTI_VIEW="true"
export DISTILL_POSTERIOR="true"
export INTERPOLATION=0.5

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================
export TARGET_DIR="resources/taggers"
export MODEL_NAME="${TRANSFORMER_MODEL}_${MAX_EPOCHS}epoch_${BATCH_SIZE}batch_${GRADIENT_ACCUMULATION}accumulate_${LEARNING_RATE}lr_sino_nom_punct_crf_multiview_posterior_minhash_rag"
export LOG_DIR="${WORKING_DIR}/logs"
