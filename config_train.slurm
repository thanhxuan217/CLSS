# Configuration file for CLSS Training on SLURM
# Sourced by run_train.slurm — chỉ chứa biến THỰC SỰ được dùng trong script
#
# Lưu ý: Các hyperparameters (learning_rate, batch_size, epochs, ...)
# được cấu hình trong YAML config, KHÔNG phải ở đây.

# ============================================================================
# WORKING DIRECTORY & ENVIRONMENT
# ============================================================================
export WORKING_DIR="/media02/ddien02/thanhxuan217/main_src"
export CONDA_PATH="/media02/ddien02/datnd/miniconda3/etc/profile.d/conda.sh"
export CONDA_ENV="/media02/ddien02/thanhxuan217/envs/sikubert"

# ============================================================================
# TRAINING CONFIG
# ============================================================================
# YAML config file (relative to WORKING_DIR) — mọi hyperparameters nằm trong đây
# Đổi thành "config/sino_nom_doc_cl.yaml" nếu muốn train full fine-tune (không QLoRA)
export TRAIN_CONFIG="config/sino_nom_doc_cl_qlora.yaml"

# Override transformer model path (local directory hoặc HuggingFace name)
# Nếu server offline: đổi thành đường dẫn local, ví dụ "${WORKING_DIR}/models/sikubert"
export TRANSFORMER_MODEL="${WORKING_DIR}/pretrained/sikubert"

# ============================================================================
# OUTPUT
# ============================================================================
export OUT_DIR="${WORKING_DIR}/output"
export LOG_DIR="${WORKING_DIR}/logs"
