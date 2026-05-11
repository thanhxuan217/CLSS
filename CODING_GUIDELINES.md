# Coding Guidelines & Important Information

This document contains important rules and guidelines for developing, training, and evaluating models in the Sino-Nom Sentence Segmentation and Punctuation project. Please read and follow these instructions carefully when contributing or writing code.

## 1. Project Directory Structure
Maintain the following standard directory structure:
- `data/`: Contains datasets partitioned into `train`, `val`, and `test` subdirectories (Parquet format).
- `logs/`: SLURM logs and training progress logs.
- `models/`: Saved model checkpoints and final models.
- `outputs/`: Evaluation results (JSON reports) and other output files.

## 2. Dataset and Labels
- **Data Format:** Input data must be in Parquet format to support efficient streaming. Each row should contain a `text` column (string or list of characters) and a `labels` column (list of labels).
- **Punctuation Labels:** `O` (no punctuation), `，`, `。`, `：`, `、`, `；`, `？`, `！`.

## 3. Training Paradigm
- **QLoRA Finetuning:** Because the dataset is extremely large, always prioritize using QLoRA (4-bit quantization) when fine-tuning the models to optimize VRAM and computational resources.
- **Hugging Face Trainer:** Prioritize using the default Hugging Face `Trainer` API for training loops instead of writing manual loops, as it provides stability, distributed training support, and built-in features.
- **Step-based Training:** Train by `steps` instead of `epochs` due to the large dataset size.
- **Validation Subset:** Evaluate on a smaller, representative subset of the validation data rather than the entire validation set to save time during training steps.
- **Mandatory Training Features:** Any training code must implement and allow configuration of:
  - **Resume from Checkpoint:** Training must be able to resume seamlessly from saved checkpoints.
  - **Early Stopping:** Implement early stopping based on the validation metric.
  - **Save and Eval Steps:** Allow configuring `save_steps` and `eval_steps`.

## 4. Evaluation Metrics
- **Ignore 'O' Label:** For the punctuation task, the `O` label (representing no punctuation) must be ignored when calculating metrics to prevent skewed, overly optimistic results.
- **F1 Scores:** Always calculate and report both **Micro F1** and **Macro F1** scores to provide a comprehensive view of model performance across both balanced and imbalanced classes.

## 5. SLURM Usage
- Use provided SLURM scripts (`.slurm`) to run training and evaluation jobs on the server.
- **Config Files:** Whenever a new SLURM script (e.g., `run_task.slurm`) is written, a corresponding configuration file (e.g., `config_task.slurm`) MUST be created to store all environment variables and hyperparameter settings for that script.
