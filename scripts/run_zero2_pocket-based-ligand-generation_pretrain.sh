#!/bin/bash
set -e

# --- Environment Settings ---
export NCCL_P2P_LEVEL=NVL
export TOKENIZERS_PARALLELISM=True

# --- Configuration ---
# 1. Model and LoRA Settings
BASE_MODEL="Meta-Llama-3-8B-Instruct"
LORA_R=32
LORA_ALPHA=64
LORA_TARGETS="q_proj,v_proj,k_proj,o_proj,up_proj,down_proj,gate_proj"
MODULES_TO_SAVE="feature_fuser,regression_head"

# 2. Data and Task Settings
TASK_TYPE="pdbbind_reg"
TASK_NAME_SUFFIX="pdbv2016_refined" # For naming the output directory
DATA_TYPE="1d,2d,3d"
TRAIN_DATA_PATH="/path/to/your/datasets/pdbbind_v2016/.../index_except-crossdockedtest_refined-train.json"
MOL_EMB_PATH="/path/to/your/datasets/pdbbind_v2016/.../index_except-crossdockedtest.json.cfm.pt"
PROTEIN_EMB_PATH="/path/to/your/datasets/pdbbind_v2016/.../index_except-crossdockedtest.protein-emb.pt"

# 3. Training Hyperparameters
MAX_LENGTH=300
EPOCHS=10
TRAIN_BATCH_SIZE=9
GRAD_ACCUMULATION=4
LEARNING_RATE=1e-5

# 4. Infrastructure Settings
GPUS="0,1,2,3,4,5,6,7"
MASTER_PORT=29501

# --- Execution ---
# Set paths
BASE_MODEL_PATH="/path/to/your/models/${BASE_MODEL}"
OUTPUT_DIR="../checkpoints/${BASE_MODEL}_${TASK_NAME_SUFFIX}_${TASK_TYPE}"
mkdir -p "$OUTPUT_DIR"
cp "$0" "${OUTPUT_DIR}/"

# Launch training with DeepSpeed
deepspeed --include "localhost:${GPUS}" --master_port ${MASTER_PORT} --module run_regression.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path "${BASE_MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --data_path "${TRAIN_DATA_PATH}" \
    --task_type "${TASK_TYPE}" \
    --data_type "${DATA_TYPE}" \
    --emb_dict_mol "${MOL_EMB_PATH}" \
    --emb_dict_protein "${PROTEIN_EMB_PATH}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_targets "${LORA_TARGETS}" \
    --modules_to_save "${MODULES_TO_SAVE}" \
    --merge_when_finished True \
    --max_length "${MAX_LENGTH}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUMULATION}" \
    --learning_rate "${LEARNING_RATE}" \
    --bf16 True \
    --do_train \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --logging_steps 4 \
    --warmup_ratio 0.1 \
    --report_to "none"

echo "Training complete. Checkpoints saved to ${OUTPUT_DIR}"