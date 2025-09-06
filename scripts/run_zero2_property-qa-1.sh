#!/bin/bash
set -e

# --- Environment Settings ---
export NCCL_P2P_LEVEL=NVL
export TOKENIZERS_PARALLELISN=True

# --- Configuration ---
# 1. Model and LoRA Settings
BASE_MODEL="HME_comprehension-pretrain_merged"
LORA_R=16
LORA_ALPHA=32
LORA_TARGETS="q_proj,v_proj,k_proj,o_proj,up_proj,gate_proj,down_proj"
MODULES_TO_SAVE="feature_fuser"

# 2. Data and Task Settings
TASK_TYPE="qa"
TASK_NAME_SUFFIX="property-qa-1" # Used for naming the output directory
DATA_TYPE="1d,2d,3d,frg"
TRAIN_DATA_PATH="../datasets/property_qa_train_1.json"
MOL_EMB_PATH="../datasets/pubchem_train_test.json.cfm.pt"
PROTEIN_EMB_PATH="none"

# 3. Training Hyperparameters
MAX_LENGTH=300
EPOCHS=5
TRAIN_BATCH_SIZE=4
GRAD_ACCUMULATION=1
LEARNING_RATE=5e-5

# 4. Infrastructure Settings
GPUS="0,1,2,3,4,5,6,7"
MASTER_PORT=29502

# --- Execution ---
# Set paths
BASE_MODEL_PATH="../checkpoints/${BASE_MODEL}"
OUTPUT_DIR="../checkpoints/HME_property-qa-1"
mkdir -p "$OUTPUT_DIR"
cp "$0" "${OUTPUT_DIR}/"

# Launch training with DeepSpeed
deepspeed --include "localhost:${GPUS}" --master_port ${MASTER_PORT} --module hme.run_clm \
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
    --save_total_limit 8 \
    --logging_steps 4 \
    --warmup_ratio 0.05 \
    --report_to "none"

echo "Training complete. Checkpoints saved to ${OUTPUT_DIR}"