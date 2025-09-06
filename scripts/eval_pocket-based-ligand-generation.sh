#!/bin/bash
set -e

# --- Configuration ---
# 1. Model and Task Settings
MODEL_NAME="HME_pocket-based-ligand-generation_merged"
TASK_TYPE="textprotein2frgsmi"
DATA_TYPE="1d,frg"

# 2. Path Settings (UPDATE THESE PATHS)
MODEL_PATH="../checkpoints/${MODEL_NAME}"
DATA_PATH="../datasets/CrossDocked2020/crossdocked_pocket10_generation.nn.json"
MOL_EMB_PATH="none"
PROTEIN_EMB_PATH="../datasets/CrossDocked2020/crossdocked_pocket10_test.pt"

# 3. GPU and Output Settings
GPU_ID=0
MAX_LENGTH=512

# --- Execution ---
# Auto-generate output path, including the '_nn' suffix from the original script
OUTPUT_DIR="../results/${MODEL_NAME}_${TASK_TYPE}"
OUTPUT_PATH="${OUTPUT_DIR}/result.jsonl"
mkdir -p "$OUTPUT_DIR"
cp "$0" "${OUTPUT_DIR}/" # Save a copy of the script

# Set CUDA device and launch inference
# Note: This task uses sampling (do_sample=True)
CUDA_VISIBLE_DEVICES=${GPU_ID} torchrun --nproc_per_node=1 --master_port=29504 infer.py \
    --model_name_or_path "${MODEL_PATH}" \
    --max_length "${MAX_LENGTH}" \
    --data_path "${DATA_PATH}" \
    --data_type "${DATA_TYPE}" \
    --task_type "${TASK_TYPE}" \
    --emb_dict_mol "${MOL_EMB_PATH}" \
    --emb_dict_protein "${PROTEIN_EMB_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --do_sample True \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 50 \

echo "Inference complete. Results saved to ${OUTPUT_PATH}"