#!/bin/bash
set -e

# ==============================================================================
# HME Model Merger (User-Configurable)
# ==============================================================================
# INSTRUCTIONS:
# 1. Set your Llama-3 base model path in `BASE_MODEL_PATH`.
# 2. Uncomment the `merge` commands for the models you wish to create.
#    Each merge operation is independent.
# ==============================================================================

# --- Configuration ---
BASE_MODEL_PATH="../checkpoints/Meta-Llama-3-8B-Instruct"
CHECKPOINTS_DIR="../checkpoints"

# --- Pre-flight Check ---
if [[ "${BASE_MODEL_PATH}" == "/path/to/your/"* || ! -d "${BASE_MODEL_PATH}" ]]; then
    echo "Error: Please update BASE_MODEL_PATH in this script to a valid directory." >&2
    exit 1
fi

# --- Core Merge Function ---
merge() {
    local base_model="$1"
    local adapter="$2"
    local task_type_arg=""
    [[ -n "$3" ]] && task_type_arg="--task_type $3"

    # Automatically derive output path from adapter name
    local output_model="${CHECKPOINTS_DIR}/$(basename ${adapter})_merged"

    echo "---"
    echo "Base:    $(basename ${base_model})"
    echo "Adapter: $(basename ${adapter})"
    echo "Output:  $(basename ${output_model})"

    python -m hme.merge \
        --base_model_path "${base_model}" \
        --adapter_path "${adapter}" \
        ${task_type_arg}
}

# ==============================================================================
# MERGE WORKFLOW
# Uncomment the models you need. Note the dependencies in the base model paths.
# ==============================================================================



merge "${BASE_MODEL_PATH}" \
      "${CHECKPOINTS_DIR}/HME_comprehension-pretrain"


merge "${CHECKPOINTS_DIR}/HME_comprehension-pretrain_merged" \
      "${CHECKPOINTS_DIR}/HME_comprehension-pretrain-s2"


merge "${CHECKPOINTS_DIR}/HME_comprehension-pretrain_merged" \
      "${CHECKPOINTS_DIR}/HME_general-qa"


merge "${CHECKPOINTS_DIR}/HME_comprehension-pretrain_merged" \
      "${CHECKPOINTS_DIR}/HME_property-qa-1"


merge "${CHECKPOINTS_DIR}/HME_comprehension-pretrain_merged" \
      "${CHECKPOINTS_DIR}/HME_property-qa-2"


merge "${CHECKPOINTS_DIR}/HME_comprehension-pretrain-s2_merged" \
      "${CHECKPOINTS_DIR}/HME_captioning"



merge "${BASE_MODEL_PATH}" \
      "${CHECKPOINTS_DIR}/HME_pocket-based-ligand-generation_pretrain" \
      "pdbbind_reg"


merge "${CHECKPOINTS_DIR}/HME_pocket-based-ligand-generation_pretrain_merged" \
      "${CHECKPOINTS_DIR}/HME_pocket-based-ligand-generation"


echo "---"
echo "Model merging process finished for all selected models."