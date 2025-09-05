#!/bin/bash

# =====================================================================================
# HME Model Downloader
# =====================================================================================
# This script downloads all necessary model weights for the HME project.
# It uses the `hme.download` utility for each repository.
#
# Usage:
#   - To download all models: bash scripts/download_models.sh
#   - To download a specific model, uncomment the desired lines below.
# =====================================================================================

# Ensure the script exits if any command fails
set -e

# Define the root directory for checkpoints
CHECKPOINTS_DIR="../checkpoints"
echo "All models will be downloaded to the '${CHECKPOINTS_DIR}' directory."

# --- Helper Function ---
# A small function to reduce code duplication.
download() {
    local repo_id="$1"
    local local_name="$2"
    echo "-----------------------------------------------------"
    python -m hme.download_models --repo_id "${repo_id}" --local_dir "${CHECKPOINTS_DIR}/${local_name}"
}

# --- HME Adapter Checkpoints ---
echo "[1/9] Downloading: HME_comprehension-pretrain"
download "GreatCaptainNemo/HME_comprehension-pretrain" "HME_comprehension-pretrain"

echo "[2/9] Downloading: HME_comprehension-pretrain-s2"
download "GreatCaptainNemo/HME_comprehension-pretrain-s2" "HME_comprehension-pretrain-s2"

echo "[3/9] Downloading: HME_captioning"
download "GreatCaptainNemo/HME_captioning" "HME_captioning"

echo "[4/9] Downloading: HME_general-qa"
download "GreatCaptainNemo/HME_general-qa" "HME_general-qa"

echo "[5/9] Downloading: HME_property-qa-1"
download "GreatCaptainNemo/HME_property-qa-1" "HME_property-qa-1"

echo "[6/9] Downloading: HME_property-qa-2"
download "GreatCaptainNemo/HME_property-qa-2" "HME_property-qa-2"

echo "[7/9] Downloading: HME_pocket-based-ligand-generation_pretrain"
download "GreatCaptainNemo/HME_pocket-based-ligand-generation_pretrain" "HME_pocket-based-ligand-generation_pretrain"

echo "[8/9] Downloading: HME_pocket-based-ligand-generation"
download "GreatCaptainNemo/HME_pocket-based-ligand-generation" "HME_pocket-based-ligand-generation"


# --- Base Model (Requires Gated Access) ---
# NOTE: This download will fail if you have not been granted access on Hugging Face.
# You must also be logged in via `huggingface-cli login`.
echo "[9/9] Downloading: Meta-Llama-3-8B-Instruct (Requires Access)"
echo "        Please ensure you have requested access at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"
download "meta-llama/Meta-Llama-3-8B-Instruct" "Meta-Llama-3-8B-Instruct"


echo "-----------------------------------------------------"
echo "All downloads complete."