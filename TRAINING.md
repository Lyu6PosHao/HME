This guide provides detailed instructions for preparing data and running the training scripts to reproduce the results from our paper.

## Prerequisites

Before proceeding, please ensure you have completed the first three steps from the main [README.md](README.md):
1.  **Installation**: The `hme` package and all dependencies are installed.
2.  **Dataset Download**: All raw datasets are downloaded and extracted into the `datasets/` directory.
3.  **Data Preprocessing Setup**: Here, the full data preprocessing needs to be performed if you haven’t done so already.

    ```python
    from hme.preprocess_mol import get_2d3d_tensors
    from hme.preprocess_prot import preprocess_crossdocked
    get_2d3d_tensors('./property_qa_test_2.json.cfm')
    get_2d3d_tensors('./property_qa_train_2.json.cfm')
    get_2d3d_tensors('./pretrain.json.cfm')
    preprocess_crossdocked('/crossdocked_pocket10_train.json')
    ```

## Running the Training Scripts

Our training process is staged. You typically start with a base model, perform pre-training, and then fine-tune on specific tasks. All training scripts are located in the `scripts/` directory and are designed for multi-GPU training using DeepSpeed.

### Important: Before You Run

*   **Edit the Scripts**: Before launching any training script, you **must** open the `.sh` file and configure the paths (e.g., `base_model_path`, `data_path`, `output_dir`).
*   **Hardware**: The provided scripts are configured for a multi-GPU setup (e.g., 8 GPUs). You may need to adjust the `--num_gpus` flag and the DeepSpeed config (`scripts/ds_zero2_no_offload.json`) based on your hardware.

### Training Workflow

The training follows the same dependencies as the model merging process.

#### - Pre-training

This is the first training stage, which aligns molecular features with the LLM.

1.  Configure: Edit `scripts/run_zero2_comprehension-pretrain.sh`.
    *   Set `base_model_path` to your Llama-3 model path.
    *   Set `data_path` to your preprocessed pre-training data file.
2.  Launch:
    ```bash
    bash run_zero2_comprehension-pretrain.sh
    ```

#### - Second-Stage (S2) Pre-training (Optional)

This stage continues the pre-training with the same dataset.

1.  Configure: Edit `scripts/run_zero2_comprehension-pretrain-s2.sh`.
    *   Set `base_model_path` to the checkpoint from Stage 1.
2.  Launch:
    ```bash
    bash run_zero2_comprehension-pretrain-s2.sh
    ```

#### - Fine-tuning on Downstream Tasks

After pre-training, you can fine-tune the model on specific tasks.

Example: Fine-tuning for General QA

1.  Configure: Edit `scripts/run_zero2_general-qa.sh`.
    *   Set `base_model_path` to your Stage 1 pre-trained checkpoint.
    *   Set `data_path` to the general QA training data.
2.  Launch:
    ```bash
    bash scripts/run_zero2_general-qa.sh
    ```

Other Fine-tuning Tasks:
Follow the same "Configure -> Launch" pattern for other tasks:
*   `run_zero2_captioning.sh` (uses the Stage 2 checkpoint as base)
*   `run_zero2_property-qa-1.sh` (uses Stage 1)
*   `run_zero2_property-qa-2.sh` (uses Stage 1)
*   ... and so on for all other training scripts.

The ligand generation branch follows a similar, independent training path starting from the base Llama-3 model.

## After Training
After training, you will find the trained adapter weights in the checkpoints folder, along with the merged full model (if the parameter merge_when_finish is set to True). The full model can be used directly for inference.