This guide provides detailed instructions for preparing data and running the training scripts to reproduce the results from our paper.

## Hardware

It is recommended to train on a GPU with at least 48GB of memory. To speed up training, we suggest using a multi-GPU setup, preferably with NVLink support, and GPUs that support Flash Attention.



---

## Prerequisites

Before proceeding, ensure you have completed the first three steps from the main [README.md](README.md):

1. **Installation**: Install the `hme` package and all dependencies.  
2. **Dataset Download**: Download and extract all raw datasets into the `datasets/` directory.  
3. **Data Preprocessing Setup**: Perform the full data preprocessing if you haven’t done so already:

```python
from hme.preprocess_mol import get_2d3d_tensors
from hme.preprocess_prot import preprocess_crossdocked

get_2d3d_tensors('./datasets/property_qa_test_2.json.cfm')
get_2d3d_tensors('./datasets/property_qa_train_2.json.cfm')
get_2d3d_tensors('./datasets/pretrain.json.cfm')
preprocess_crossdocked('./datasets/crossdocked_pocket10_train.json')
```

---

## Running the Training Scripts

All training scripts are in the `scripts/` directory and designed for multi-GPU training with DeepSpeed.

### Important: Before You Run

- **Edit the Scripts**: Review the `.sh` files and set the paths (`base_model_path`, `data_path`, `output_dir`). **Default paths are provided** for convenience.  
- **Hardware**: Scripts are configured for multi-GPU setups (e.g., 8 GPUs). Adjust the `--num_gpus` flag and DeepSpeed config (`scripts/ds_zero2_no_offload.json`) according to your hardware.

---

## Training Workflow

The training follows the same dependencies as the model merging process.

### 1. Pre-training (Stage 1)

Align molecular features with the LLM.

1. Configure: Edit `scripts/run_zero2_comprehension-pretrain.sh`.  
2. Launch:

```bash
bash scripts/run_zero2_comprehension-pretrain.sh
```

---

### 2. Second-Stage (S2) Pre-training (Optional)

Continue pre-training with the same dataset.

1. Configure: Edit `scripts/run_zero2_comprehension-pretrain-s2.sh` and set `base_model_path` to the Stage 1 checkpoint.  
2. Launch:

```bash
bash scripts/run_zero2_comprehension-pretrain-s2.sh
```

---

### 3. Fine-tuning on Comprehension Tasks

Fine-tune the model on specific tasks. Follow the **Configure -> Launch** pattern.

#### Example: General QA

1. Configure: Edit `scripts/run_zero2_general-qa.sh` and set `base_model_path` to the Stage 1 checkpoint.  
2. Launch:

```bash
bash scripts/run_zero2_general-qa.sh
```

#### Other Fine-tuning Tasks

| Script | Base Checkpoint |
|--------|----------------|
| `run_zero2_captioning.sh` | Stage 2 checkpoint |
| `run_zero2_property-qa-1.sh` | Stage 1 checkpoint |
| `run_zero2_property-qa-2.sh` | Stage 1 checkpoint |
| … | … |

---

### 4. Pocket-based Ligand Generation Task


```bash
bash scripts/run_zero2_pocket-based-ligand-generation_pretrain.sh
bash scripts/run_zero2_pocket-based-ligand-generation.sh
```

---

### 5. Description-Based Molecular Generation Task

```bash
bash scripts/run_zero2_description-based-mol-gen_pretrain.sh
bash scripts/run_zero2_description-based-mol-gen.sh
```

---

## After Training

Trained adapter weights are saved in the `checkpoints` folder. If `merge_when_finish` is set to True, the merged full model will also be available for direct inference.
