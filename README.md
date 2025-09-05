## Navigating Chemical-Linguistic Sharing Space with Heterogeneous Molecular Encoding

<div align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2412.20888-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.20888)
[![Model Checkpoints](https://img.shields.io/badge/Model-Checkpoints-blue)](https://huggingface.co/collections/GreatCaptainNemo/hme-checkpoints-6772a4b9d3a8d679c97f3bc3)
[![Dataset on Zenodo](https://img.shields.io/badge/Dataset-Zenodo-blue.svg?logo=zenodo)](https://doi.org/10.5281/zenodo.16963804)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](./LICENSE)

</div>

This repository contains the official implementation for the paper **"Navigating Chemical-Linguistic Sharing Space with Heterogeneous Molecular Encoding"**.


---

## 📖 Getting Started: From Setup to Inference

Follow this checklist to get HME running.

### Installation

First, clone the repository and install the `hme` package. This command will also install all required dependencies from `pyproject.toml`.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/HME.git
cd HME

# 2. Create and activate a conda environment
conda create -n hme python=3.10
conda activate hme

# 3. Install the package in editable mode
pip install -e .
```

### Download Datasets

Our datasets are hosted on Zenodo. Download and extract them into a `datasets/` directory.

```bash
# 1. Create the target directory
mkdir -p datasets

# 2. Download the data archive from Zenodo
wget -O datasets/data.zip "https://zenodo.org/records/16963804/files/data.zip?download=1"

# 3. Unzip the archive
unzip datasets/data.zip -d datasets/
rm datasets/data.zip # Clean up the zip file
```

### Preprocess Data

Use the following script to preprocess the downloaded data. If you don’t want to retrain the model, you can skip the commented-out parts, since some preprocessing steps are unnecessary in that case.

```python
from hme.preprocess_mol import get_2d3d_tensors
from hme.preprocess_prot import preprocess_crossdocked
get_2d3d_tensors('./datasets/property_qa_test_2.json.cfm')
# get_2d3d_tensors('./datasets/property_qa_train_2.json.cfm')
# get_2d3d_tensors('./datasets/pretrain.json.cfm')
# preprocess_crossdocked('./datasets/crossdocked_pocket10_train.json')
```

This will generate `.pt` feature files alongside your `.json` files.

### Download Models

We provide a script to download all HME adapter checkpoints and the Llama-3 base model from Hugging Face. You can comment out certain parts of the shell script to download only a subset of the models.

```bash
mkdir -p checkpoints
cd ./scripts
bash download_models.sh
```
**Note on Llama-3**: This step requires gated access to `meta-llama/Meta-Llama-3-8B-Instruct`. Please ensure you have requested access on its [Hugging Face page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and are logged in via `huggingface-cli login`.

### Merge Model Weights

We provide a script to easily merge the models. Since the Llama-3 license only allows us to release adapter weights, you’ll need to run the merge to obtain the full model.

You can comment out certain parts of the shell script to download only a subset of the models.

```bash
bash merge_models.sh
```

This will create several downstream models in the `checkpoints/` directory.

### Run Evaluation

You are now ready to run inference. Use the provided scripts in the `scripts/` directory.

Example: Evaluate on Molecular Captioning
```bash
bash eval_captioning.sh
```
Predictions will be saved in the `results/` directory. You can then use our metric scripts to score the output:
```bash
python -m hme.metrics.mol2text_metrics --prediction_file ./results/captioning_predictions.json
```

---

## 📖 How to Train

For detailed instructions on preparing data for training and running the training scripts, please refer to our comprehensive guide:

➡️ **[TRAINING.md](TRAINING.md)**

---


## Citation

If you find our work useful, please cite our paper:
```bibtex
@article{lv2024navigating,
  title={Navigating Chemical-Linguistic Sharing Space with Heterogeneous Molecular Encoding},
  author={Lv, Liuzhenghao and Li, Hao and Wang, Yu and Yan, Zhiyuan and Chen, Zijun and Lin, Zongying and Yuan, Li and Tian, Yonghong},
  journal={arXiv preprint arXiv:2412.20888},
  year={2024}
}
```

## Acknowledgements
Our work builds upon several fantastic open-source projects. We thank the authors of LLaVa, PS-VAE, Uni-Mol, and the various data sources used. A full list of acknowledgements can be found in our paper.s