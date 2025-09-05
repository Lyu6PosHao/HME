import logging
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem.rdchem import BondDir, BondType, ChiralType
from torch_geometric.data import Data
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

from hme.configuration_llava import HMEConfig
from hme.modeling_llava import (
    HMEForConditionalGeneration,
    HMEForSequenceClassification,
    HMEForSequenceRegression,
)

logger = logging.getLogger(__name__)

# --- RDKit Enum to Integer Mappings ---
BOND_TYPE_MAP = {
    BondType.SINGLE: 0,
    BondType.DOUBLE: 1,
    BondType.TRIPLE: 2,
    BondType.AROMATIC: 3,
}
BOND_DIR_MAP = {BondDir.NONE: 0, BondDir.ENDUPRIGHT: 1, BondDir.ENDDOWNRIGHT: 2}
CHIRALITY_MAP = {
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_TETRAHEDRAL_CW: 1,
    ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    ChiralType.CHI_OTHER: 3,
}


def set_seed(seed: int = 42) -> None:
    """Sets the seed for generating random numbers for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_nb_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Returns the number of trainable parameters and the total number of parameters in the model.
    (Copied from Hugging Face PEFT library)
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        if param.__class__.__name__ == "Params4bit":
            num_bytes = (
                param.quant_storage.itemsize
                if hasattr(param, "quant_storage")
                else 1
            )
            num_params *= 2 * num_bytes
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return trainable_params, all_param


def print_trainable_parameters(model: nn.Module) -> None:
    """Prints the number of trainable parameters in the model."""
    trainable_params, all_param = get_nb_trainable_parameters(model)
    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )


def bond_dir(bond) -> int:
    return BOND_DIR_MAP[bond.GetBondDir()]


def bond_type(bond) -> int:
    return BOND_TYPE_MAP[bond.GetBondType()]


def atom_chiral(atom) -> int:
    return CHIRALITY_MAP[atom.GetChiralTag()]


def atom_to_feature(atom) -> List[int]:
    num = atom.GetAtomicNum() - 1
    if num == -1:  # Handle wildcard atom '*'
        num = 118
    return [num, atom_chiral(atom)]


def bond_to_feature(bond) -> List[int]:
    return [bond_type(bond), bond_dir(bond)]


def smiles2graph(smiles_string: str, device: torch.device) -> Data:
    """
    Converts a SMILES string to a PyTorch Geometric Data object.

    Parameters
    ----------
    smiles_string : str
        The input SMILES string.
    device : torch.device
        The device to place the resulting tensors on.

    Returns
    -------
    Data
        A PyG Data object representing the molecular graph.
    """
    mol = Chem.MolFromSmiles(smiles_string)
    atom_features_list = [atom_to_feature(atom) for atom in mol.GetAtoms()]
    x = np.array(atom_features_list, dtype=np.int64)

    if len(mol.GetBonds()) > 0:
        edges_list, edge_features_list = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_feature = bond_to_feature(bond)
            edges_list.extend([(i, j), (j, i)])
            edge_features_list.extend([edge_feature, edge_feature])
        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_attr = np.array(edge_features_list, dtype=np.int64)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, 2), dtype=np.int64)

    return Data(
        x=torch.as_tensor(x, device=device),
        edge_attr=torch.as_tensor(edge_attr, device=device),
        edge_index=torch.as_tensor(edge_index, device=device),
    )


def smart_tokenizer_and_embedding_resize(
    tokenizer, model, num_new_tokens: int
) -> None:
    """
    Resizes the tokenizer and model embeddings, initializing new embeddings with the average
    of the old ones. 
    """
    if num_new_tokens <= 0:
        if num_new_tokens < 0:
            raise ValueError("num_new_tokens must be non-negative")
        logging.info("No new tokens added.")
        return

    model.language_model.resize_token_embeddings(len(tokenizer))
    input_embeddings = model.language_model.get_input_embeddings().weight.data
    output_embeddings = model.language_model.get_output_embeddings().weight.data

    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
        dim=0, keepdim=True
    )

    input_embeddings[-num_new_tokens:] = input_embeddings_avg
    output_embeddings[-num_new_tokens:] = output_embeddings_avg
    logging.info(
        f"{num_new_tokens} new tokens added to the tokenizer; embed_tokens and lm_head resized."
    )


def get_frg_list(task_type:str) -> List[str]:
    """Loads a list of formatted fragment tokens from a vocabulary file."""
    from periodictable import elements
    MAX_VALENCE = {element.symbol: 10 for element in elements}
    import importlib.resources
    if task_type=='textfrg2smi':
        vocab_file = importlib.resources.files('hme.fragment_vocabs').joinpath('vocab_800_multi_obj_mol_design.txt')
    else:
        vocab_file = importlib.resources.files('hme.fragment_vocabs').joinpath('vocab_1000.txt')

    with vocab_file.open('r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f]
    logger.warning(
        f"Using vocab_file: {os.path.basename(vocab_file)} to load fragment list."
    )

    frg_list = []
    for line in vocab:
        parts = line.strip().split("\t")
        if len(parts) == 3:  # Skip header/config line
            token = parts[0]
            if token in MAX_VALENCE:  # Single atom fragment
                frg_list.append(f"<|[{token}]|>")
            else:  # Multi-atom fragment
                frg_list.append(f"<|{token}|>")
    return frg_list


def load_hme(
    modelargs, dataargs, add_frg_vocab: bool
) -> Tuple[nn.Module, AutoTokenizer, HMEConfig, List[str]]:
    """
    Loads an HME model and tokenizer from a checkpoint.

    This function handles two main scenarios:
    1. Loading from a pre-trained HME checkpoint: It expects specific files like
       `feature_fuser.pth` and loads the full HME architecture.
    2. Loading from a base LLM checkpoint (e.g., Llama-3): It initializes a new
       HME model, adds special multi-modal tokens to the tokenizer and embeddings,
       and sets up the model for fine-tuning.

    It also selects the correct model head (Generation, Regression, or Classification)
    based on the `task_type` specified in `dataargs`.

    Parameters
    ----------
    modelargs : object
        Arguments containing model-related paths and configurations.
        Expected attributes: `model_name_or_path`.
    dataargs : object
        Arguments containing data and task-related configurations.
        Expected attributes: `task_type`.
    add_frg_vocab : bool
        If True, adds fragment tokens to the tokenizer's vocabulary.

    Returns
    -------
    Tuple[nn.Module, AutoTokenizer, HMEConfig, List[str]]
        A tuple containing:
        - The loaded HME model.
        - The tokenizer.
        - The HME configuration object.
        - The list of special tokens added.
    """
    frg_tokens = get_frg_list(dataargs.task_type) if add_frg_vocab else []
    modal_special_tokens = ["<molecule_2d>", "<molecule_3d>", "<protein>"] + frg_tokens

    language_model = AutoModelForCausalLM.from_pretrained(
        modelargs.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
    )

    if "llama2" in modelargs.model_name_or_path.lower():
        tokenizer = LlamaTokenizer.from_pretrained(modelargs.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(modelargs.model_name_or_path)

    checkpoint_path = modelargs.model_name_or_path
    if os.path.exists(os.path.join(checkpoint_path, "feature_fuser.pth")):
        # --- Scenario 1: Loading a pre-trained HME model ---
        logging.info("Found feature_fuser.pth. Loading as a pre-trained HME model.")
        config = HMEConfig.from_pretrained(checkpoint_path)
        
        if dataargs is not None and "_reg" in dataargs.task_type:
            model = HMEForSequenceRegression(config, language_model)
            model.feature_fuser.load_state_dict(
                torch.load(os.path.join(checkpoint_path, "feature_fuser.pth"))
            )
            if os.path.exists(os.path.join(checkpoint_path, "regression_head.pth")):
                model.regression_head.load_state_dict(
                    torch.load(os.path.join(checkpoint_path, "regression_head.pth"))
                )
        else: # Default to conditional generation
            model = HMEForConditionalGeneration(config, language_model)
            model.feature_fuser.load_state_dict(
                torch.load(os.path.join(checkpoint_path, "feature_fuser.pth"))
            )
            model.generation_config.max_new_tokens = 512
            model.generation_config.do_sample = False

        # Ensure all loaded parts are in the correct dtype
        model.to(torch.bfloat16)

    else:
        # --- Scenario 2: Initializing from a base LLM ---
        logging.info("No feature_fuser.pth found. Initializing a new HME model.")
        ori_length = len(tokenizer)
        tokenizer.add_special_tokens(
            {"additional_special_tokens": modal_special_tokens, "pad_token": "<pad>"}
        )
        language_model.config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token is None:
            tokenizer.eos_token = tokenizer.convert_ids_to_tokens(
                language_model.config.eos_token_id
            )
        language_model.config.vocab_size = len(tokenizer)

        config = HMEConfig(
            text_config=language_model.config,
            molecule_2d_hidden_size=300,
            molecule_3d_hidden_size=512,
            protein_hidden_size=128,
            ignore_index=-100,
            modal_padding=-100,
            projector_hidden_act="gelu",
            protein_token_index=tokenizer.convert_tokens_to_ids("<protein>"),
            molecule_2d_token_index=tokenizer.convert_tokens_to_ids("<molecule_2d>"),
            molecule_3d_token_index=tokenizer.convert_tokens_to_ids("<molecule_3d>"),
        )
        
        if dataargs is not None and ("_reg" in dataargs.task_type):
            model = HMEForSequenceRegression(config, language_model)
        else: # Default to conditional generation
            model = HMEForConditionalGeneration(config, language_model)
            model.generation_config.max_new_tokens = 512
            model.generation_config.do_sample = False
            
        smart_tokenizer_and_embedding_resize(
            tokenizer=tokenizer, model=model, num_new_tokens=len(tokenizer) - ori_length
        )
        model.to(torch.bfloat16)

    model.config.pad_token_id = tokenizer.pad_token_id
    config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer, config, modal_special_tokens