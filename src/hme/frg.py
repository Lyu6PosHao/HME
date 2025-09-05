"""
This script provides utility functions to process molecular SMILES strings,
specifically for tokenizing them into fragments using a BPE-like tokenizer.
"""

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across different libraries.

    Parameters
    ----------
    seed : int, optional
        The seed to use, by default 42.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)


def get_fragments(mol_list: List) -> str:
    """
    Extracts fragment SMILES from a list of tokenized molecule objects.

    Each fragment's SMILES is wrapped in '<|...|>' and all fragments from a
    single molecule are concatenated. If multiple molecules are in the list,
    their fragment strings are joined by '.'.

    Parameters
    ----------
    mol_list : List
        A list of molecule objects, as produced by the `psvae` tokenizer.

    Returns
    -------
    str
        A string representing the concatenated fragments.
    """
    result = []
    for mol in mol_list:
        if mol is None:
            continue
        fragments = ""
        for node in mol.nodes:
            fragments += f"<|{mol.get_node(node).smiles}|>"
        result.append(fragments)
    return ".".join(result)


def get_frg_from_one_smiles(
    smiles: str, vocab_file: str, verbose: bool = False
) -> Tuple[str, Optional[str], Optional[List]]:
    """
    Tokenizes a single SMILES string into fragments.

    This function canonicalizes the input SMILES, tokenizes it into a list of
    molecular fragments using a BPE vocabulary, and then formats these fragments
    into a single string.

    Parameters
    ----------
    smiles : str
        The input SMILES string to process.
    vocab_file : str
        Path to the vocabulary file required by the `psvae` tokenizer.
    verbose : bool, optional
        If True, prints an error message if tokenization fails, by default False.

    Returns
    -------
    Tuple[str, Optional[str], Optional[List]]
        A tuple containing:
        - The formatted fragment string.
        - The canonicalized SMILES string (or None on failure).
        - The list of raw tokenized molecule objects (or None on failure).
    """
    # Import locally as it's a specific dependency for this function
    from hme.psvae.mol_bpe import Tokenizer

    mol_tokenizer = Tokenizer(vocab_file)
    try:
        can_smiles = Chem.CanonSmiles(smiles)
        mol_list = mol_tokenizer.tokenize(can_smiles)
    except Exception:
        if verbose:
            print(f"An error occurred when getting fragments from SMILES: {smiles}")
        return "", smiles, None

    if not isinstance(mol_list, list):
        mol_list = [mol_list]

    fragments = get_fragments(mol_list)
    return fragments, can_smiles, mol_list