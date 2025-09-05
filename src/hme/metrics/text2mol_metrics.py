"""
This script evaluates the performance of SMILES string generation models.

It calculates a variety of metrics, including:
- Validity: Percentage of chemically valid SMILES strings.
- Exact Match: Percentage of generated SMILES that are identical to the ground truth.
- BLEU score: For character-level similarity.
- Levenshtein distance: For string edit distance.
- Fingerprint Similarity: Tanimoto similarity for MACCS, RDKit, and Morgan fingerprints.
- FCD (Fréchet ChemNet Distance): To assess the distribution similarity of generated molecules.
"""

import json
import os
from typing import Dict, Tuple

import numpy as np
from fcd_torch import FCD
from Levenshtein import distance as lev
from nltk.translate.bleu_score import corpus_bleu
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from tqdm import tqdm


def get_out_gt_from_line(file_path: str, line: str) -> Tuple[str, str]:
    """
    Parses a single line from a prediction file to extract generated and ground truth text.

    Parameters
    ----------
    file_path : str
        The path to the input file, used to determine the format by its extension.
    line : str
        A single line from the file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the (generated_text, ground_truth_text).
    """
    if file_path.endswith(".txt"):
        parts = line.strip().split("<iamsplit>")
        if len(parts) != 2:
            raise ValueError(f"Invalid format in .txt file line: {line}")
        return parts[0].strip(), parts[1].strip()
    else:
        data = json.loads(line)
        return data["gen"].strip(), data["gt"].strip()


def calc_fcd_torch(generated_smiles: list[str], reference_smiles: list[str]) -> float:
    """
    Calculates the Fréchet ChemNet Distance (FCD) between two sets of SMILES.

    Parameters
    ----------
    generated_smiles : list[str]
        A list of generated SMILES strings.
    reference_smiles : list[str]
        A list of reference (ground truth) SMILES strings.

    Returns
    -------
    float
        The computed FCD score.
    """
    fcd_calculator = FCD()
    fcd_score = fcd_calculator(reference_smiles, generated_smiles)

    return fcd_score


def eval_smi_generation(file_path: str) -> Dict[str, float]:
    """
    Evaluates a SMILES generation task from a result file.

    Parameters
    ----------
    file_path : str
        The path to the file containing generated and ground truth SMILES.

    Returns
    -------
    Dict[str, float]
        A dictionary containing all calculated evaluation metrics.
    """
    # --- Setup ---
    np.random.seed(42)
    RDLogger.DisableLog("rdApp.*")

    with open(file_path, "r") as f:
        lines = f.readlines()

    # --- Initialization ---
    total_samples = len(lines)
    output_tokens, gt_tokens = [], []
    levs = []
    maccs_sim, rdk_sim, morgan_sim = [], [], []
    invalid_smiles_count, exact_matches = 0, 0
    valid_generated_smiles = []
    all_gt_smiles = []

    # --- Processing Loop ---
    for line in tqdm(lines, desc=f"Evaluating {os.path.basename(file_path)}"):
        out_raw, gt_raw = get_out_gt_from_line(file_path=file_path, line=line)

        # Extract SMILES from potentially longer strings
        out_smi = out_raw.split("Molecular SMILES is: ")[-1].strip()
        gt_smi = gt_raw.split("Molecular SMILES is: ")[-1].strip()
        all_gt_smiles.append(gt_smi)

        try:
            # Canonicalize SMILES to ensure a standard representation
            canon_out_smi = Chem.CanonSmiles(out_smi)
            canon_gt_smi = Chem.CanonSmiles(gt_smi)
            valid_generated_smiles.append(canon_out_smi)

            mol_output = Chem.MolFromSmiles(canon_out_smi)
            mol_gt = Chem.MolFromSmiles(canon_gt_smi)

            if mol_output is None or mol_gt is None:
                invalid_smiles_count += 1
                continue

            # --- Metric Calculations for Valid Pairs ---
            if Chem.MolToInchi(mol_output) == Chem.MolToInchi(mol_gt):
                exact_matches += 1

            output_tokens.append(list(canon_out_smi))
            gt_tokens.append([list(canon_gt_smi)])
            levs.append(lev(canon_out_smi, canon_gt_smi))

            # Fingerprint Similarities
            maccs_sim.append(
                DataStructs.TanimotoSimilarity(
                    MACCSkeys.GenMACCSKeys(mol_output), MACCSkeys.GenMACCSKeys(mol_gt)
                )
            )
            rdk_sim.append(
                DataStructs.TanimotoSimilarity(
                    Chem.RDKFingerprint(mol_output), Chem.RDKFingerprint(mol_gt)
                )
            )
            morgan_sim.append(
                DataStructs.TanimotoSimilarity(
                    AllChem.GetMorganFingerprint(mol_output, 2),
                    AllChem.GetMorganFingerprint(mol_gt, 2),
                )
            )
        except Exception:
            invalid_smiles_count += 1
            continue

    # --- Final Score Aggregation ---
    results = {
        "Validity": 1.0 - (invalid_smiles_count / total_samples),
        "Exact Match": exact_matches / total_samples,
        "BLEU": corpus_bleu(gt_tokens, output_tokens),
        "Levenshtein": np.mean(levs) if levs else 0.0,
        "MACCS Tanimoto": np.mean(maccs_sim) if maccs_sim else 0.0,
        "RDKit Tanimoto": np.mean(rdk_sim) if rdk_sim else 0.0,
        "Morgan Tanimoto": np.mean(morgan_sim) if morgan_sim else 0.0,
        "FCD": (
            calc_fcd_torch(valid_generated_smiles, all_gt_smiles)
            if valid_generated_smiles
            else 0.0
        ),
    }

    # --- Print Results ---
    print("\n--- Evaluation Results ---")
    for key, value in results.items():
        print(f"{key:<18}: {value:.4f}")
    print("--------------------------\n")

    return results


if __name__ == "__main__":
    # This setting should be here to not affect other modules if this file is imported.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # --- Example file paths for evaluation ---
    # Please update these paths to your actual result files.
    file_path = "result.jsonl"
    eval_smi_generation(file_path)
