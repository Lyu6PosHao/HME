import json
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import RDLogger, Chem
from rdkit.Chem import AllChem, CanonSmiles
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm
from unimol_tools import UniMolRepr

from hme.modeling_tower import Molecule2DTower
from hme.util import set_seed

# --- Initial Setup ---
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings(action="ignore")
set_seed(42)


def inner_smi2coords(
    smi: str, seed: int = 42, mode: str = "fast", remove_hs: bool = False
) -> Tuple[List[str], Optional[np.ndarray]]:
    """
    Converts a SMILES string into 3D coordinates for each atom.

    Falls back to 2D coordinates if 3D generation fails.

    Parameters
    ----------
    smi : str
        The SMILES representation of the molecule.
    seed : int, optional
        The random seed for conformation generation, by default 42.
    mode : str, optional
        Conformation generation mode ('fast' or 'heavy'), by default "fast".
    remove_hs : bool, optional
        Whether to remove hydrogen atoms from the final coordinates, by default False.

    Returns
    -------
    Tuple[List[str], Optional[np.ndarray]]
        A tuple containing the list of atom symbols and their corresponding 3D coordinates.
        Coordinates are None if generation fails completely.
    """
    returned_none = False
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    assert len(atoms) > 0, f"No atoms in molecule: {smi}"

    try:
        res = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if res == 0:  # Success
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass  # Use the unoptimized conformation
            coordinates = mol.GetConformer().GetPositions().astype(np.float32)
        elif res == -1 and mode == "heavy":  # Failed but try harder
            AllChem.EmbedMolecule(mol, maxAttempts=5000, randomSeed=seed)
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass
            coordinates = mol.GetConformer().GetPositions().astype(np.float32)
        else:  # Fallback to 2D
            AllChem.Compute2DCoords(mol)
            coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    except Exception:
        print(f"Failed to generate conformer for {smi}, replacing with zeros.")
        coordinates = np.zeros((len(atoms), 3), dtype=np.float32)
        returned_none = True

    assert len(atoms) == len(coordinates), f"Coordinates shape mismatch for {smi}"

    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != "H"]
        atoms = [atoms[i] for i in idx]
        coordinates = coordinates[idx]
        assert len(atoms) == len(
            coordinates
        ), f"No-H coordinates shape mismatch for {smi}"

    return atoms, None if returned_none else coordinates


def smi2coords(smi: str) -> Dict:
    """Wrapper for `inner_smi2coords` to format output as a dictionary."""
    atoms, coords = inner_smi2coords(smi)
    return {
        "smiles": smi,
        "atoms": atoms,
        "coordinates": coords.tolist() if coords is not None else None,
    }


def get_json_list(file_path: str) -> List[Dict]:
    """Loads a list of dictionaries from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def check_can_smiles(file_path: str) -> None:
    """Checks if all SMILES in a JSON file are canonical."""
    json_list = get_json_list(file_path)
    smiles_set = {item["smiles"] for item in json_list}
    print(f"Checking {len(smiles_set)} unique SMILES for canonicalization...")
    for smiles in tqdm(smiles_set):
        if CanonSmiles(smiles) != smiles:
            print(f"Non-canonical SMILES found: {smiles}")
            print(f"Canonical form: {CanonSmiles(smiles)}")
    print(f"Check complete for {file_path}.")


def get_conformation(file_path: str, save_every: int = 1000) -> None:
    """
    Generates and saves 3D conformations for all unique SMILES in a JSON file.

    This function is resumable. It checks for an intermediate `.cfm` file and
    continues from where it left off.

    Parameters
    ----------
    file_path : str
        Path to the input JSON file containing SMILES.
    save_every : int, optional
        How often to save the intermediate results, by default 1000.
    """
    json_list = get_json_list(file_path)
    smiles_set = sorted(list({item["smiles"] for item in json_list}))
    output_path = file_path + ".cfm"

    conformations = []
    processed_smiles = set()
    if os.path.exists(output_path):
        print(f"Resuming from existing file: {output_path}")
        conformations = get_json_list(output_path)
        processed_smiles = {item["smiles"] for item in conformations}

    smiles_to_process = [smi for smi in smiles_set if smi not in processed_smiles]
    print(f"Total unique SMILES: {len(smiles_set)}")
    print(f"SMILES to process: {len(smiles_to_process)}")

    if not smiles_to_process:
        print("All SMILES already processed.")
        return

    for i, smi in enumerate(tqdm(smiles_to_process, desc="Generating conformations")):
        conformations.append(smi2coords(smi))
        if (i + 1) % save_every == 0:
            with open(output_path, "w") as f:
                json.dump(conformations, f)

    with open(output_path, "w") as f:
        json.dump(conformations, f)
    print(f"Finished! Saved {len(conformations)} conformations to {output_path}")


def get_2d3d_tensors(file_path_with_cfm: str) -> None:
    """
    Generates and saves 2D and 3D molecular feature tensors.

    This function reads a conformation file (`.json.cfm`), computes 2D features
    using Molecule2DTower and 3D features using UniMolRepr, and saves the
    results as a dictionary in a .pt file.

    Parameters
    ----------
    file_path_with_cfm : str
        Path to the conformation file (should end with '.cfm').
    """
    molecule_list = get_json_list(file_path_with_cfm)
    print(f"Loaded {len(molecule_list)} molecules with conformations.")

    # Initialize models
    molecule_2d_tower = Molecule2DTower(device="cuda", config=None)
    molecule_3d_tower = UniMolRepr(data_type="molecule", remove_hs=False, use_gpu=True)

    # --- 2D Feature Generation ---
    for mol in tqdm(molecule_list, desc="Generating 2D features"):
        try:
            mol["molecule_raw_2d_features"] = torch.tensor(
                molecule_2d_tower(mol["smiles"])
            )
        except Exception as e:
            print(f"Failed to generate 2D features for {mol['smiles']}: {e}")
            mol["molecule_raw_2d_features"] = None

    # --- 3D Feature Generation (in batches) ---
    batch_size = 2048
    valid_mols = [mol for mol in molecule_list if mol.get("coordinates") is not None]
    print(f"Generating 3D features for {len(valid_mols)} valid molecules...")
    for i in range(0, len(valid_mols), batch_size):
        batch = valid_mols[i : i + batch_size]
        input_data = {
            "atoms": [mol["atoms"] for mol in batch],
            "coordinates": [mol["coordinates"] for mol in batch],
        }
        repr_3d = molecule_3d_tower.get_repr(input_data, return_atomic_reprs=True)[
            "atomic_reprs"
        ]
        for mol, feature in zip(batch, repr_3d):
            mol["molecule_raw_3d_features"] = torch.tensor(feature)

    # Assign None to molecules that failed 3D conformation
    for mol in molecule_list:
        if "molecule_raw_3d_features" not in mol:
            mol["molecule_raw_3d_features"] = None

    # --- Save Results ---
    output_path = os.path.splitext(file_path_with_cfm)[0] + ".pt"
    molecule_dict = {mol["smiles"]: mol for mol in molecule_list}
    torch.save(molecule_dict, output_path)
    print(f"Saved 2D & 3D features for {len(molecule_dict)} molecules to {output_path}")

# The curation procedures implemented here are based on and derived from the methodology proposed in 
# "Trust, but Verify: On the Importance of Chemical Structure Curation in Cheminformatics and QSAR Modeling Research"
class ChemicalCuration:
    """
    Implements a comprehensive chemical structure curation pipeline following the
    principles outlined in:

        "Trust, but Verify: On the Importance of Chemical Structure Curation in
        Cheminformatics and QSAR Modeling Research"

    The pipeline performs the essential curation steps required for reliable
    downstream molecular modeling, including:
        - Organic compound filtering
        - Metal/organometallic filtration
        - Largest-fragment selection (desalting)
        - Functional group normalization
        - Charge neutralization
        - Valence validation
        - Aromaticity normalization
        - Canonical tautomer selection
        - Hydrogen representation normalization

    All steps are implemented with RDKit's MolStandardize and chemistry utilities.
    """

    def __init__(self) -> None:
        """Initializes RDKit MolStandardize tools."""
        self.largest_fragment = rdMolStandardize.LargestFragmentChooser()
        self.normalizer = rdMolStandardize.Normalizer()
        self.uncharger = rdMolStandardize.Uncharger()
        self.tautomer_enum = rdMolStandardize.TautomerEnumerator()

        # A simple but robust metal list sufficient for curation/QSAR
        self.metal_atomic_nums = {
            3, 4, 11, 12, 13, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 37, 38, 39, 40, 47, 48, 49, 50,
            55, 56, 57, 72, 73, 74, 75, 76, 77, 78, 79
        }

    # ----------------------------- Utility Methods -----------------------------

    def is_organic(self, mol: Chem.Mol) -> bool:
        """Checks if a molecule contains carbon (minimum requirement for organic)."""
        return any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())

    def has_metal(self, mol: Chem.Mol) -> bool:
        """Detects whether the molecule contains metal atoms."""
        return any(atom.GetAtomicNum() in self.metal_atomic_nums for atom in mol.GetAtoms())

    def has_valence_problem(self, mol: Chem.Mol) -> bool:
        """Uses RDKit sanitization to detect valence issues."""
        try:
            Chem.SanitizeMol(mol)
            return False
        except Exception:
            return True

    # ------------------------------ Core Workflow ------------------------------

    def process(self, smiles: str) -> Tuple[Optional[Chem.Mol], str]:
        """
        Executes the complete chemical curation pipeline.

        Parameters
        ----------
        smiles : str
            Input SMILES string.

        Returns
        -------
        Tuple[Optional[Chem.Mol], str]
            - The curated RDKit molecule (or None if rejected).
            - A status message describing success or rejection reason.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES"

        # --- Step 1: Reject inorganic molecules ---
        if not self.is_organic(mol):
            return None, "Rejected: Inorganic"

        # --- Step 2: Reject organometallic compounds ---
        if self.has_metal(mol):
            return None, "Rejected: Contains metal"

        # --- Step 3: Select the largest organic fragment (desalting) ---
        try:
            mol = self.largest_fragment.choose(mol)
        except Exception:
            return None, "Error during desalting"

        # --- Step 4: Functional group normalization ---
        mol = self.normalizer.normalize(mol)

        # --- Step 5: Charge neutralization ---
        mol = self.uncharger.uncharge(mol)

        # --- Step 6: Aromaticity normalization ---
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass  # Some structures cannot be kekulized

        Chem.SetAromaticity(mol)

        # --- Step 7: Hydrogen normalization ---
        mol = Chem.AddHs(mol)
        mol = Chem.RemoveHs(mol)

        # --- Step 8: Tautomer canonicalization ---
        mol = self.tautomer_enum.Canonicalize(mol)

        # --- Step 9: Valence validation ---
        if self.has_valence_problem(mol):
            return None, "Rejected: Valence error"

        return mol, "Success"


def curate_smiles_list(smiles_list: List[str]) -> Dict[str, str]:
    """
    Applies the full curation pipeline to a list of SMILES strings and removes duplicates.

    Deduplication uses InChIKey, which is the strongest duplicate detector for curated structures.

    Parameters
    ----------
    smiles_list : List[str]
        Raw SMILES list (possibly noisy, uncurated).

    Returns
    -------
    Dict[str, str]
        Mapping from InChIKey → canonical cleaned SMILES.
    """
    curator = ChemicalCuration()
    unique_mols: Dict[str, str] = {}

    for smi in smiles_list:
        mol, status = curator.process(smi)
        if mol is None:
            continue

        key = Chem.MolToInchiKey(mol)
        can_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

        if key not in unique_mols:
            unique_mols[key] = can_smi

    return unique_mols

if __name__ == "__main__":
    # This should be set only when running the script directly.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # --- Example Preprocessing Pipeline ---
    # STEP 1: Generate 3D conformations from a SMILES JSON file.
    # Replace this path with the path to your input JSON file.
    input_json_path = "/path/to/your/dataset.json"

    print(f"--- Step 1: Generating conformations for {input_json_path} ---")
    get_conformation(input_json_path)

    # STEP 2: Generate 2D and 3D feature embeddings from the conformation file.
    conformation_file_path = input_json_path + ".cfm"

    print(f"\n--- Step 2: Generating 2D/3D tensors from {conformation_file_path} ---")
    if os.path.exists(conformation_file_path):
        get_2d3d_tensors(conformation_file_path)
    else:
        print(f"Error: Conformation file not found at {conformation_file_path}")
