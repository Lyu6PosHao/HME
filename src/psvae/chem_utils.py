"""
A collection of utility functions for cheminformatics tasks using RDKit.

This module provides helpers for converting between SMILES and RDKit molecule objects,
extracting sub-molecules, mapping atom indices, and counting atoms.
"""
from typing import Dict, List, Optional, Set

from rdkit import Chem
from rdkit.Chem.rdchem import Mol


def smi2mol(
    smiles: str, kekulize: bool = False, sanitize: bool = True
) -> Optional[Mol]:
    """
    Converts a SMILES string to an RDKit molecule object.

    Parameters
    ----------
    smiles : str
        The SMILES string to convert.
    kekulize : bool, optional
        Whether to kekulize the molecule, by default False.
    sanitize : bool, optional
        Whether to sanitize the molecule, by default True.

    Returns
    -------
    Optional[Mol]
        The corresponding RDKit Mol object, or None if conversion fails.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol and kekulize:
        Chem.Kekulize(mol, True)
    return mol


def mol2smi(mol: Mol, canonical: bool = True) -> str:
    """
    Converts an RDKit molecule object to a SMILES string.

    Parameters
    ----------
    mol : Mol
        The RDKit molecule object.
    canonical : bool, optional
        Whether to generate a canonical SMILES string, by default True.

    Returns
    -------
    str
        The resulting SMILES string.
    """
    return Chem.MolToSmiles(mol, canonical=canonical)


def get_submol(
    mol: Mol, atom_indices: List[int], kekulize: bool = False
) -> Optional[Mol]:
    """
    Extracts a sub-molecule from a larger molecule based on a list of atom indices.

    This function only considers bonds that are fully contained within the specified
    set of atoms.

    Parameters
    ----------
    mol : Mol
        The parent RDKit molecule object.
    atom_indices : List[int]
        A list of atom indices that define the sub-molecule.
    kekulize : bool, optional
        Whether to kekulize the resulting sub-molecule, by default False.

    Returns
    -------
    Optional[Mol]
        The extracted sub-molecule as an RDKit Mol object, or None if it fails.
    """
    if len(atom_indices) == 1:
        # Handle single-atom fragments, which PathToSubmol cannot process correctly.
        atom = mol.GetAtomWithIdx(atom_indices[0])
        atom_symbol = f"[{atom.GetSymbol()}]"
        return smi2mol(atom_symbol, kekulize)

    aid_set = set(atom_indices)
    edge_indices = []
    for bond in mol.GetBonds():
        # Check if both atoms of the bond are within the specified subgraph.
        if bond.GetBeginAtomIdx() in aid_set and bond.GetEndAtomIdx() in aid_set:
            edge_indices.append(bond.GetIdx())

    # PathToSubmol creates a new molecule from a list of bond indices.
    # It returns an empty molecule if edge_indices is empty.
    submolecule = Chem.PathToSubmol(mol, edge_indices)
    return submolecule


def get_submol_atom_map(
    mol: Mol, submol: Mol, group: Set[int], kekulize: bool = False
) -> Optional[Dict[int, int]]:
    """
    Maps atom indices from a parent molecule to a sub-molecule.

    Finds the matching pattern of `submol` within `mol` that corresponds to the
    original `group` of atom indices and returns the mapping from old (parent)
    to new (sub-molecule) indices.

    Parameters
    ----------
    mol : Mol
        The parent RDKit molecule.
    submol : Mol
        The sub-molecule object.
    group : Set[int]
        The original set of atom indices from the parent molecule that formed the sub-molecule.
    kekulize : bool, optional
        Whether to kekulize the sub-molecule before matching, by default False.

    Returns
    -------
    Optional[Dict[int, int]]
        A dictionary mapping old atom indices to new atom indices. Returns None if no
        valid mapping is found.
    """
    if len(group) == 1:
        # For a single-atom subgraph, the mapping is trivial.
        return {list(group)[0]: 0}

    # Standardize the sub-molecule by converting to and from SMILES to get a canonical ordering.
    smi = mol2smi(submol)
    submol_canonical = smi2mol(smi, kekulize, sanitize=False)
    if not submol_canonical:
        return None

    matches = mol.GetSubstructMatches(submol_canonical)
    for match in matches:
        # A valid match must be a permutation of the original atom indices in the group.
        if set(match) == group:
            return {old_idx: new_idx for new_idx, old_idx in enumerate(match)}
    return None


def cnt_atom(smi: str, return_dict: bool = False) -> Union[int, Dict[str, int]]:
    """
    Counts the number of atoms in a SMILES string.

    This function is designed to handle multi-character element symbols like 'Br' and 'Cl'.

    Parameters
    ----------
    smi : str
        The SMILES string.
    return_dict : bool, optional
        If True, returns a dictionary with counts for each atom type.
        If False, returns the total atom count, by default False.

    Returns
    -------
    Union[int, Dict[str, int]]
        Either the total count of atoms or a dictionary of counts per atom symbol.
    """
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return {} if return_dict else 0

    atom_dict = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atom_dict[symbol] = atom_dict.get(symbol, 0) + 1

    if return_dict:
        return atom_dict
    else:
        return sum(atom_dict.values())