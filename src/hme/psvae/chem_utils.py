from rdkit import Chem
from periodictable import elements

MAX_VALENCE = {element.symbol: 10 for element in elements}


def smi2mol(smiles: str, kekulize: bool = False, sanitize: bool = True):
    """
    Convert a SMILES string to an RDKit molecule.

    Args:
        smiles (str): Input SMILES string.
        kekulize (bool, optional): Whether to kekulize aromatic bonds. Defaults to False.
        sanitize (bool, optional): Whether to sanitize the molecule. Defaults to True.

    Returns:
        Chem.Mol: RDKit molecule object.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if kekulize:
        Chem.Kekulize(mol, True)
    return mol


def mol2smi(mol, canonical: bool = True) -> str:
    """
    Convert an RDKit molecule to a SMILES string.

    Args:
        mol (Chem.Mol): RDKit molecule.
        canonical (bool, optional): Whether to generate canonical SMILES. Defaults to True.

    Returns:
        str: SMILES representation of the molecule.
    """
    return Chem.MolToSmiles(mol, canonical=canonical)


def get_submol(mol: Chem.Mol, atom_indices: list, kekulize: bool = False) -> Chem.Mol:
    """
    Extract a sub-molecule from a larger molecule based on atom indices.

    Args:
        mol (Chem.Mol): Original molecule.
        atom_indices (list): Atom indices of the subgraph within the molecule.
        kekulize (bool, optional): Whether to kekulize aromatic bonds. Defaults to False.

    Returns:
        Chem.Mol: Sub-molecule corresponding to the given atom indices.
    """
    if len(atom_indices) == 1:
        atom_symbol = mol.GetAtomWithIdx(atom_indices[0]).GetSymbol()
        atom_symbol = f"[{atom_symbol}]"
        return smi2mol(atom_symbol, kekulize)

    aid_dict = {i: True for i in atom_indices}
    edge_indices = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        begin_aid, end_aid = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin_aid in aid_dict and end_aid in aid_dict:
            edge_indices.append(i)
    return Chem.PathToSubmol(mol, edge_indices)


def get_submol_atom_map(mol: Chem.Mol, submol: Chem.Mol, group: list, kekulize: bool = False) -> dict:
    """
    Build a mapping from atom indices in the original molecule to atom indices in the sub-molecule.

    Args:
        mol (Chem.Mol): Original molecule.
        submol (Chem.Mol): Sub-molecule extracted from the original.
        group (list): Atom indices of the sub-molecule in the original molecule.
        kekulize (bool, optional): Whether to kekulize aromatic bonds. Defaults to False.

    Returns:
        dict: Mapping {original_atom_idx: submol_atom_idx}.
    """
    if len(group) == 1:
        return {group[0]: 0}

    smi = mol2smi(submol)
    submol = smi2mol(smi, kekulize, sanitize=False)

    matches = mol.GetSubstructMatches(submol)
    old2new = {i: 0 for i in group}
    found = False
    for m in matches:
        hit = True
        for i, atom_idx in enumerate(m):
            if atom_idx not in old2new:
                hit = False
                break
            old2new[atom_idx] = i
        if hit:
            found = True
            break
    assert found
    return old2new


def cnt_atom(smi: str, return_dict: bool = False):
    """
    Count atoms of each type in a SMILES string.

    Args:
        smi (str): SMILES string.
        return_dict (bool, optional): If True, return a dictionary mapping atom type to count.
                                      If False, return the total atom count. Defaults to False.

    Returns:
        dict or int: Dictionary of atom counts or total atom count.
    """
    atom_dict = {atom: 0 for atom in MAX_VALENCE}
    for i in range(len(smi)):
        symbol = smi[i].upper()
        next_char = smi[i + 1] if i + 1 < len(smi) else None
        if symbol == 'B' and next_char == 'r':
            symbol += next_char
        elif symbol == 'C' and next_char == 'l':
            symbol += next_char
        if symbol in atom_dict:
            atom_dict[symbol] += 1
    return atom_dict if return_dict else sum(atom_dict.values())
