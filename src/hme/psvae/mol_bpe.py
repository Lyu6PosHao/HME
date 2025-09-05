#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
This script implements a Graph-based Byte Pair Encoding (BPE) algorithm to
extract principal subgraphs from a molecule corpus and generate a vocabulary.

The main entry point `Graph_BPE_to_get_vocab` drives the process. Other classes
and functions, such as `Tokenizer`, can be imported and used by other scripts.
"""
import argparse
import json
import multiprocessing as mp
from typing import Any, Dict, List, Tuple

from rdkit import Chem
from tqdm import tqdm

from chem_utils import MAX_VALENCE, cnt_atom, get_submol, mol2smi, smi2mol
from logger import print_log
from molecule import Molecule

# --- Classes for Principal Subgraph Extraction ---


class MolInSubgraph:
    """
    A class to manage a molecule and its evolving subgraphs during BPE.

    This class starts by treating each atom as a separate subgraph and iteratively
    merges them based on frequency, tracking the state of the molecule's fragmentation.

    Parameters
    ----------
    mol : Chem.Mol
        The RDKit molecule object.
    kekulize : bool, optional
        Whether to kekulize the molecule, by default False.
    """

    def __init__(self, mol: Chem.Mol, kekulize: bool = False):
        self.mol = mol
        self.smi = mol2smi(mol)
        self.kekulize = kekulize
        self.subgraphs: Dict[int, Dict[int, str]] = {}  # pid -> {atom_idx: symbol}
        self.subgraphs_smis: Dict[int, str] = {}  # pid -> smi
        for atom in mol.GetAtoms():
            idx, symbol = atom.GetIdx(), atom.GetSymbol()
            self.subgraphs[idx] = {idx: symbol}
            self.subgraphs_smis[idx] = symbol

        self.inversed_index: Dict[int, int] = {
            aid: aid for aid in range(mol.GetNumAtoms())
        }  # atom_idx -> pid
        self.upid_cnt = mol.GetNumAtoms()  # Unique pid counter

        self._dirty = True  # Flag to indicate if neighbor info needs recalculation
        self._smi2pids: Dict[str, List[Tuple[int, int]]] = (
            {}
        )  # Cache for smi -> [(pid1, pid2), ...]

    def get_nei_subgraphs(self) -> Tuple[List[Dict[int, str]], List[Tuple[int, int]]]:
        """Finds all potential new subgraphs by merging adjacent existing subgraphs."""
        nei_subgraphs, merge_pids = [], []
        for pid1 in self.subgraphs:
            subgraph1 = self.subgraphs[pid1]
            local_nei_pids = set()
            for aid in subgraph1:
                atom = self.mol.GetAtomWithIdx(aid)
                for nei in atom.GetNeighbors():
                    nei_idx = nei.GetIdx()
                    if nei_idx not in subgraph1 and nei_idx > aid:
                        local_nei_pids.add(self.inversed_index[nei_idx])

            for pid2 in local_nei_pids:
                new_subgraph = self.subgraphs[pid1].copy()
                new_subgraph.update(self.subgraphs[pid2])
                nei_subgraphs.append(new_subgraph)
                merge_pids.append((pid1, pid2))
        return nei_subgraphs, merge_pids

    def get_nei_smis(self) -> List[str]:
        """
        Gets the SMILES of all potential new subgraphs, caching the results.
        """
        if self._dirty:
            nei_subgraphs, merge_pids = self.get_nei_subgraphs()
            self._smi2pids = {}
            for i, subgraph in enumerate(nei_subgraphs):
                submol = get_submol(self.mol, list(subgraph.keys()), self.kekulize)
                smi = mol2smi(submol)
                if smi not in self._smi2pids:
                    self._smi2pids[smi] = []
                self._smi2pids[smi].append(merge_pids[i])
            self._dirty = False
        return list(self._smi2pids.keys())

    def merge(self, smi: str) -> None:
        """
        Merges subgraphs based on the most frequent neighboring SMILES pattern.

        If a given `smi` corresponds to a valid merge operation, the two participating
        subgraphs are combined into a new one, and the old ones are removed.

        Parameters
        ----------
        smi : str
            The SMILES string of the subgraph pattern to merge.
        """
        if self._dirty:
            self.get_nei_smis()

        if smi in self._smi2pids:
            merge_pids_list = self._smi2pids[smi]
            for pid1, pid2 in merge_pids_list:
                if pid1 in self.subgraphs and pid2 in self.subgraphs:
                    # Combine subgraphs
                    self.subgraphs[pid1].update(self.subgraphs[pid2])
                    new_pid = self.upid_cnt
                    self.subgraphs[new_pid] = self.subgraphs[pid1]
                    self.subgraphs_smis[new_pid] = smi

                    # Update inverse index for all atoms in the new subgraph
                    for aid in self.subgraphs[new_pid]:
                        self.inversed_index[aid] = new_pid

                    # Clean up old entries
                    del self.subgraphs[pid1], self.subgraphs[pid2]
                    del self.subgraphs_smis[pid1], self.subgraphs_smis[pid2]
                    self.upid_cnt += 1
        self._dirty = True  # Mark as revised after a merge operation

    def get_smis_subgraphs(self) -> List[Tuple[str, List[int]]]:
        """Returns the final list of (SMILES, atom_indices) for the fragments."""
        res = []
        for pid, smi in self.subgraphs_smis.items():
            idxs = list(self.subgraphs[pid].keys())
            res.append((smi, idxs))
        return res


def freq_cnt(mol: MolInSubgraph) -> Tuple[Dict[str, int], "MolInSubgraph"]:
    """Counts the frequency of neighboring subgraph patterns for a single molecule."""
    freqs = {}
    nei_smis = mol.get_nei_smis()
    for smi in nei_smis:
        freqs[smi] = freqs.get(smi, 0) + 1
    return freqs, mol


def graph_bpe(
    fname: str, vocab_len: int, vocab_path: str, cpus: int, kekulize: bool
) -> Tuple[List[str], Dict[str, List[Any]]]:
    """
    Performs Graph BPE on a corpus of SMILES to generate a vocabulary.

    This function loads molecules, iteratively finds the most frequent subgraph
    pattern, merges it, and repeats until the desired vocabulary size is reached.

    Parameters
    ----------
    fname : str
        Path to the SMILES corpus file (one SMILES per line).
    vocab_len : int
        The desired size of the vocabulary.
    vocab_path : str
        Path to save the generated vocabulary file.
    cpus : int
        Number of CPU cores to use for parallel processing.
    kekulize : bool
        Whether to kekulize molecules.

    Returns
    -------
    Tuple[List[str], Dict[str, List[Any]]]
        A tuple containing:
        - A list of the selected subgraph SMILES in the vocabulary.
        - A dictionary with details about each subgraph (atom count, frequency).
    """
    print_log(f"Loading mols from {fname} ...")
    with open(fname, "r") as fin:
        smis = [line.strip() for line in fin]

    mols = []
    for smi in tqdm(smis, desc="Initializing molecules"):
        try:
            mol_obj = smi2mol(smi, kekulize)
            if mol_obj:
                mols.append(MolInSubgraph(mol_obj, kekulize))
        except Exception:
            print_log(f"Parsing {smi} failed. Skip.", level="ERROR")

    selected_smis = list(MAX_VALENCE.keys())
    details = {atom: [1, 0] for atom in selected_smis}  # smi -> [atom_count, freq]
    for smi in smis:
        cnts = cnt_atom(smi, return_dict=True)
        for atom, count in cnts.items():
            if atom in details:
                details[atom][1] += count

    add_len = vocab_len - len(selected_smis)
    print_log(
        f"Added {len(selected_smis)} atoms, {add_len} principal subgraphs to extract"
    )

    with mp.Pool(cpus) as pool:
        with tqdm(total=add_len, desc="BPE progressing") as pbar:
            while len(selected_smis) < vocab_len:
                res_list = pool.map(freq_cnt, mols)
                freqs, mols = {}, []
                for freq, mol_state in res_list:
                    mols.append(mol_state)
                    for key, val in freq.items():
                        freqs[key] = freqs.get(key, 0) + val

                if not freqs:
                    print_log("No more patterns to merge. Stopping.", level="WARNING")
                    break

                merge_smi = max(freqs, key=freqs.get)
                max_cnt = freqs[merge_smi]

                for mol in mols:
                    mol.merge(merge_smi)

                if merge_smi in details:
                    continue
                selected_smis.append(merge_smi)
                details[merge_smi] = [cnt_atom(merge_smi), max_cnt]
                pbar.update(1)

    print_log("Sorting vocab by atom count...")
    selected_smis.sort(key=lambda x: details[x][0], reverse=True)

    with open(vocab_path, "w") as fout:
        fout.write(json.dumps({"kekulize": kekulize}) + "\n")
        for smi in selected_smis:
            fout.write(f"{smi}\t{details[smi][0]}\t{details[smi][1]}\n")

    return selected_smis, details


class Tokenizer:
    """A tokenizer to segment molecules into principal subgraphs based on a BPE vocabulary."""

    def __init__(self, vocab_path: str):
        with open(vocab_path, "r") as fin:
            lines = fin.read().strip().split("\n")
        config = json.loads(lines[0])
        self.kekulize = config["kekulize"]
        lines = lines[1:]

        self.vocab_dict: Dict[str, Tuple[int, int, int]] = {}
        self.idx2subgraph: List[str] = []
        self.subgraph2idx: Dict[str, int] = {}
        for idx, line in enumerate(lines):
            smi, atom_num, freq = line.strip().split("\t")
            self.vocab_dict[smi] = (int(atom_num), int(freq), idx)
            self.subgraph2idx[smi] = len(self.idx2subgraph)
            self.idx2subgraph.append(smi)

        # Add special tokens
        self.pad, self.end = "<pad>", "<s>"
        for smi in [self.pad, self.end]:
            self.subgraph2idx[smi] = len(self.idx2subgraph)
            self.idx2subgraph.append(smi)

    def tokenize(self, mol: Union[str, Chem.Mol]) -> Optional[Union[List, Molecule]]:
        """
        Tokenizes a molecule into a `Molecule` object or a list of them.

        The molecule is decomposed into principal subgraphs found in the vocabulary.

        Parameters
        ----------
        mol : Union[str, Chem.Mol]
            The input molecule as a SMILES string or an RDKit Mol object.

        Returns
        -------
        Optional[Union[List, Molecule]]
            A `Molecule` object representing the tokenized graph. If the input SMILES
            contains multiple disconnected fragments ('.'), a list of `Molecule` objects
            is returned. Returns `None` for single-atom molecules.
        """
        smiles = mol if isinstance(mol, str) else mol2smi(mol)
        rdkit_mol = smi2mol(smiles, self.kekulize) if isinstance(mol, str) else mol

        if not rdkit_mol or rdkit_mol.GetNumAtoms() <= 1:
            return None
        if "." in smiles:
            fragments = smiles.split(".")
            return [self.tokenize(frag) for frag in fragments]

        mol_in_subgraph = MolInSubgraph(rdkit_mol, kekulize=self.kekulize)
        while True:
            nei_smis = mol_in_subgraph.get_nei_smis()
            max_freq, merge_smi = -1, ""
            for smi in nei_smis:
                if smi not in self.vocab_dict:
                    continue
                freq = self.vocab_dict[smi][1]
                if freq > max_freq:
                    max_freq, merge_smi = freq, smi

            if max_freq == -1:
                break
            mol_in_subgraph.merge(merge_smi)

        res = mol_in_subgraph.get_smis_subgraphs()
        group_idxs = [x[1] for x in res]
        return Molecule(rdkit_mol, group_idxs, self.kekulize)

    def __call__(self, mol: Union[str, Chem.Mol]) -> Optional[Union[List, Molecule]]:
        return self.tokenize(mol)

    def __len__(self) -> int:
        return len(self.idx2subgraph)


def remove_charge(mol: Chem.Mol) -> Chem.Mol:
    """Removes formal charges from all atoms in a molecule."""
    for atom in mol.GetAtoms():
        atom.SetFormalCharge(0)
    return mol


def remove_chiral(mol: Chem.Mol) -> Chem.Mol:
    """Removes all chiral information from a molecule."""
    Chem.RemoveStereochemistry(mol)
    return mol


def remove_radical_electrons(mol: Chem.Mol) -> Chem.Mol:
    """Removes radical electrons from all atoms in a molecule."""
    for atom in mol.GetAtoms():
        atom.SetNumRadicalElectrons(0)
    return mol


def parse_args():
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Principal subgraph extraction with BPE"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the molecule corpus file (SMILES).",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=800, help="Desired size of the vocabulary."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the generated vocabulary.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of CPU cores to use.",
    )
    parser.add_argument(
        "--kekulize",
        action="store_true",
        help="Kekulize molecules (replace aromatic with single/double bonds).",
    )
    return parser.parse_args()


def Graph_BPE_to_get_vocab():
    """Main function to run the graph BPE vocabulary generation process."""
    args = parse_args()
    graph_bpe(
        args.data,
        vocab_len=args.vocab_size,
        vocab_path=args.output,
        cpus=args.workers,
        kekulize=args.kekulize,
    )

    print("\n--- Vocabulary Generation Complete ---")
    print(f"Vocabulary saved to: {args.output}")

    # --- Example Tokenization ---
    print("\n--- Testing tokenizer with an example ---")
    tokenizer = Tokenizer(args.output)
    example_smiles = "Cc1c(Cl)cccc1-n1c(SCCNS(C)(=O)=O)nc2ccccc2c1=O"
    print(f"Example SMILES: {example_smiles}")
    tokenized_mol = tokenizer.tokenize(example_smiles)

    if tokenized_mol:
        print("Tokenized molecule object created.")
        reconstructed_smi = (
            ".".join([frag.to_smiles() for frag in tokenized_mol])
            if isinstance(tokenized_mol, list)
            else tokenized_mol.to_smiles()
        )
        print(f"Reconstructed SMILES: {reconstructed_smi}")
        if Chem.CanonSmiles(reconstructed_smi) == Chem.CanonSmiles(example_smiles):
            print("Assertion test passed: Reconstructed SMILES matches original.")
        else:
            print("Warning: Reconstructed SMILES does not match original.")
    else:
        print("Tokenization failed for the example SMILES.")


if __name__ == "__main__":
    import sys

    # Add project-specific path only when running as a script
    sys.path.append("../psvae")
    Graph_BPE_to_get_vocab()
