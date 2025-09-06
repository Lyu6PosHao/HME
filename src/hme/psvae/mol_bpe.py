#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Graph-based BPE (Byte Pair Encoding) for molecule subgraph extraction.
Main function: generate vocab; auxiliary functions are used in other scripts.
"""
import sys
sys.path.append("/home/lvliuzhenghao/llzh/LightMoLlama/psvae")

import json
from copy import copy
import argparse
import multiprocessing as mp
from tqdm import tqdm

from hme.psvae.chem_utils import smi2mol, mol2smi, get_submol
from hme.psvae.chem_utils import cnt_atom, MAX_VALENCE
from hme.psvae.logger import print_log
from hme.psvae.molecule import Molecule
import selfies as sf
from rdkit import Chem


class MolInSubgraph:
    """Utility class for principal subgraph extraction from a molecule."""

    def __init__(self, mol, kekulize=False):
        """
        Initialize a molecule wrapper for subgraph extraction.

        Args:
            mol (RDKitMol): RDKit Mol object.
            kekulize (bool): Whether to kekulize the molecule.
        """
        self.mol = mol
        self.smi = mol2smi(mol)
        self.kekulize = kekulize
        self.subgraphs, self.subgraphs_smis = {}, {}
        for atom in mol.GetAtoms():
            idx, symbol = atom.GetIdx(), atom.GetSymbol()
            self.subgraphs[idx] = {idx: symbol}
            self.subgraphs_smis[idx] = symbol
        self.inversed_index = {}
        self.upid_cnt = len(self.subgraphs)
        for aid in range(mol.GetNumAtoms()):
            for key in self.subgraphs:
                subgraph = self.subgraphs[key]
                if aid in subgraph:
                    self.inversed_index[aid] = key
        self.dirty = True
        self.smi2pids = {}

    def get_nei_subgraphs(self):
        """
        Find neighboring subgraphs and possible merges.

        Returns:
            tuple: (list of new subgraphs, list of pid pairs to merge)
        """
        nei_subgraphs, merge_pids = [], []
        for key in self.subgraphs:
            subgraph = self.subgraphs[key]
            local_nei_pid = []
            for aid in subgraph:
                atom = self.mol.GetAtomWithIdx(aid)
                for nei in atom.GetNeighbors():
                    nei_idx = nei.GetIdx()
                    if nei_idx in subgraph or nei_idx > aid:
                        continue
                    local_nei_pid.append(self.inversed_index[nei_idx])
            local_nei_pid = set(local_nei_pid)
            for nei_pid in local_nei_pid:
                new_subgraph = copy(subgraph)
                new_subgraph.update(self.subgraphs[nei_pid])
                nei_subgraphs.append(new_subgraph)
                merge_pids.append((key, nei_pid))
        return nei_subgraphs, merge_pids
    
    def get_nei_smis(self):
        """
        Get SMILES of all possible neighboring subgraphs.

        Returns:
            list[str]: List of neighboring subgraph SMILES.
        """
        if self.dirty:
            nei_subgraphs, merge_pids = self.get_nei_subgraphs()
            nei_smis, self.smi2pids = [], {}
            for i, subgraph in enumerate(nei_subgraphs):
                submol = get_submol(self.mol, list(subgraph.keys()), kekulize=self.kekulize)
                smi = mol2smi(submol)
                nei_smis.append(smi)
                self.smi2pids.setdefault(smi, [])
                self.smi2pids[smi].append(merge_pids[i])
            self.dirty = False
        else:
            nei_smis = list(self.smi2pids.keys())
        return nei_smis

    def merge(self, smi):
        """
        Merge subgraphs corresponding to a given SMILES.

        Args:
            smi (str): SMILES of the subgraph to merge.
        """
        if self.dirty:
            self.get_nei_smis()
        if smi in self.smi2pids:
            merge_pids = self.smi2pids[smi]
            for pid1, pid2 in merge_pids:
                if pid1 in self.subgraphs and pid2 in self.subgraphs:
                    self.subgraphs[pid1].update(self.subgraphs[pid2])
                    self.subgraphs[self.upid_cnt] = self.subgraphs[pid1]
                    self.subgraphs_smis[self.upid_cnt] = smi
                    for aid in self.subgraphs[pid2]:
                        self.inversed_index[aid] = pid1
                    for aid in self.subgraphs[pid1]:
                        self.inversed_index[aid] = self.upid_cnt
                    del self.subgraphs[pid1]
                    del self.subgraphs[pid2]
                    del self.subgraphs_smis[pid1]
                    del self.subgraphs_smis[pid2]
                    self.upid_cnt += 1
        self.dirty = True

    def get_smis_subgraphs(self):
        """
        Get current subgraphs with their atom indices.

        Returns:
            list[tuple[str, list[int]]]: (subgraph SMILES, atom indices).
        """
        res = []
        for pid in self.subgraphs_smis:
            smi = self.subgraphs_smis[pid]
            group_dict = self.subgraphs[pid]
            idxs = list(group_dict.keys())
            res.append((smi, idxs))
        return res


def freq_cnt(mol):
    """
    Count frequency of neighboring subgraphs for one molecule.

    Args:
        mol (MolInSubgraph): Molecule wrapper.

    Returns:
        tuple: (dict of SMILES frequency, updated molecule)
    """
    freqs = {}
    nei_smis = mol.get_nei_smis()
    for smi in nei_smis:
        freqs.setdefault(smi, 0)
        freqs[smi] += 1
    return freqs, mol


def graph_bpe(fname, vocab_len, vocab_path, cpus, kekulize):
    """
    Extract principal subgraphs using a BPE-like algorithm.

    Args:
        fname (str): Path to molecule SMILES corpus.
        vocab_len (int): Target vocabulary size.
        vocab_path (str): Path to save vocabulary.
        cpus (int): Number of CPU workers.
        kekulize (bool): Whether to kekulize molecules.

    Returns:
        tuple: (selected SMILES list, detail dict of [atom count, frequency])
    """
    print_log(f'Loading mols from {fname} ...')
    with open(fname, 'r') as fin:
        smis = [line.strip() for line in fin.readlines()]
    mols = []
    for smi in tqdm(smis):
        try:
            mol = MolInSubgraph(smi2mol(smi, kekulize), kekulize)
            mols.append(mol)
        except Exception:
            print_log(f'Parsing {smi} failed. Skip.', level='ERROR')

    selected_smis, details = list(MAX_VALENCE.keys()), {}
    for atom in selected_smis:
        details[atom] = [1, 0]
    for smi in smis:
        cnts = cnt_atom(smi, return_dict=True)
        for atom in details:
            if atom in cnts:
                details[atom][1] += cnts[atom]

    add_len = vocab_len - len(selected_smis)
    print_log(f'Added {len(selected_smis)} atoms, {add_len} principal subgraphs to extract')
    pbar = tqdm(total=add_len)
    pool = mp.Pool(cpus)

    while len(selected_smis) < vocab_len:
        res_list = pool.map(freq_cnt, mols)
        freqs, mols = {}, []
        for freq, mol in res_list:
            mols.append(mol)
            for key in freq:
                freqs.setdefault(key, 0)
                freqs[key] += freq[key]

        max_cnt, merge_smi = 0, ''
        for smi in freqs:
            if freqs[smi] > max_cnt:
                max_cnt, merge_smi = freqs[smi], smi

        for mol in mols:
            mol.merge(merge_smi)
        if merge_smi in details:
            continue
        selected_smis.append(merge_smi)
        details[merge_smi] = [cnt_atom(merge_smi), max_cnt]
        pbar.update(1)

    pbar.close()
    print_log('sorting vocab by atom num')
    selected_smis.sort(key=lambda x: details[x][0], reverse=True)
    pool.close()
    with open(vocab_path, 'w') as fout:
        fout.write(json.dumps({'kekulize': kekulize}) + '\n')
        fout.writelines(list(map(lambda smi: f'{smi}\t{details[smi][0]}\t{details[smi][1]}\n', selected_smis)))
    return selected_smis, details


class Tokenizer:
    """Subgraph-based tokenizer for molecules."""

    def __init__(self, vocab_path):
        """
        Initialize the tokenizer with a given vocabulary.

        Args:
            vocab_path (str): Path to vocabulary file.
        """
        with open(vocab_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
        config = json.loads(lines[0])
        self.kekulize = config['kekulize']
        lines = lines[1:]
        
        self.vocab_dict = {}
        self.idx2subgraph, self.subgraph2idx = [], {}
        self.max_num_nodes = 0
        for idx, line in enumerate(lines):
            smi, atom_num, freq = line.strip().split('\t')
            self.vocab_dict[smi] = (int(atom_num), int(freq), int(idx))
            self.max_num_nodes = max(self.max_num_nodes, int(atom_num))
            self.subgraph2idx[smi] = len(self.idx2subgraph)
            self.idx2subgraph.append(smi)
        self.pad, self.end = '<pad>', '<s>'
        for smi in [self.pad, self.end]:
            self.subgraph2idx[smi] = len(self.idx2subgraph)
            self.idx2subgraph.append(smi)
        self.bond_start = '<bstart>'
        self.max_num_nodes += 2
    
    def tokenize(self, mol):
        """
        Tokenize a molecule into subgraph-level representation.

        Args:
            mol (str | RDKitMol): Molecule as SMILES or RDKit Mol.

        Returns:
            Molecule | list[Molecule] | None
        """
        smiles = mol
        if isinstance(mol, str):
            mol = smi2mol(mol, self.kekulize)
        else:
            smiles = mol2smi(mol)
        rdkit_mol = mol
        if rdkit_mol.GetNumAtoms() <= 1:
            return None
        if '.' in smiles:
            fragments = smiles.split('.')
            return [self.tokenize(frag) for frag in fragments]
        mol = MolInSubgraph(mol, kekulize=self.kekulize)
        while True:
            nei_smis = mol.get_nei_smis()
            max_freq, merge_smi = -1, ''
            for smi in nei_smis:
                if smi not in self.vocab_dict:
                    continue
                freq = self.vocab_dict[smi][1]
                if freq > max_freq:
                    max_freq, merge_smi = freq, smi
            if max_freq == -1:
                break
            mol.merge(merge_smi)
        res = mol.get_smis_subgraphs()
        aid2pid = {}
        for pid, subgraph in enumerate(res):
            _, aids = subgraph
            for aid in aids:
                aid2pid[aid] = pid
        ad_mat = [[0 for _ in res] for _ in res]
        for aid in range(rdkit_mol.GetNumAtoms()):
            atom = rdkit_mol.GetAtomWithIdx(aid)
            for nei in atom.GetNeighbors():
                nei_id = nei.GetIdx()
                i, j = aid2pid[aid], aid2pid[nei_id]
                if i != j:
                    ad_mat[i][j] = ad_mat[j][i] = 1
        group_idxs = [x[1] for x in res]
        return Molecule(rdkit_mol, group_idxs, self.kekulize)

    def idx_to_subgraph(self, idx):
        """Convert index to subgraph SMILES."""
        return self.idx2subgraph[idx]
    
    def subgraph_to_idx(self, subgraph):
        """Convert subgraph SMILES to index."""
        return self.subgraph2idx[subgraph]
    
    def pad_idx(self):
        """Return padding token index."""
        return self.subgraph2idx[self.pad]
    
    def end_idx(self):
        """Return end token index."""
        return self.subgraph2idx[self.end]
    
    def num_subgraph_type(self):
        """Return number of subgraph types in the vocab."""
        return len(self.idx2subgraph)
    
    def atom_pos_pad_idx(self):
        """Return padding index for atom positions."""
        return self.max_num_nodes - 1
    
    def atom_pos_start_idx(self):
        """Return start index for atom positions."""
        return self.max_num_nodes - 2

    def __call__(self, mol):
        return self.tokenize(mol)
    
    def __len__(self):
        return len(self.idx2subgraph)


def parse():
    """Parse command-line arguments for Graph BPE vocab extraction."""
    parser = argparse.ArgumentParser(description='Principal subgraph extraction with BPE')
    parser.add_argument('--smiles', type=str, default='Cc1c(Cl)cccc1-n1c(SCCNS(C)(=O)=O)nc2ccccc2c1=O',
                        help='Example molecule to tokenize')
    parser.add_argument('--data', type=str, default='/remote-home1/lihao/llzhs/tools/reverse_design_datasets/smiles_for_graphBPE.txt',
                        help='Path to molecule corpus')
    parser.add_argument('--vocab_size', type=int, default=800, help='Vocabulary size')
    parser.add_argument('--output', type=str, default='/remote-home1/lihao/llzhs/tools/reverse_design_datasets/vocab_1600.txt',
                        help='Path to save vocabulary')
    parser.add_argument('--workers', type=int, default=128, help='Number of CPU workers')
    parser.add_argument('--kekulize', action='store_false',
                        help='Kekulize molecules (replace aromatic bonds with alternating single/double bonds)')
    return parser.parse_args()


def Graph_BPE_to_get_vocab():
    """Run the Graph BPE pipeline to extract vocabulary and test tokenization."""
    args = parse()
    graph_bpe(args.data, vocab_len=args.vocab_size, vocab_path=args.output,
              cpus=args.workers, kekulize=args.kekulize)
    tokenizer = Tokenizer(args.output)
    print(f'Example: {args.smiles}')
    mol = tokenizer.tokenize(args.smiles)
    print('Tokenized mol:')
    print(mol)
    print('Reconstruct smiles to check correctness:')
    if isinstance(mol, list):
        smi = '.'.join([frag.to_smiles() for frag in mol])
    else:
        smi = mol.to_smiles()
        mol.to_SVG('example.svg')
    print(smi)
    assert smi == args.smiles
    print('Assertion test passed')


def remove_charge(mol):
    """
    Neutralize formal charges in a molecule.

    Args:
        mol (RDKitMol): Input molecule.

    Returns:
        RDKitMol: Charge-neutral molecule.
    """
    for atom in mol.GetAtoms():
        atom.SetFormalCharge(0)
    return mol


def remove_chiral(mol):
    """
    Remove chirality from a molecule.

    Args:
        mol (RDKitMol): Input molecule.

    Returns:
        RDKitMol: Molecule with unspecified chirality.
    """
    chirality = Chem.FindMolChiralCenters(mol)
    for atom, _ in chirality:
        mol.GetAtomWithIdx(atom).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return mol


def remove_radical_electrons(mol):
    """
    Remove radical electrons from a molecule.

    Args:
        mol (RDKitMol): Input molecule.

    Returns:
        RDKitMol: Molecule with zero radical electrons.
    """
    from rdkit.Chem.Descriptors import NumRadicalElectrons
    from rdkit.Chem import RWMol
    rwmol = RWMol(mol)
    for at in rwmol.GetAtoms():
        at.SetNumRadicalElectrons(0)
    return rwmol.GetMol()


if __name__ == '__main__':
    Graph_BPE_to_get_vocab()
