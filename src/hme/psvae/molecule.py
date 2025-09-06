#!/usr/bin/python
# -*- coding:utf-8 -*-
from copy import copy, deepcopy
from typing import Union

import networkx as nx
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import Mol as RDKitMol
import numpy as np

from hme.psvae.chem_utils import smi2mol, mol2smi
from hme.psvae.chem_utils import get_submol, get_submol_atom_map


class SubgraphNode:
    """A node representing a molecular subgraph."""

    def __init__(self, smiles: str, pos: int, atom_mapping: dict, kekulize: bool):
        """
        Initialize a subgraph node.

        Args:
            smiles (str): SMILES representation of the subgraph.
            pos (int): Position index of the subgraph in the parent molecule.
            atom_mapping (dict): Mapping from atom indices in the original molecule
                                 to atom indices in the subgraph.
            kekulize (bool): Whether to kekulize the subgraph molecule.
        """
        self.smiles = smiles
        self.pos = pos
        self.mol = smi2mol(smiles, kekulize, sanitize=False)
        self.atom_mapping = copy(atom_mapping)
    
    def get_mol(self) -> RDKitMol:
        """Return the RDKit Mol object of the subgraph."""
        return self.mol

    def get_atom_mapping(self) -> dict:
        """Return a copy of the atom mapping dictionary."""
        return copy(self.atom_mapping)

    def __str__(self):
        return f"""
                    smiles: {self.smiles},
                    position: {self.pos},
                    atom map: {self.atom_mapping}
                """


class SubgraphEdge:
    """An edge connecting two molecular subgraphs."""

    def __init__(self, src: int, dst: int, edges: list):
        """
        Initialize a subgraph edge.

        Args:
            src (int): Index of the source subgraph.
            dst (int): Index of the destination subgraph.
            edges (list): List of tuples (a, b, bond_type) describing the bonds
                          connecting atoms between subgraphs.
        """
        self.edges = copy(edges)
        self.src = src
        self.dst = dst
        self.dummy = len(self.edges) == 0
    
    def get_edges(self) -> list:
        """Return a copy of the bond list connecting two subgraphs."""
        return copy(self.edges)
    
    def get_num_edges(self) -> int:
        """Return the number of bonds in this edge."""
        return len(self.edges)

    def __str__(self):
        return f"""
                    src subgraph: {self.src}, dst subgraph: {self.dst},
                    atom bonds: {self.edges}
                """


colors = [
    (247, 192, 197, 1),
    (255, 229, 190, 1),
    (197, 233, 199, 1),
    (190, 232, 240, 1),
    (191, 207, 228, 1),
    (226, 197, 222, 1),
    (217, 217, 217, 1)
]
colors = [(r / 255, g / 255, b / 255, a) for r, g, b, a in colors]


class Molecule(nx.Graph):
    """Molecule represented at the subgraph level, extending networkx.Graph."""

    def __init__(self, mol: Union[str, RDKitMol] = None, groups: list = None, kekulize: bool = False):
        """
        Initialize a molecule from RDKit Mol or SMILES and subgraph partitions.

        Args:
            mol (str | RDKitMol): Molecule input (SMILES string or RDKit Mol).
            groups (list): List of atom groups defining subgraphs.
            kekulize (bool): Whether to kekulize subgraphs.
        """
        super().__init__()
        if mol is None:
            return
        if isinstance(mol, str):
            smiles, rdkit_mol = mol, smi2mol(mol, kekulize)
        else:
            smiles, rdkit_mol = mol2smi(mol), mol
        self.graph['smiles'] = smiles

        aid2pos = {}
        for pos, group in enumerate(groups):
            for aid in group:
                aid2pos[aid] = pos
            subgraph_mol = get_submol(rdkit_mol, group, kekulize)
            subgraph_smi = mol2smi(subgraph_mol)
            atom_mapping = get_submol_atom_map(rdkit_mol, subgraph_mol, group, kekulize)
            node = SubgraphNode(subgraph_smi, pos, atom_mapping, kekulize)
            self.add_node(node)

        edges_arr = [[[] for _ in groups] for _ in groups]
        for edge_idx in range(rdkit_mol.GetNumBonds()):
            bond = rdkit_mol.GetBondWithIdx(edge_idx)
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            begin_subgraph_pos = aid2pos[begin]
            end_subgraph_pos = aid2pos[end]
            begin_mapped = self.nodes[begin_subgraph_pos]['subgraph'].atom_mapping[begin]
            end_mapped = self.nodes[end_subgraph_pos]['subgraph'].atom_mapping[end]
            bond_type = bond.GetBondType()
            edges_arr[begin_subgraph_pos][end_subgraph_pos].append((begin_mapped, end_mapped, bond_type))
            edges_arr[end_subgraph_pos][begin_subgraph_pos].append((end_mapped, begin_mapped, bond_type))

        for i in range(len(groups)):
            for j in range(len(groups)):
                if not i < j or len(edges_arr[i][j]) == 0:
                    continue
                edge = SubgraphEdge(i, j, edges_arr[i][j])
                self.add_edge(edge)
    
    @classmethod
    def from_nx_graph(cls, graph: nx.Graph, deepcopy=True):
        """Convert a networkx Graph into a Molecule object."""
        if deepcopy:
            graph = deepcopy(graph)
        graph.__class__ = Molecule
        return graph

    @classmethod
    def merge(cls, mol0, mol1, edge=None):
        """
        Merge two Molecule objects into one, connected by a given edge.

        Args:
            mol0 (Molecule): First molecule.
            mol1 (Molecule): Second molecule.
            edge (SubgraphEdge): Edge connecting the two molecules.

        Returns:
            Molecule: Merged molecule.
        """
        node_mappings = [{}, {}]
        mols = [mol0, mol1]
        mol = Molecule.from_nx_graph(nx.Graph())
        for i in range(2):
            for n in mols[i].nodes:
                node_mappings[i][n] = len(node_mappings[i])
                node = deepcopy(mols[i].get_node(n))
                node.pos = node_mappings[i][n]
                mol.add_node(node)
            for src, dst in mols[i].edges:
                edge = deepcopy(mols[i].get_edge(src, dst))
                edge.src = node_mappings[i][src]
                edge.dst = node_mappings[i][dst]
                mol.add_edge(src, dst, connects=edge)

        edge = deepcopy(edge)
        edge.src = node_mappings[0][edge.src]
        edge.dst = node_mappings[1][edge.dst]
        mol.add_edge(edge)
        return mol

    def get_edge(self, i, j) -> SubgraphEdge:
        """Return the SubgraphEdge object between nodes i and j."""
        return self[i][j]['connects']
    
    def get_node(self, i) -> SubgraphNode:
        """Return the SubgraphNode object at index i."""
        return self.nodes[i]['subgraph']

    def add_edge(self, edge: SubgraphEdge) -> None:
        """Add a SubgraphEdge to the Molecule."""
        src, dst = edge.src, edge.dst
        super().add_edge(src, dst, connects=edge)
    
    def add_node(self, node: SubgraphNode) -> None:
        """Add a SubgraphNode to the Molecule."""
        n = node.pos
        super().add_node(n, subgraph=node)

    def subgraph(self, nodes: list):
        """Return a Molecule subgraph induced by the given nodes."""
        graph = super().subgraph(nodes)
        assert isinstance(graph, Molecule)
        return graph

    def to_rdkit_mol(self) -> RDKitMol:
        """
        Convert the Molecule into an RDKit Mol object.

        Returns:
            RDKitMol: Reconstructed molecule.
        """
        mol = Chem.RWMol()
        aid_mapping, order = {}, []

        for n in self.nodes:
            subgraph = self.get_node(n)
            submol = subgraph.get_mol()
            local2global = {subgraph.atom_mapping[g]: g for g in subgraph.atom_mapping}
            for atom in submol.GetAtoms():
                mol.AddAtom(atom)
                aid_mapping[(n, atom.GetIdx())] = len(aid_mapping)
                order.append(local2global[atom.GetIdx()])
            for bond in submol.GetBonds():
                begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                begin, end = aid_mapping[(n, begin)], aid_mapping[(n, end)]
                mol.AddBond(begin, end, bond.GetBondType())

        for src, dst in self.edges:
            subgraph_edge = self.get_edge(src, dst)
            pid_src, pid_dst = subgraph_edge.src, subgraph_edge.dst
            for begin, end, bond_type in subgraph_edge.edges:
                begin, end = aid_mapping[(pid_src, begin)], aid_mapping[(pid_dst, end)]
                mol.AddBond(begin, end, bond_type)

        mol = mol.GetMol()
        new_order = [-1 for _ in order]
        for cur_i, ordered_i in enumerate(order):
            new_order[ordered_i] = cur_i
        mol = Chem.RenumberAtoms(mol, new_order)

        mol.UpdatePropertyCache(strict=False)
        ps = Chem.DetectChemistryProblems(mol)
        if not ps:
            Chem.SanitizeMol(mol)
            return mol
        for p in ps:
            if p.GetType() == 'AtomValenceException':
                at = mol.GetAtomWithIdx(p.GetAtomIdx())
                if at.GetAtomicNum() == 7 and at.GetFormalCharge() == 0 and at.GetExplicitValence() == 4:
                    at.SetFormalCharge(1)
        Chem.SanitizeMol(mol)
        return mol
    
    def to_SVG(self, path= None, size: tuple = (200, 200), add_idx=False) -> str:
        """
        Render the Molecule as an SVG image.

        Args:
            path (str): Output file path.
            size (tuple): Image size (width, height).
            add_idx (bool): Whether to annotate atoms with indices.

        Returns:
            str: SVG string content.
        """
        mol = self.to_rdkit_mol()
        if add_idx:
            for i in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(i)
                atom.SetAtomMapNum(i)
        tm = rdMolDraw2D.PrepareMolForDrawing(mol)
        view = rdMolDraw2D.MolDraw2DSVG(*size)
        option = view.drawOptions()
        option.useBWAtomPalette()
        option.legendFontSize = 18
        option.bondLineWidth = 1
        option.highlightBondWidthMultiplier = 20

        sg_atoms, sg_bonds = [], []
        atom2subgraph, atom_color, bond_color = {}, {}, {}

        for i in self.nodes:
            node = self.get_node(i)
            color = colors[np.random.choice(7)]
            for atom_id in node.atom_mapping:
                sg_atoms.append(atom_id)
                atom2subgraph[atom_id] = i
                atom_color[atom_id] = color

        for bond_id in range(mol.GetNumBonds()):
            bond = mol.GetBondWithIdx(bond_id)
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if atom2subgraph[begin] == atom2subgraph[end]:
                sg_bonds.append(bond_id)
                bond_color[bond_id] = tuple(list(atom_color[begin])[:-1] + [1])
                
        view.DrawMolecules([tm], highlightAtoms=[sg_atoms],
                           highlightBonds=[sg_bonds],
                           highlightAtomColors=[atom_color],
                           highlightBondColors=[bond_color])
        view.FinishDrawing()
        svg = view.GetDrawingText()
        if path is not None:
            with open(path, 'w') as fout:
                fout.write(svg)
        return svg

    def to_smiles(self) -> str:
        """Convert the Molecule into a SMILES string."""
        rdkit_mol = self.to_rdkit_mol()
        return mol2smi(rdkit_mol)

    def __str__(self):
        desc = 'nodes: \n'
        for ni, node in enumerate(self.nodes):
            desc += f'{ni}:{self.get_node(node)}\n'
        desc += 'edges: \n'
        for src, dst in self.edges:
            desc += f'{src}-{dst}:{self.get_edge(src, dst)}\n'
        return desc
