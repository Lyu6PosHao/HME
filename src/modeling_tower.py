from unimol_tools import UniMolRepr
from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import (MessagePassing, global_add_pool,
                                global_max_pool, global_mean_pool)
from torch_geometric.utils import  degree
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from util import smiles2graph
from typing import Union, Dict, Tuple,List
import numpy as np

class GINConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(GINConv, self).__init__(aggr=aggr)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        # WARN: some weird thing happend if excute in bfloat16, so we force to cast to float32
        dtype = x.dtype
        inter = (1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        if dtype == torch.bfloat16:
            inter = inter.float()
            out = self.mlp.float()(inter)
            out = out.to(dtype)
        else:
            out = self.mlp(inter)
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__(aggr=aggr)

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNN(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0., gnn_type="gin"):

        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        super(GNN, self).__init__()
        self.drop_ratio = drop_ratio
        self.num_layer = num_layer
        self.JK = JK

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.atom_encoder(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError("not implemented.")
        return node_representation


class GNN_graphpred(nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        arg.emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536 """

    def __init__(
        self, 
        emb_dim,  
        graph_pooling, 
        projection_dim:int=None,
        molecule_node_model=None,
        init_checkpoint=None,
    ):
        super(GNN_graphpred, self).__init__()

        self.molecule_node_model = molecule_node_model
        self.emb_dim = emb_dim

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
        
        if projection_dim is not None:
            self.projector = nn.Linear(emb_dim, projection_dim)
            self.output_dim = projection_dim
        else:
            self.projector = None
            self.output_dim = emb_dim
        
        if init_checkpoint is not None:
            self._load_state_dict(init_checkpoint, strict=False)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, None#data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_node_model(x, edge_index, edge_attr)
        graph_representation = None#self.pool(node_representation, batch) 
        return graph_representation, node_representation
    
    def encode_mol(self, mol, proj=False, return_node_feats=False, eval=True):
        if eval:
            self.molecule_node_model.eval() # hard code: set to eval mode
            with torch.no_grad():
                h_graph, h_node = self.forward(mol)
        else:
            self.molecule_node_model.train() # set to train mode
            h_graph, h_node = self.forward(mol)
        if proj and self.projector is not None:
            h_graph = self.projector(h_graph)
            h_node = self.projector(h_node)
        if return_node_feats:
            return h_graph, h_node
        else:
            return h_graph
    
    def _load_state_dict(self, model_file, strict=True):
        print("Loading from {} ...".format(model_file))
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict, strict=strict)
        return
    
    @property
    def dummy_feature(self):
        return self.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
    
    @property
    def hidden_size(self):
        return self.output_dim

#moleculeSTM
class Molecule2DTower(nn.Module):
    def __init__(self,config=None):
        super(Molecule2DTower, self).__init__()
        if config is not None:
            self.encoder= self.build_graph_tower(config)
        else:
            self.encoder= self.build_graph_tower(config)
        self.molecule_hidden_size=300
        self.eval()
        self.encoder.eval()
        
    def build_graph_tower(self,graph_tower_cfg):
        # actually, 'graph_tower_cfg' is identical to 'model_args'
        molecule_node_model = GNN(
            num_layer=5,#graph_tower_cfg.gin_num_layers, #5
            emb_dim=300,#graph_tower_cfg.gin_hidden_dim, #300
            JK='last', # default to 'last' 
            drop_ratio=0.1,#graph_tower_cfg.drop_ratio, #0.1
            gnn_type='gin', # default to 'gin'
        )
        return GNN_graphpred(
            emb_dim=300,#graph_tower_cfg.gin_hidden_dim, #300
            graph_pooling='mean',#graph_tower_cfg.graph_pooling, #mean
            molecule_node_model=molecule_node_model,
            init_checkpoint='./molecule_towers/molecule_model.pth'        #graph_tower_cfg.init_checkpoint
        )
    #encode one smiles into a vector
    def forward(self, batch:Union[Dict[str, str], str])->torch.Tensor:
        with torch.no_grad():
            smiles=batch['smiles'] if isinstance(batch,dict) else batch
            graph=smiles2graph(smiles)
            _,h_node=self.encoder.encode_mol(graph, proj=False, return_node_feats=True)
            return h_node
    
#unimol
class Molecule3DTower(nn.Module):
    def __init__(self,config=None):
        super(Molecule3DTower, self).__init__()
        self.encoder= UniMolRepr(data_type='molecule', remove_hs=False,use_gpu=True)
        self.molecule_hidden_size=512
        self.eval()

    #encode one smiles into a vector
    def forward(self, batch:Union[Dict[str, List], str,List[str]])->np.ndarray:
        with torch.no_grad():
            smiles = batch['smiles'] if isinstance(batch,dict) else batch
            embeddings_3d = self.encoder.get_repr(smiles,return_atomic_reprs=True)['atomic_reprs']
            return embeddings_3d
