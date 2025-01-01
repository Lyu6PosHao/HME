
import torch.nn as nn
import torch,os
import random
import numpy as np
from rdkit import Chem
import numpy as np
from typing import Dict
from rdkit.Chem.rdchem import BondType, BondDir, ChiralType
from torch_geometric.data import Data
from transformers import AutoTokenizer, LlamaForCausalLM,AutoModelForCausalLM
from modeling_llava import MoLlamaForConditionalGeneration
from configuration_llava import MoLlamaConfig
import logging
logger = logging.getLogger(__name__)
BOND_TYPE = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3}
BOND_DIR = {BondDir.NONE: 0, BondDir.ENDUPRIGHT: 1, BondDir.ENDDOWNRIGHT: 2}
CHI = {ChiralType.CHI_UNSPECIFIED: 0, ChiralType.CHI_TETRAHEDRAL_CW: 1, ChiralType.CHI_TETRAHEDRAL_CCW: 2, ChiralType.CHI_OTHER: 3}

def set_seed(seed: int=42) -> None:
    r"""
    Sets the seed for generating random numbers.

    Args:
        seed (int): The seed to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
# copy code from https://github.com/huggingface/peft/blob/2f5360a7da22a236b5ad4c059572fff5321c867c/src/peft/peft_model.py#L617
def get_nb_trainable_parameters(model:nn.Module) -> tuple[int, int]:
    r"""
    Returns the number of trainable parameters and the number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "element_size"):
                num_bytes = param.element_size()
            elif not hasattr(param, "quant_storage"):
                num_bytes = 1
            else:
                num_bytes = param.quant_storage.itemsize
            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


# copy code from https://github.com/huggingface/peft/blob/2f5360a7da22a236b5ad4c059572fff5321c867c/src/peft/peft_model.py#L647
def print_trainable_parameters(model: nn.Module) -> None:
    """
    Prints the number of trainable parameters in the model.

    Note: print_trainable_parameters() uses get_nb_trainable_parameters() which is different from
    num_parameters(only_trainable=True) from huggingface/transformers. get_nb_trainable_parameters() returns
    (trainable parameters, all parameters) of the Peft Model which includes modified backbone transformer model.
    For techniques like LoRA, the backbone transformer model is modified in place with LoRA modules. However, for
    prompt tuning, the backbone transformer model is unmodified. num_parameters(only_trainable=True) returns number
    of trainable parameters of the backbone transformer model which can be different.
    """
    trainable_params, all_param = get_nb_trainable_parameters(model)

    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
    )



def bond_dir(bond):
    d = bond.GetBondDir()
    return BOND_DIR[d]

def bond_type(bond):
    t = bond.GetBondType()
    return BOND_TYPE[t]

def atom_chiral(atom):
    c = atom.GetChiralTag()
    return CHI[c]

def atom_to_feature(atom):
    num = atom.GetAtomicNum() - 1
    if num == -1:
        # atom.GetAtomicNum() is 0, which is the generic wildcard atom *, may be used to symbolize an unknown atom of any element.
        # See https://biocyc.org/help.html?object=smiles
        num = 118  # normal num is [0, 117], so we use 118 to denote wildcard atom *
    return [num, atom_chiral(atom)]

def bond_to_feature(bond):
    return [bond_type(bond), bond_dir(bond)]

def smiles2graph(smiles_string)->Dict:
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 2
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)
    return Data(
            x=torch.asarray(graph['node_feat']),
            edge_attr=torch.asarray(graph['edge_feat']),
            edge_index=torch.asarray(graph['edge_index']),
        )
    


#from haotian_liu's llava
def smart_tokenizer_and_embedding_resize(
    tokenizer,
    model,
    num_new_tokens: int,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    if num_new_tokens <0:
        raise ValueError("num_new_tokens must be non-negative")
    if num_new_tokens == 0:
        logging.info("No new tokens added to the tokenizer and embedding")
        return
    model.resize_token_embeddings(len(tokenizer))

    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
        dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
        dim=0, keepdim=True)

    input_embeddings[-num_new_tokens:] = input_embeddings_avg
    output_embeddings[-num_new_tokens:] = output_embeddings_avg
    logging.info(f"{num_new_tokens} new tokens added to the tokenizer; embed_tokens and lm_head are resized")

def get_frg_list(vocab_file):
    logger.warning(f"vocab_file: {vocab_file.split('/')[-1]} is used to load fragment list!!!")
    from periodictable import elements
    MAX_VALENCE =  {element.symbol: 10 for element in elements}
    
    with open(vocab_file) as f:
        lines = f.readlines()
    frg_list=[]
    for line in lines:
        line=line.strip().split('\t')
        if len(line)==3: #indicate that it is not the first line used for kekulize
            if line[0] in MAX_VALENCE:  #fragment with only one atom
                frg_list.append(f"<|[{line[0]}]|>")
            else:  #fragment with more than one atom
                frg_list.append(f"<|{line[0]}|>")
    assert len(frg_list)==len(lines) or len(frg_list)==len(lines)-1
    return frg_list



def load_mollama(modelargs):
    #load language_model from checkpoint
    language_model=LlamaForCausalLM.from_pretrained(
        modelargs.model_name_or_path,
        torch_dtype=torch.bfloat16,
        #attn_implementation='flash_attention_2',
        #device_map={"":int(os.environ.get("LOCAL_RANK") or 0)},
        )
    
    #load tokenizer from checkpoint
    #autotokenizer is Fasttokenizer by default, which will cause the extension of the vocabulary to be invalid for Llama2; but this problem will not occur for llama3, which is a bit strange
    from transformers import LlamaTokenizer
    if 'llama2' in modelargs.model_name_or_path.lower() or 'llama-2' in modelargs.model_name_or_path.lower():
        tokenizer=LlamaTokenizer.from_pretrained(modelargs.model_name_or_path)
    else:
        tokenizer=AutoTokenizer.from_pretrained(modelargs.model_name_or_path)

    #If feature_fuser.pth exists, it means that the model checkpoint is not the original llama2 or llama3
    if os.path.exists(os.path.join(modelargs.model_name_or_path,'feature_fuser.pth')):
        logging.info('Load feature_fuser.pth successfully. Assume that the checkpoint is HME.')
        config=MoLlamaConfig(text_config=language_model.config,
                            molecule_2d_hidden_size=300,
                            molecule_3d_hidden_size=512,
                            ignore_index=-100,
                            molecule_token_index=tokenizer.convert_tokens_to_ids("<molecule>"),
        )
        model=MoLlamaForConditionalGeneration(config,language_model)
        model.feature_fuser.load_state_dict(torch.load(
            os.path.join(modelargs.model_name_or_path,'feature_fuser.pth')
            ))
        
    else:
        logging.info('Load feature_fuser.pth failed. Assume that the checkpoint is original Llama.')
        ori_length=len(tokenizer)
        tokenizer.add_tokens(get_frg_list())
        tokenizer.add_tokens(["<molecule>"])
        tokenizer.add_special_tokens(
            {"pad_token":"<pad>"}
            )
        language_model.config.pad_token_id=tokenizer.pad_token_id
        language_model.config.vocab_size=len(tokenizer)
        config=MoLlamaConfig(text_config=language_model.config,
                            molecule_2d_hidden_size=300,
                            molecule_3d_hidden_size=512,
                            ignore_index=-100,
                            molecule_token_index=tokenizer.convert_tokens_to_ids("<molecule>"),
        )
        model=MoLlamaForConditionalGeneration(config,language_model)
        smart_tokenizer_and_embedding_resize(tokenizer=tokenizer,model=model,num_new_tokens=len(tokenizer)-ori_length)
    model.feature_fuser=model.feature_fuser.bfloat16()
    return model,tokenizer
