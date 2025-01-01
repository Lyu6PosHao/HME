#the main function is used to convert raw data to data with fragment and skeleton
from psvae.mol_bpe import Tokenizer
import json
import torch
import numpy as np
import random
from tqdm import tqdm
import pickle
from rdkit import Chem

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(42)
def get_fragments(mol_list):
        result=[]
        #get fragments of one mol
        for mol in mol_list:
            if mol is None:
                continue
            fragments=''
            for t,node in enumerate(mol.nodes):
                fragments+=f'<|{mol.get_node(node).smiles}|>'
            result.append(fragments)
        #use '.' to connect different mols
        result='.'.join(result)
        return result
                
#convert raw file to standard format, which is a list of dict, each dict contains {"smiles","selfies","description","fragments","metadata"}
def convert_raw_to_standard_format(in_file,vocab_file,out_suffix):
    def get_already_have(in_file):
        ftype=in_file.split('.')[-1]
        if ftype=='json':
            with open(in_file, "r") as f:
                json_list=json.load(f)
        elif ftype=='pkl':
            with open(in_file, "rb") as f:
                json_list=pickle.load(f)
        elif ftype=='txt':
            with open(in_file, "r") as f:
                json_list=f.readlines()
                json_list=[
                    {"smiles":item.strip()} for item in json_list
                ]
        else:
            raise ValueError
        return json_list,ftype
    
    def get_skeleton(mol_list):
        result=[]
        for mol in mol_list:
            skeleton=''
            for src,dst in mol.edges:
                skeleton+=f'<|{int(src)}|><|{int(dst)}|>$'
            else:
                skeleton=skeleton[:-1]
            result.append(skeleton)
        result='.'.join(result)
        result=f"{result}"
        return result
    mol_tokenizer=Tokenizer(vocab_file)
    json_list,ftype=get_already_have(in_file=in_file)
    bad_mol=0
    for item in tqdm(json_list,desc=in_file.split('/')[-3]+in_file.split('/')[-2]+in_file.split('/')[-1]):
        smiles=item['smiles']
        if "<iamsplit>" in smiles:
            smiles=smiles.split("<iamsplit>")
            assert len(smiles)==2
        else:
            smiles=[smiles]
        assert type(smiles)==list
        item_smiles=[]
        item_fragments=[]
        for smi in smiles:
            try:
                smi=Chem.CanonSmiles(smi)
                mol_list=mol_tokenizer.tokenize(smi)
            except:
                bad_mol+=1
                break
            mol_list = [mol_list] if not isinstance(mol_list, list) else mol_list
            item_fragments.append(get_fragments(mol_list))
            item_smiles.append(smi)
        item['fragments']='<iamsplit>'.join(item_fragments)
        item['smiles']='<iamsplit>'.join(item_smiles)
        item['selfies']=''

    if ftype=='pkl':
        with open(in_file+out_suffix+'.pkl', "wb") as f:
            pickle.dump(json_list,f)
    elif ftype=='json':
        with open(in_file+out_suffix+'.json', "w") as f:
            json.dump(json_list, f, indent=4)
    elif ftype=='txt':
        with open(in_file+out_suffix+'.txt', "w") as f:
            for item in json_list:
                f.write(item['smiles']+'\n'+item['fragments']+'\n')
    else:
        raise ValueError
    print(bad_mol)

def add_selfies(in_file,in_file_biot5):
    json_list=[]
    json_list_biot5=[]
    with open(in_file, "r") as f:
        json_list=json.load(f)
    with open(in_file_biot5, "r") as f:
        json_list_biot5=json.load(f)
    for idx,item in enumerate(tqdm(json_list)):
        assert item['description']==json_list_biot5[idx]['description']
        item['selfies']=json_list_biot5[idx]['selfies']
    with open(in_file, "w") as f:
        json.dump(json_list, f, indent=4)

def get_frg_from_one_smiles(smiles,vocab_file,verbose:bool=False):
    mol_tokenizer=Tokenizer(vocab_file)
    try:
        can_smiles=Chem.CanonSmiles(smiles)
        mol_list=mol_tokenizer.tokenize(can_smiles)
    except:
        if verbose:
            print(f'Some error occurs when get frgs from smiles:{smiles}')
        return '',smiles
    mol_list = [mol_list] if not isinstance(mol_list, list) else mol_list
    fragments=get_fragments(mol_list)
    return fragments,can_smiles,mol_list

def check_conditional_frgs_and_true_frgs(conditional_frgs:str,gen_smiles:str):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('path/to/HME/tokenizer')
    gen_frgs=get_frg_from_one_smiles(gen_smiles)[0]
    return tokenizer.tokenize(conditional_frgs),tokenizer.tokenize(gen_frgs)
    
    
if __name__ == '__main__':
    convert_raw_to_standard_format(in_file='json_file_path',
    vocab_file='path/to/fragment_vocabulary.txt',
    out_suffix='frg')