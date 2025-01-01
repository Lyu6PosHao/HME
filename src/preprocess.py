import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from modeling_tower import Molecule3DTower, Molecule2DTower
import json
import pickle
from rdkit.Chem import CanonSmiles
import torch
from tqdm import tqdm
from unimol_tools import UniMolRepr
from frg import get_frg_from_one_smiles

#you may modify the following function to get the content of your file
#suppose the file is a json file
def get_json_list(file_path):
    with open(file_path) as f:
        json_list = json.load(f)
    return json_list

def get_fragments_and_can_smiles(file_path, output_file_path):
    # get smiles list
    json_list = get_json_list(file_path)
    print('total smiles:', len(json_list))
    
    # get fragments and canonical smiles
    for item in tqdm(json_list):
        frag,can_smiles=get_frg_from_one_smiles(item['smiles'])
        item['fragments']=frag
        item['smiles']=can_smiles
    print("failed fragments:", len([item for item in json_list if item['fragments']=='']))

    # save to file
    with open(output_file_path, "w") as f:
        json.dump(json_list, f, indent=4)
    print("all ended!")


def get_2d3d_tensors(file_path):
    # get smiles list
    json_list = get_json_list(file_path)
    smiles_list = [item['smiles'] for item in json_list]
    print('total smiles:', len(smiles_list))
    
    # init towers
    molecule_3d_tower = UniMolRepr(data_type='molecule', remove_hs=False, use_gpu=False)
    molecule_2d_tower = Molecule2DTower(config=None)
    
    # get 2d features
    molecule_raw_2d_features = []
    for item in tqdm(smiles_list):
        try:
            molecule_raw_2d_features.append(
                torch.tensor(molecule_2d_tower(item))  # hidden_size=300
            )
        except:
            molecule_raw_2d_features.append(None)
    print("failed 2d:", len([item for item in molecule_raw_2d_features if item is None]))
    print('size of one sample:',molecule_raw_2d_features[0].shape)
    continue_flag=input('if continue?')
    if continue_flag=='1':
        pass
    else:
        assert 0
    # get 3d features
    molecule_raw_3d_features = []
    split=1024
    for i in range(0, len(smiles_list), split):
        if i+split > len(smiles_list):
            repr_result = molecule_3d_tower.get_repr(smiles_list[i:], return_atomic_reprs=True)['atomic_reprs']
        else:
            repr_result = molecule_3d_tower.get_repr(smiles_list[i:i+split], return_atomic_reprs=True)['atomic_reprs']
        molecule_raw_3d_features.extend(repr_result)
        print(f'processed: {i/len(smiles_list):.2f}')
    molecule_raw_3d_features=[torch.tensor(x) for x in molecule_raw_3d_features]
    print('size of one sample:',molecule_raw_3d_features[0].shape)
    assert len(molecule_raw_3d_features) == len(smiles_list)
    print('total 3d features:', len(molecule_raw_3d_features))

    #save tensors
    torch.save(molecule_raw_2d_features,os.path.join(file_path+'.2d.pt'))
    torch.save(molecule_raw_3d_features,os.path.join(file_path+'.3d.pt'))
    print(f'2d3d have been saved to {file_path}.2d.pt and {file_path}.3d.pt')

if __name__ == '__main__':
    get_2d3d_tensors('path/to/your/json/file')
