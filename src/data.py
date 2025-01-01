import pickle
import torch
from numpy import ndarray
from torch.utils.data import Dataset
from typing import Tuple,List,Union, Optional,Dict
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType
import json
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
import re

class MoLlamaDataset(Dataset):
    def __init__(self, data_path,task_type,data_type,val=False):
        super().__init__()  
        self.data_path = data_path
        self.task_type=task_type
        self.data_type=data_type.split(',')
        
        if len(self.data_type)==0:
            raise ValueError('No data type provided.')
        for d in self.data_type:
            if d not in ['1d','2d','3d','frg']:
                raise ValueError(f'Unsupported data type: {d}')
            
        self.__load_raw_dataset(self.data_path,val=val) 
        self.__clear_frg()   
    def __load_raw_dataset(self, data_path: str,val:bool) -> List[dict]:
        file_type=data_path.split('.')[-1]
        assert file_type =='json'
        with open(data_path, 'r') as f:
            self.json_list = json.load(f)
        if '2d' in self.data_type:
            try:
                self.pt_2d_list=torch.load(data_path[:-5]+'.2d.pt')
            except:
                self.pt_2d_list=torch.load(data_path+'.2d.pt')
            #replace None with random tensor
            self.pt_2d_list=[i if i is not None else torch.randn(1,300) for i in self.pt_2d_list]
        else:
            self.pt_2d_list=[None]*len(self.json_list)
        if '3d' in self.data_type:
            try:
                self.pt_3d_list=torch.load(data_path[:-5]+'.3d.pt')
            except:
                self.pt_3d_list=torch.load(data_path+'.3d.pt')
            #replace None with random tensor
            self.pt_3d_list=[i if i is not None else torch.randn(1,512) for i in self.pt_3d_list]
        else:
            self.pt_3d_list=[None]*len(self.json_list)
        if val:
            self.json_list=self.json_list[:10000]
            self.pt_2d_list=self.pt_2d_list[:10000]
            self.pt_3d_list=self.pt_3d_list[:10000]

    def __len__(self) -> int:
        return len(self.json_list)

    def __sequence_format(self,seq:str,smiles:str,fragments:str)->str:
        if '2d' in self.data_type or '3d' in self.data_type:
            seq=f'{seq} Molecular features are: <molecule>'
        if '1d' in self.data_type:
            seq=f'{seq} Molecular SMILES is: {smiles}'
        if 'frg' in self.data_type:
            seq=f'{seq} Molecular fragments are: {fragments}'
        return seq
    def __clear_frg(self)->str:
        if not 'fragments' in self.json_list[0]:
            return
        for item in self.json_list:
            #extract each fragment from a string #example: '<|Cc1cccc(C)c1|><|NC(=O)CO|>' -> ['Cc1cccc(C)c1','NC(=O)CO']
            frg_without_ffix=re.findall(r'(?<=<\|).+?(?=\|>)',item['fragments'])
            #delete duplicate fragments
            frg_without_ffix=list(set(frg_without_ffix))
            #add '<|' and '|>' to each fragment
            frg_with_ffix=[f'<|{i}|>' for i in frg_without_ffix]
            #sort
            frg_with_ffix.sort()
            #join into a string
            item['fragments']=''.join(frg_with_ffix)
            if item['fragments']=='':
                item['fragments']='None'
    @torch.no_grad()
    def __getitem__(self, idx: int) -> Tuple[str, str, torch.FloatTensor, torch.FloatTensor]:
        item = self.json_list[idx]
        item_2d=self.pt_2d_list[idx]
        item_3d=self.pt_3d_list[idx]
        if self.task_type=='qa': #qa task
            (quesiton, answer, molecule_raw_2d_features,molecule_raw_3d_features)=(
                self.__sequence_format(item['instruction'],item['smiles'],item['fragments']),
                item['output'],
                item_2d,
                item_3d
            )

        elif self.task_type=='caption': #captioning task
            (quesiton, answer, molecule_raw_2d_features,molecule_raw_3d_features) = (
                self.__sequence_format("Please describe the molecule:",item['smiles'],item['fragments']),
                item['description'],
                item_2d,
                item_3d
            )
        elif self.task_type=='pretrain': #pretraining
            (quesiton, answer, molecule_raw_2d_features,molecule_raw_3d_features) = (
                self.__sequence_format("",item['smiles'],item['fragments']),
                item['enriched_description'],
                item_2d,
                item_3d
            )

        elif self.task_type=='text2smi': #text2smi
            (quesiton, answer, molecule_raw_2d_features,molecule_raw_3d_features) = (
                f"Please give me molecular SMILES based on the description: {item['description']}",
                f"{item['smiles']}",
                item_2d,
                item_3d
            )
        elif self.task_type=='text2frgsmi': #description-based molecular generation task  #ligand design task #Here fragment as Chain of Thought
            (quesiton, answer, molecule_raw_2d_features,molecule_raw_3d_features) = (
                f"Please give me molecular fragments based on the description. And then give me the molecular SMILES based on both the fragments and the description. The description is: {item['description']}",
                f"Molecular fragments are: {item['fragments']} Molecular SMILES is: {item['smiles']}",
                item_2d,
                item_3d
            )
        elif self.task_type=='textfrg2smi':  #=#multi-objective molecular reverse design task  #Here fragment as condition
            (quesiton, answer, molecule_raw_2d_features,molecule_raw_3d_features) = (
                f"There are some conditions, including logp (the hydrophobicity and solubility balance), qed (the drug-likeness), sas (the synthetic accessibility score), and the fragments (include specific fragments). Now please design a molecule under the given constraints: {item['description']}",
                f"{item['output']}" ,
                item_2d,
                item_3d
            )

        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")
        return [quesiton, answer, 
                molecule_raw_2d_features.bfloat16() if molecule_raw_2d_features is not None else molecule_raw_2d_features,
                molecule_raw_3d_features.bfloat16() if molecule_raw_3d_features is not None else molecule_raw_3d_features,
        ]
            

#Mainly to tokenize the text with various parameters, and return the molecule features as they are (return None if None)
class MoLlamaProcessor:
    def __init__(self, tokenizer,max_length):
        self.tokenizer = tokenizer
        self.max_length=max_length
    @torch.no_grad()
    def __call__(self, text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
                 molecule_raw_2d_features: torch.FloatTensor= None,
                 molecule_raw_3d_features: torch.FloatTensor= None,
                 padding: Union[bool, str, PaddingStrategy] = False,
                 truncation: Union[bool, str, TruncationStrategy] = None,
                 max_length=None,
                 add_special_tokens: bool = False,
                 return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
                 ) -> BatchFeature:
        if text is not None:
            text_inputs = self.tokenizer(
                text, return_tensors=return_tensors, padding=padding, truncation=True if self.max_length>0 else truncation, 
                max_length=self.max_length if self.max_length>0 else max_length,add_special_tokens=add_special_tokens
            )
        else:
            raise ValueError('text is required')

        return BatchFeature(data={**text_inputs, 
                                "molecule_raw_2d_features": molecule_raw_2d_features, #[num_atoms, hidden_size]
                                "molecule_raw_3d_features": molecule_raw_3d_features}) #[num_atoms, hidden_size]
    @torch.no_grad()
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)
    @torch.no_grad()
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
    
@dataclass
class MoleculeQAObject:
    q_input_ids: torch.Tensor
    a_input_ids: torch.Tensor
    molecule_raw_2d_features: torch.Tensor
    molecule_raw_3d_features: torch.Tensor
    
@torch.no_grad()
def build_molecule_qa_input(processor:MoLlamaProcessor,
                            question:str,
                            answer:str,
                            molecule_raw_2d_features:torch.Tensor=None,
                            molecule_raw_3d_features:torch.Tensor=None,
                            )->MoleculeQAObject:
    messages = [
        {"role": "system", "content": "You are a chemist."},
        {"role": "user", "content": question},
    ]
    prompt=processor.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,add_special_tokens=False)
    inputs=processor(text=prompt,
                    molecule_raw_2d_features=molecule_raw_2d_features,
                    molecule_raw_3d_features=molecule_raw_3d_features,
                    return_tensors='pt',
                    )
    #no need to return molecule_raw_2d_features and molecule_raw_3d_features
    outputs=processor(text=answer,
                    return_tensors='pt',
                    )
    # #for debug
    # if not (inputs['input_ids'].dtype==torch.int64 and outputs['input_ids'].dtype==torch.int64):
    #     print(inputs['input_ids'])
    #     print(outputs['input_ids'])
    #     raise ValueError('input_ids and output_ids should be int64')
    
    return MoleculeQAObject(
        q_input_ids=inputs['input_ids'],
        a_input_ids=outputs['input_ids'],
        molecule_raw_2d_features=inputs['molecule_raw_2d_features'],
        molecule_raw_3d_features=inputs['molecule_raw_3d_features']
    )


class TrainMoLlamaCollator:
    def __init__(self, processor:MoLlamaProcessor,IGNORE_INDEX:int)->None:
        self.processor = processor
        self.ignore_index = IGNORE_INDEX
        self.mol_padding=0.
    @torch.no_grad()
    def convert_one_piece(
            self,mqa_obj:MoleculeQAObject
    ):
        
        input_ids = torch.concat([
            mqa_obj.q_input_ids,
            mqa_obj.a_input_ids,
            torch.tensor(self.processor.tokenizer.eos_token_id,dtype=torch.int64).view(1,-1)
            ],
            dim=1
        )
        labels = torch.concat([
            torch.full(mqa_obj.q_input_ids.shape, self.ignore_index),  #the part corresponding to 'q' is ignore_index (-100)
            mqa_obj.a_input_ids,                                       #the part corresponding to 'a' is 'a' itself
            torch.tensor(self.processor.tokenizer.eos_token_id,dtype=torch.int64).view(1,-1)  #the part corresponding to 'eos' is 'eos' itself
            ],
            dim=1
        )
        
        return input_ids.squeeze(0),labels.squeeze(0),mqa_obj.molecule_raw_2d_features,mqa_obj.molecule_raw_3d_features
    
    @torch.no_grad() #features is [quesiton, answer, molecule_raw_2d_features,molecule_raw_3d_features], which is the output of MoLlamaDataset.__getitem__
    def __call__(self, features: List[Tuple[str, str, torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        molecule_raw_2d_features_list = []
        molecule_raw_3d_features_list = []
        
        for feature in features:
            mqa_obj = build_molecule_qa_input( 
                processor=self.processor,
                question=feature[0], #[1,L]
                answer=feature[1], #[1,L]
                molecule_raw_2d_features=feature[2], #[num_atoms,hidden_size]
                molecule_raw_3d_features=feature[3], #[num_atoms,hidden_size]
            )

            temp_input_ids, temp_labels, temp_molecule_raw_2d_features, temp_molecule_raw_3d_features = self.convert_one_piece(mqa_obj)
            #[L] [L] [num_atoms,hidden_size] [num_atoms,hidden_size]   
            #torch.Size([267]) torch.Size([267]) torch.Size([51, 300]) torch.Size([131, 512])
            #print(temp_input_ids.shape,temp_labels.shape,temp_molecule_raw_2d_features.shape,temp_molecule_raw_3d_features.shape)
            
            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            
            if temp_molecule_raw_2d_features is not None:
                molecule_raw_2d_features_list.append(temp_molecule_raw_2d_features)
            if temp_molecule_raw_3d_features is not None:
                molecule_raw_3d_features_list.append(temp_molecule_raw_3d_features)
        
        # Pad sequences with pad_token_id
        final_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
        final_labels = pad_sequence(labels_list, batch_first=True, padding_value=self.ignore_index)
        # Generate attention mask, this is just for tokens, not for molecule_features
        attention_mask=final_input_ids.ne(self.processor.tokenizer.pad_token_id).long()
            
        # Pad 2D and 3D molecule features with -100
        if len(molecule_raw_2d_features_list)==0:
            final_molecule_raw_2d_features=None
        else:
            final_molecule_raw_2d_features = pad_sequence(molecule_raw_2d_features_list, batch_first=True, padding_value=self.mol_padding)
        if len(molecule_raw_3d_features_list)==0:
            final_molecule_raw_3d_features=None
        else:
            final_molecule_raw_3d_features = pad_sequence(molecule_raw_3d_features_list, batch_first=True, padding_value=self.mol_padding)
    

        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "molecule_raw_2d_features": final_molecule_raw_2d_features,
            "molecule_raw_3d_features": final_molecule_raw_3d_features,
            "attention_mask": attention_mask
        }