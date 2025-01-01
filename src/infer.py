from util import set_seed
set_seed(42)
import logging
import os
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from transformers import HfArgumentParser
from tqdm import tqdm
from transformers import (
    GenerationConfig
)
from frg import get_frg_from_one_smiles
from data import  TrainMoLlamaCollator,MoLlamaProcessor
from util import load_mollama
from run import load_dataset
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="test_model/model001")

    max_length: Optional[int] = field(
        default=-1,
        metadata={"help": "Truncate the input/output sequence to this length if specified."},
    )
        
    


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data. Use ^ to connect multiple files."}
    )
    data_type: str = field(
        default=None, metadata={"help": """Use ',' to concat different data_types.
                                Example: 1d,2d,frg
                                """}
    )
    task_type: Optional[str] = field(
        default=None, metadata={"help": """Subset of the training data. This is passed to the MoLlamaDataset class.
                                1. qa
                                2. caption
                                3. pretrain
                                4. text2mol
                                etc. See data.py for more details.
                                """}
    )
        
    val_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the validation data. Use ^ to connect multiple files."}
    )

    output_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the save the generated results."}
    )
    noise: Optional[float] = field(
        default=None, metadata={"help": "The std of noise to the input."}
    )
    add_noise_to:Optional[str] = field(
        default=None, metadata={"help": "The type of data to add noise to."}
    )

def add_noise(raw_data,noise:float):
    if noise==0:
        return raw_data
    # add gaussian noise to raw_data, raw_data is a tensor
    noise_data=torch.normal(mean=0,std=noise,size=raw_data.size()).to(raw_data)
    return raw_data+noise_data

def replace_random_chars(input_str, num_replacements=5):
    import random
    import string
    str_list = list(input_str)
    
    str_len = len(str_list)
    
    # replace num_replacements characters
    for _ in range(num_replacements):
        random_index = random.randint(0, str_len - 1)
        
        random_char = random.choice(['N','C','O','=','c',')','('])

        str_list[random_index] = random_char

    return ''.join(str_list)

def run():
    #load the arguments
    parser = HfArgumentParser(
        (ModelArguments, DataArguments)
    )
    model_args, data_args = parser.parse_args_into_dataclasses()
    
    #load the merged model and tokenizer
    model,tokenizer=load_mollama(model_args)
    model.to("cuda")
    model=torch.compile(model=model)
    
    #load the dataset
    processor = MoLlamaProcessor(tokenizer=tokenizer,max_length=model_args.max_length)
    data_collator = TrainMoLlamaCollator(processor=processor, IGNORE_INDEX=-100)
    test_dataset = load_dataset(data_args)
    
    #generation config
    generation_config=GenerationConfig(do_sample=False)#,max_new_tokens=512,top_p=0.95,top_k=50)
    #-------------------------

    #-------------------------
    #inference
    i=0
    inner_loop=1
    print(f'Pay attention that the inner_loop maybe not 1. It is {inner_loop}!!!')
    from tqdm import tqdm
    bar=tqdm(total=len(test_dataset))
    while i < len(test_dataset):
        #get the data
        item=test_dataset[i]
        
        #add some noise
        # if data_args.add_noise_to=='1d':
        #     temp=item[0].split(' ')
        #     if temp[-1].endswith('>'):
        #         smiles=temp[-2]
        #         smiles=replace_random_chars(smiles,num_replacements=len(smiles)//20)
        #         temp[-2]=smiles
        #     else:
        #         smiles=temp[-1]
        #         smiles=replace_random_chars(smiles,
        #                                     num_replacements=min(
        #                                         len(smiles)//20,1
        #                                     )
        #         )
        #         temp[-1]=smiles
        #     item[0]=' '.join(temp)
        # if data_args.add_noise_to=='2d':
        #     item[2]=add_noise(item[2],data_args.noise)
        # elif data_args.add_noise_to=='3d':
        #     item[3]=add_noise(item[3],data_args.noise)
            
            
        if i==0:
            print('!!!!',item[0])

        data=data_collator([item])
        if i==0:
            print(tokenizer.tokenize(item[0]+item[1]))
            print('-'*100)
            print(tokenizer.decode(data['input_ids'][0]))
            print('-'*100)
        # the part of data where labels is -100 is the part serving as the input
        data["input_ids"]=data["input_ids"][data["labels"]==-100].unsqueeze(dim=0)
        data["attention_mask"]=data["attention_mask"][data["labels"]==-100].unsqueeze(dim=0)
        

        data.pop("labels")
        for k,v in data.items():
            if v is not None:
                data[k]=v.to("cuda")
        
        if i==0:
            print(tokenizer.decode(data['input_ids'][0]))
            print('-'*100)
            
        #the inner loop: generate multiple times for each sample
        for _ in range(inner_loop):
            #generate    
            with torch.no_grad():
                output=model.generate(
                    **data,
                    generation_config=generation_config,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            #process the output
            if 'llama2' in model_args.model_name_or_path or 'llama-2' in model_args.model_name_or_path:
                gen=tokenizer.decode(output[0],skip_special_tokens=True).split("[/INST]")[-1].strip()
            else:
                gen=tokenizer.decode(output[0],skip_special_tokens=True).split("assistant")[-1].strip()
            gt=item[1].strip()
            gen=gen.replace('\n',' ')
            #save the output
            with open(data_args.output_path,'a') as f:
                f.write(f'{gen}<iamsplit>{gt}\n')
        
        #print the output
        if i==0:
            print(tokenizer.decode(output[0],skip_special_tokens=False))
            print('-'*100)
            print(gen)
            print('-'*100)
        #update the bar
        bar.update(1)    
        i=i+1
        if i/float(len(test_dataset))*100%10==0:
            print(f'processed {i} samples; {i/len(test_dataset)}')
    bar.close()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run()