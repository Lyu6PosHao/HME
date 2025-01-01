import numpy as np
import selfies
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from nltk.translate.bleu_score import corpus_bleu
from Levenshtein import distance as lev
import json
from tqdm import tqdm
import os
import re
from transformers import AutoTokenizer
import pickle
from fcd_torch import FCD

np.random.seed(42)
RDLogger.DisableLog("rdApp.*")
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def frg_replace(file_path,gen_file_path):
    with open(file_path, 'rb') as f:
        json_list=pickle.load(f)
    with open(gen_file_path, 'r') as f:
        lines=f.readlines()
    assert len(json_list)==len(lines)
    for i,item in enumerate(tqdm(json_list)):
        item['fragments']=lines[i].strip().split('<iamsplit>')[0]
    with open(file_path+'.gen.pkl', 'wb') as f:
        pickle.dump(json_list,f)
    print('Done')

def eval_frg_generation(file_path):
    with open(file_path, 'r') as f:
        lines=f.readlines()
    precision_list=[]
    recall_list=[]
    f1_list=[]
    equal_list=[]
    for item in tqdm(lines):
        if not item.startswith('<|'):
            continue
        temp = item.strip().split('<iamsplit>')
        assert len(temp)==2
        out = temp[0]
        gt = temp[1]
        # out=tokenizer.tokenize(out)
        # gt=tokenizer.tokenize(gt)

        out=re.findall(r'<\|.*?\|>',out)

        gt=re.findall(r'<\|.*?\|>',gt)

        #calculate the precision
        try:
            precision=len(set(out).intersection(set(gt)))/len(set(out))
        except:
            print(out,gt)
        #calculate the recall
        try:
            recall=len(set(out).intersection(set(gt)))/len(set(gt))
        except:
            print(out,gt)
        try:
            equal=len(set(out).intersection(set(gt)))/len(set(out).union(set(gt)))
        except:
            print(out,gt)
        #calculate the f1
        #f1=2*precision*recall/(precision+recall)
        equal_list.append(equal)
        precision_list.append(precision)
        recall_list.append(recall)
        #f1_list.append(f1)
    assert len(precision_list)==len(recall_list)==len(equal_list)
    print('len:',len(precision_list))
    print({
        "Precision": np.mean(precision_list),
        "Recall": np.mean(recall_list),
        'Equal':np.mean(equal_list)
        #"F1": np.mean(f1_list)
    }
    )

    
    
def eval_smi_generation(file_path):
    with open(file_path, 'r') as f:
        lines=f.readlines()
    N=len(lines)
    output_tokens, gt_tokens = [], []
    levs = []
    maccs_sim, rdk_sim, morgan_sim = [], [], []
    n_bad_mols, n_exact = 0, 0
    valid_outputs=[]
    valid_gts=[]
    gts=[]
    outputs=[]
    for item in tqdm(lines):
        temp = item.strip().split('<iamsplit>')
        assert len(temp)==2
        out = temp[0]
        gt = temp[1]
        #to extract the smiles
        out=out.split('Molecular SMILES is: ')[-1].strip()
        gt=gt.split('Molecular SMILES is: ')[-1].strip()
        
        gts.append(gt)
        outputs.append(out)
        try:
            out=Chem.CanonSmiles(out)
            gt=Chem.CanonSmiles(gt)
            valid_outputs.append(out)
            valid_gts.append(gt)
        except:
            n_bad_mols+=1
            continue
        output_tokens.append([c for c in out])
        gt_tokens.append([[c for c in gt]])
        try:
            mol_output = Chem.MolFromSmiles(out)
            mol_gt = Chem.MolFromSmiles(gt)
            if Chem.MolToInchi(mol_output) == Chem.MolToInchi(mol_gt):
                n_exact += 1
            maccs_sim.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(mol_output), MACCSkeys.GenMACCSKeys(mol_gt), metric=DataStructs.TanimotoSimilarity))
            rdk_sim.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(mol_output), Chem.RDKFingerprint(mol_gt), metric=DataStructs.TanimotoSimilarity))
            morgan_sim.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(mol_output, 2), AllChem.GetMorganFingerprint(mol_gt, 2)))
        except:
            n_bad_mols += 1
        levs.append(lev(out, gt))

    bleu = corpus_bleu(gt_tokens, output_tokens)
    print( {
        "BLEU": bleu,
        "Exact": n_exact * 1.0 / N,
        "Levenshtein": np.mean(levs),
        "MACCS FTS": np.mean(maccs_sim),
        "RDKit FTS": np.mean(rdk_sim),
        "Morgan FTS": np.mean(morgan_sim),
        "FCD": calc_fcd_torch(valid_outputs,gts),
        "Valid": 1 - n_bad_mols * 1.0 / N,
    })
    return {
        "BLEU": bleu,
        "Exact": n_exact * 1.0 / N,
        "Levenshtein": (levs),
        "MACCS FTS": (maccs_sim),
        "RDKit FTS": (rdk_sim),
        "Morgan FTS": (morgan_sim),
        "Valid": 1 - n_bad_mols * 1.0 / N,
    }

def calc_fcd_torch(outputs,gts):
    fcd=FCD()
    fcd_score=fcd(gts,outputs)
    print(len(outputs),len(gts))
    return fcd_score
    


if __name__ == "__main__":
    eval_smi_generation('path/to/HME/generated_results.txt')