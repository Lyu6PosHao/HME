
'''
Code from https://github.com/blender-nlp/MolT5

```bibtex
@article{edwards2022translation,
  title={Translation between Molecules and Natural Language},
  author={Edwards, Carl and Lai, Tuan and Ros, Kevin and Honke, Garrett and Ji, Heng},
  journal={arXiv preprint arXiv:2204.11817},
  year={2022}
}
```
'''

import argparse
import csv

import os.path as osp

import numpy as np

from transformers import BertTokenizerFast,AutoTokenizer
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
from rouge_score import rouge_scorer
import pdb
import json
from tqdm import tqdm

# nltk.download('wordnet')
def evaluate(text_model='path/to/scibert_scivocab_uncased', input_file=None, text_trunc_length=512):
    
    with open(input_file, 'r') as f:
        lines=f.readlines()
    text_tokenizer = BertTokenizerFast.from_pretrained(text_model,clean_up_tokenization_spaces=True)

    bleu_scores = []
    meteor_scores = []

    references = []
    hypotheses = []

    for i, item in enumerate(tqdm(lines)):
        temp = item.strip().split('<iamsplit>')
        if  not len(temp)==2:
            print(i, item)
            continue
        out = temp[0]
        gt = temp[1]


        gt_tokens = text_tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = text_tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))

    _meteor_score = np.mean(meteor_scores)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for i, item in enumerate((lines)):
        temp = item.strip().split("<iamsplit>")
        if not len(temp)==2:
            continue
        out=temp[0]
        gt = temp[1]
        rs = scorer.score(out, gt)
        rouge_scores.append(rs)


    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])
    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score

if __name__ == "__main__":
    results=evaluate(input_file='path/to/HME/generated_text.txt')
    results=[str(round(item*100,2)) for item in results]
    print('&'.join(results))
