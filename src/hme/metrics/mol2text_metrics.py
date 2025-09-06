"""
This script provides functions to evaluate text generation models using standard NLP metrics:
BLEU, METEOR, and ROUGE. It can process prediction files in either .txt or .jsonl format.
"""

import json
from typing import Tuple, List

import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import BertTokenizerFast

# Note: The 'wordnet' corpus is required for METEOR score calculation.
# If not already downloaded, uncomment and run the following line once:
# import nltk
# nltk.download('wordnet')


def get_out_gt_from_line(file_path: str, line: str) -> Tuple[str, str]:
    """
    Parses a single line from the prediction file to extract the generated text and ground truth.

    Supports two formats:
    - .txt: "generated_text<iamsplit>ground_truth_text"
    - .jsonl: {"gen": "generated_text", "gt": "ground_truth_text"}

    Parameters
    ----------
    file_path : str
        The path to the input file, used to determine the format by its extension.
    line : str
        A single line from the file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the (generated_text, ground_truth_text).
    """
    if file_path.endswith(".txt"):
        parts = line.strip().split("<iamsplit>")
        if len(parts) != 2:
            raise ValueError(f"Invalid format in .txt file line: {line}")
        out, gt = parts[0].strip(), parts[1].strip()
    else:
        data = json.loads(line)
        out, gt = data["gen"].strip(), data["gt"].strip()
    return out, gt


def eval_mol2text(
    input_file: str,
    text_model: str = "allenai/scibert_scivocab_uncased",
    text_trunc_length: int = 512,
) -> Tuple[float, float, float, float, float, float]:
    """
    Evaluates generated text against ground truth using BLEU, ROUGE, and METEOR scores.

    Parameters
    ----------
    input_file : str
        Path to the prediction file.
    text_model : str, optional
        The Hugging Face model identifier for the tokenizer, by default "allenai/scibert_scivocab_uncased".
        Note: The original code used a local path; a public model is used here for portability.
    text_trunc_length : int, optional
        The maximum length for tokenization, by default 512.

    Returns
    -------
    Tuple[float, float, float, float, float, float]
        A tuple containing the scores: (BLEU-2, BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L, METEOR).
    """
    with open(input_file, "r") as f:
        lines = f.readlines()

    text_tokenizer = BertTokenizerFast.from_pretrained(
        text_model, clean_up_tokenization_spaces=True
    )
    rouge_evaluator = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])

    special_tokens_to_filter = {"[PAD]", "[CLS]", "[SEP]"}
    references_for_bleu = []
    hypotheses_for_bleu = []
    meteor_scores = []
    rouge_scores = []

    for line in tqdm(lines, desc="Evaluating"):
        out_text, gt_text = get_out_gt_from_line(input_file, line)

        # --- Prepare for BLEU and METEOR ---
        gt_tokens = text_tokenizer.tokenize(
            gt_text, truncation=True, max_length=text_trunc_length
        )
        gt_tokens_filtered = [
            token for token in gt_tokens if token not in special_tokens_to_filter
        ]

        out_tokens = text_tokenizer.tokenize(
            out_text, truncation=True, max_length=text_trunc_length
        )
        out_tokens_filtered = [
            token for token in out_tokens if token not in special_tokens_to_filter
        ]

        references_for_bleu.append([gt_tokens_filtered])
        hypotheses_for_bleu.append(out_tokens_filtered)
        meteor_scores.append(meteor_score([gt_tokens_filtered], out_tokens_filtered))

        # --- Calculate ROUGE ---
        rs = rouge_evaluator.score(out_text, gt_text)
        rouge_scores.append(rs)

    # --- Finalize Scores ---
    bleu2 = corpus_bleu(references_for_bleu, hypotheses_for_bleu, weights=(0.5, 0.5))
    bleu4 = corpus_bleu(
        references_for_bleu, hypotheses_for_bleu, weights=(0.25, 0.25, 0.25, 0.25)
    )
    avg_meteor_score = np.mean(meteor_scores)

    rouge_1 = np.mean([rs["rouge1"].fmeasure for rs in rouge_scores])
    rouge_2 = np.mean([rs["rouge2"].fmeasure for rs in rouge_scores])
    rouge_l = np.mean([rs["rougeL"].fmeasure for rs in rouge_scores])

    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, avg_meteor_score


if __name__ == "__main__":
    # Example usage:
    # Ensure the input file path is correct.
    try:
        results = eval_mol2text(input_file="result.jsonl")
        # Format results for easy copying (e.g., into a LaTeX table)
        formatted_results = [f"{item * 100:.2f}" for item in results]
        print("&".join(formatted_results))
    except FileNotFoundError:
        print("Error: The specified input file was not found.")
        print("Please update the path in the `if __name__ == '__main__'` block.")
