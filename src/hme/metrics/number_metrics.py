"""
This script evaluates the performance of a model on numerical prediction tasks.
It parses prediction files, extracts generated and ground truth numbers, filters outliers,
calculates metrics like MAE, R², Pearson, and RMSE, and provides an optional
visualization function.
"""
import json
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- Task Definitions ---
# A set of supported task types for property prediction.
SUPPORTED_TASKS = {
    "weight",
    "logp",
    "complexity",
    "topological polar surface area",
    "homo-lumo gap",
    "homo",
    "lumo",
    "scf energy",
    "docking",
}

# A dictionary defining the valid numerical ranges for different tasks to filter outliers.
OUTLIER_BOUNDS = {
    "homo-lumo gap": (-20, 20),
    "homo": (-20, 20),
    "lumo": (-20, 20),
    "scf energy": (-5, 0),
    "logp": (-30, 50),
    "topological polar surface area": (0, 2000),
    "complexity": (0, 10000),
    "weight": (0, 4000),
    "docking": None,  # No outlier check for docking
}


def must_find_task_type(str1: str, str2: str) -> Tuple[str, bool]:
    """
    Identifies the task type from input strings by searching for keywords.

    Parameters
    ----------
    str1 : str
        The first string to search (e.g., generated text).
    str2 : str
        The second string to search (e.g., ground truth text).

    Returns
    -------
    Tuple[str, bool]
        A tuple containing the found task type and a boolean flag (always True if found).

    Raises
    ------
    ValueError
        If no known task type keyword is found in either string.
    """
    for task in SUPPORTED_TASKS:
        if (task in str1.lower()) or (task in str2.lower()):
            return task, True
    raise ValueError(
        f"Task type not found in strings. Checked for: {', '.join(SUPPORTED_TASKS)}"
    )


def visualize_performance(
    gen: List[Optional[float]], gt: List[Optional[float]], task_name: str = "logp"
) -> None:
    """
    Visualizes the performance with a scatter plot of generated vs. ground truth values.

    Parameters
    ----------
    gen : List[Optional[float]]
        List of generated numerical values. Nones will be filtered out.
    gt : List[Optional[float]]
        List of ground truth numerical values. Nones will be filtered out.
    task_name : str, optional
        Name of the task, used for the plot title, by default 'logp'.
    """
    valid_gen = [item for item in gen if item is not None]
    valid_gt = [item for item in gt if item is not None]

    if len(valid_gen) != len(valid_gt):
        raise ValueError(
            "The number of valid generated and ground truth values must be the same."
        )

    plt.figure(figsize=(10, 6))
    plt.scatter(valid_gt, valid_gen, alpha=0.5)
    plt.title(f"Scatter plot of Generated vs Ground Truth for {task_name}")
    plt.xlabel("Ground Truth")
    plt.ylabel("Generated")
    plt.grid(True)
    plt.show()


def validate_numbers(
    gen_num_list: List[str],
    gt_num_list: List[str],
    task_type: str,
    exclude_outliers: bool = True,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Validates and cleans a pair of generated and ground truth numbers.

    It converts strings to floats and optionally filters outliers based on task-specific ranges.

    Parameters
    ----------
    gen_num_list : List[str]
        A list of numbers found in the generated text.
    gt_num_list : List[str]
        A list of numbers found in the ground truth text.
    task_type : str
        The task type, used to determine outlier bounds.
    exclude_outliers : bool, optional
        If True, applies outlier filtering, by default True.

    Returns
    -------
    Tuple[Optional[float], Optional[float]]
        A tuple of cleaned (generated_num, ground_truth_num), or (None, None) if validation fails.
    """
    if not (gen_num_list and gt_num_list):
        return None, None
    try:
        gen_num = float(gen_num_list[0])
        gt_num = float(gt_num_list[0])
    except (ValueError, IndexError):
        return None, None

    if exclude_outliers and task_type in OUTLIER_BOUNDS:
        bounds = OUTLIER_BOUNDS[task_type]
        if bounds is not None:
            lower, upper = bounds
            if not (lower <= gen_num < upper):
                return None, None
    return gen_num, gt_num


def get_out_gt_from_line(file_path: str, line: str) -> Tuple[str, str]:
    """
    Parses a single line from a prediction file to extract generated and ground truth text.
    """
    if file_path.endswith(".txt"):
        parts = line.strip().split("<iamsplit>")
        if len(parts) != 2:
            raise ValueError(f"Invalid format in .txt file line: {line}")
        return parts[0].strip(), parts[1].strip()
    else:
        data = json.loads(line)
        return data["gen"], data["gt"]


def _get_task_groups(input_file: str) -> Dict[str, List]:
    """Determines the task groups based on the input file name."""
    if "pubchemqc" in input_file or "pubchemqa" in input_file or "property-qa-2" in input_file:
        return {"homo": [], "lumo": [], "homo-lumo gap": [], "scf energy": []}
    elif "docking" in input_file:
        return {"docking": []}
    else:
        return {"weight": [], "logp": [], "topological polar surface area": [], "complexity": []}


def _calculate_statistics(
    gen_values: List[float], gt_values: List[float], task_name: str
) -> Tuple[float, float, float, float, float]:
    """Calculates MAE, R², Pearson, equal ratio, and RMSE."""
    if not gen_values:  # Avoid division by zero if no valid data
        return np.nan, np.nan, np.nan, np.nan, np.nan

    gt_arr = np.array(gt_values)
    gen_arr = np.array(gen_values)

    mae = np.mean(np.abs(gt_arr - gen_arr))
    ss_res = np.sum((gt_arr - gen_arr) ** 2)
    ss_tot = np.sum((gt_arr - np.mean(gt_arr)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    if len(gt_arr) > 1:
        pearson = np.corrcoef(gen_arr, gt_arr)[0, 1]
    else:
        pearson = np.nan # Cannot compute with a single point

    equal_ratio = np.mean(np.isclose(gt_arr, gen_arr))
    rmse = np.sqrt(np.mean((gt_arr - gen_arr) ** 2))

    print(f"{task_name}: RMSE value is {rmse:.4f}")
    return mae, r2, pearson, equal_ratio, rmse


def eval_number(input_file: str, exclude_outliers: bool = True) -> List[float]:
    """
    Main function to evaluate numerical predictions from a file.

    Parameters
    ----------
    input_file : str
        Path to the prediction file (.txt or .jsonl).
    exclude_outliers : bool, optional
        Whether to filter out-of-range values, by default True.

    Returns
    -------
    List[float]
        A list containing MAE and R² scores for each task group.
    """
    with open(input_file, "r") as f:
        lines = f.readlines()

    gen = _get_task_groups(input_file)
    gt = {k: [] for k in gen.keys()}
    pattern = r'-?\d+\.\d+'  # Matches integers and floats

    for line in tqdm(lines, desc=f"Processing {input_file}"):
        item_gen, item_gt = get_out_gt_from_line(input_file, line)
        task_type, _ = must_find_task_type(item_gen, item_gt)

        gen_num_list = re.findall(pattern, item_gen)
        gt_num_list = re.findall(pattern, item_gt)

        gen_num, gt_num = validate_numbers(
            gen_num_list, gt_num_list, task_type, exclude_outliers=exclude_outliers
        )

        gen[task_type].append(gen_num)
        gt[task_type].append(gt_num)

    for k in gen.keys():
        total = len(gen[k])
        valid_count = total - gen[k].count(None)
        valid_ratio = valid_count / total if total > 0 else 0
        print(f"{k}: valid ratio is {valid_ratio:.4f}; total is {total}")

    all_maes = []
    all_r2s = []
    for k in gen.keys():
        if k.lower() == "scf energy":
            print("Note: 'scf energy' values are multiplied by 10 for evaluation.")
            gen_temp = [10 * item for item in gen[k] if item is not None]
            gt_temp = [10 * item for item in gt[k] if item is not None]
        else:
            gen_temp = [item for item in gen[k] if item is not None]
            gt_temp = [item for item in gt[k] if item is not None]

        mae, r2, _, _, _ = _calculate_statistics(gen_temp, gt_temp, k)
        all_maes.append(mae)
        all_r2s.append(r2)

    return all_maes + all_r2s


if __name__ == "__main__":
    # --- Example Usage ---
    # Please replace the file paths with your actual result file locations.
    file_path = "result.jsonl"

    print(f"--- Evaluating {file_path} ---")
    results = eval_number(file_path, exclude_outliers=True)
    formatted = [f"{item:.4f}" for item in results]
    print("&".join(formatted[:4]), "\n", "&".join(formatted[4:]))
