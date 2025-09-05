# src/hme/merge.py
import os
import argparse
import sys
from pathlib import Path
from typing import Optional

from dataclasses import dataclass, field
import torch
from peft import PeftModel
from util import load_hme, set_seed


def merge_lora_model(model_args: "ModelArguments", data_args: "DataArguments") -> None:
    set_seed(42)
    print("--- Loading Base Model and Tokenizer ---")
    model, tokenizer, _, _ = load_hme(
        model_args, data_args=data_args, add_fr_vocab=True
    )
    print(f"--- Loading PEFT Adapter from {model_args.peft_model_path} ---")
    model = PeftModel.from_pretrained(model, model_args.peft_model_path)
    print("--- Merging Adapter into Base Model ---")
    model = model.merge_and_unload()
    print(f"--- Saving Merged Model to {model_args.merged_model_path} ---")
    os.makedirs(model_args.merged_model_path, exist_ok=True)
    torch.save(
        model.feature_fuser.state_dict(),
        os.path.join(model_args.merged_model_path, "feature_fuser.pth"),
    )
    model.language_model.save_pretrained(model_args.merged_model_path)
    tokenizer.save_pretrained(model_args.merged_model_path)
    if data_args.task_type and "_reg" in data_args.task_type:
        print("Saving regression head...")
        torch.save(
            model.regression_head.state_dict(),
            os.path.join(model_args.merged_model_path, "regression_head.pth"),
        )
    elif data_args.task_type and "_cls" in data_args.task_type:
        print("Saving classification head...")
        torch.save(
            model.classification_head.state_dict(),
            os.path.join(model_args.merged_model_path, "classification_head.pth"),
        )
    else:
        print("No task-specific head to save (task is likely conditional generation).")
    print(
        f"--- Model merged and saved successfully to {model_args.merged_model_path} ---"
    )


@dataclass
class ModelArguments:
    model_name_or_path: str
    peft_model_path: str
    merged_model_path: str


@dataclass
class DataArguments:
    task_type: Optional[str] = None


def main():
    parser = argparse.ArgumentParser(
        description="Merge a PEFT adapter into a base model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base_model_path",
        type=Path,
        required=True,
        help="Path to the base model to merge into.",
    )
    parser.add_argument(
        "--adapter_path",
        type=Path,
        required=True,
        help="Path to the PEFT adapter to merge.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        help="Optional task type (e.g., 'pdbbind_reg') to save a specific head.",
    )
    args = parser.parse_args()

    if not args.base_model_path.exists():
        print(
            f"Error: Base model path not found at '{args.base_model_path}'",
            file=sys.stderr,
        )
        sys.exit(1)
    if not args.adapter_path.exists():
        print(
            f"Error: Adapter path not found at '{args.adapter_path}'", file=sys.stderr
        )
        sys.exit(1)

    # Automatically determine the output path
    output_path = args.adapter_path.parent / (args.adapter_path.name + "_merged")

    model_args = ModelArguments(
        model_name_or_path=str(args.base_model_path),
        peft_model_path=str(args.adapter_path),
        merged_model_path=str(output_path),
    )
    data_args = DataArguments(task_type=args.task_type)

    merge_lora_model(model_args, data_args)


if __name__ == "__main__":
    main()
