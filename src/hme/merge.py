import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftModel

from util import load_hme, set_seed

# --- Initial Setup ---
set_seed(42)


@dataclass
class ModelArguments:
    """Arguments for specifying model paths for merging."""
    model_name_or_path: str = field(
        metadata={"help": "Path to the base language model (e.g., Llama-3)."}
    )
    peft_model_path: str = field(
        metadata={"help": "Path to the trained PEFT (LoRA) adapter checkpoint."}
    )
    merged_model_path: str = field(
        metadata={"help": "Path where the merged model will be saved."}
    )


@dataclass
class DataArguments:
    """Arguments for specifying the task type to ensure correct model head saving."""
    task_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Task type (e.g., 'pdbbind_reg', 'moleculenet_cls'). "
            "Determines which task-specific head to save."
        },
    )


def merge_lora_model(model_args: ModelArguments, data_args: DataArguments) -> None:
    """
    (Low-level) Loads a base model and a LoRA adapter, merges them, and saves the result.
    """
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

    # Save Core Components
    torch.save(
        model.feature_fuser.state_dict(),
        os.path.join(model_args.merged_model_path, "feature_fuser.pth"),
    )
    model.language_model.save_pretrained(model_args.merged_model_path)
    tokenizer.save_pretrained(model_args.merged_model_path)

    # Save Task-Specific Head
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

    print(f"--- Model merged and saved successfully to {model_args.merged_model_path} ---")


def run_merge_task(
    base_model_path: str, peft_model_path: str, task_type: Optional[str] = None
) -> None:
    """
    High-level wrapper to perform a single model merge task.

    This function simplifies the merge process by automatically deriving the output
    path and handling the instantiation of argument classes.

    Parameters
    ----------
    base_model_path : str
        Path to the base language model.
    peft_model_path : str
        Path to the trained LoRA adapter checkpoint.
    task_type : Optional[str], optional
        The task type, used to save the correct model head, by default None.
    """
    print(f"\n{'='*20} Starting Merge Task {'='*20}")
    print(f"Base Model: {base_model_path}")
    print(f"Adapter: {peft_model_path}")
    print(f"Task Type: {task_type}")

    # Automatically determine the output path by appending "_merged"
    merged_model_path = peft_model_path.rstrip("/") + "_merged"

    model_args = ModelArguments(
        model_name_or_path=base_model_path,
        peft_model_path=peft_model_path,
        merged_model_path=merged_model_path,
    )
    data_args = DataArguments(task_type=task_type)

    merge_lora_model(model_args=model_args, data_args=data_args)
    print(f"{'='*20} Finished Merge Task {'='*21}\n")


if __name__ == "__main__":
    # Set GPU device only when running the script directly
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # --- Configuration for Merging ---
    # Define the base model path once.
    BASE_MODEL_PATH = '/path/to/your/base-llama-3-8b-instruct'

    # Define all merge tasks in a list. To add a new merge, just add a dictionary here.
    tasks_to_run = [
        {
            "peft_model_path": "/path/to/your/regression-lora-checkpoint",
            "task_type": "pdbbind_reg"
        },
        {
            "peft_model_path": "/path/to/your/generation-lora-checkpoint",
            "task_type": "qa"  # Or None for conditional generation
        },
        # Add more tasks here as needed
        # {
        #     "peft_model_path": "/path/to/another/lora-checkpoint",
        #     "task_type": "moleculenet_cls"
        # },
    ]

    # --- Run All Merge Tasks ---
    for task in tasks_to_run:
        run_merge_task(
            base_model_path=BASE_MODEL_PATH,
            peft_model_path=task["peft_model_path"],
            task_type=task.get("task_type") # .get() handles cases where task_type is omitted
        )