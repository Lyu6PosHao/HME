import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from hme.data import HMEDataset, HMEProcessor, TrainHMECollator
from hme.util import load_hme, print_trainable_parameters, set_seed

# --- Setup ---
set_seed(42)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""

    model_name_or_path: Optional[str] = field(
        default="test_model/model001",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    lora_r: Optional[int] = field(
        default=-1, metadata={"help": "LoRA attention dimension (rank)."}
    )
    lora_alpha: Optional[int] = field(
        default=None, metadata={"help": "The alpha parameter for LoRA scaling."}
    )
    lora_targets: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of module names to apply LoRA to."},
    )
    modules_to_save: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated list of module names to keep trainable without applying LoRA."
        },
    )
    merge_when_finished: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to merge the LoRA adapter into the base model upon completion."
        },
    )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    data_path: str = field(metadata={"help": "Path to the training data file."})
    data_type: str = field(
        metadata={
            "help": "Comma-separated string of data modalities (e.g., '1d,2d,3d')."
        }
    )
    emb_dict_mol: str = field(
        metadata={"help": "Path to the pre-computed molecule embeddings dictionary."}
    )
    emb_dict_protein: str = field(
        metadata={"help": "Path to the pre-computed protein embeddings dictionary."}
    )
    task_type: str = field(
        metadata={"help": "The type of task (e.g., 'qa', 'caption', 'pdbbind_reg')."}
    )
    val_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the validation data file."}
    )
    max_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization."
        },
    )


def load_model_processor_for_training(
    model_args: ModelArguments, data_args: DataArguments
) -> Tuple[torch.nn.Module, HMEProcessor, "HMEConfig"]:
    """
    Loads the model, tokenizer, and processor, and applies PEFT (LoRA) configuration.

    This function initializes the HME model using the `load_hme` utility, wraps it with
    a PEFT model for LoRA fine-tuning, and sets up the HMEProcessor.

    Returns
    -------
    Tuple[torch.nn.Module, HMEProcessor, "HMEConfig"]
        A tuple containing the PEFT--enhanced model, the data processor, and the model config.
    """
    model, tokenizer, config, _ = load_hme(model_args, data_args, add_frg_vocab=True)
    processor = HMEProcessor(tokenizer=tokenizer, max_length=data_args.max_length)
    logging.info(f"Tokenizer length: {len(tokenizer)}")

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=model_args.lora_targets.split(","),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=(
            model_args.modules_to_save.split(",")
            if model_args.modules_to_save is not None
            else None
        ),
    )
    model = get_peft_model(model, lora_config)

    if model_args.lora_r == 1:
        logging.info("lora_r is 1, freezing all LoRA parameters.")
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = False

    print_trainable_parameters(model)
    return model, processor, config


def prepare_dataset(data_args: DataArguments, is_eval: bool = False) -> HMEDataset:
    """Initializes and returns the custom HMEDataset for training or evaluation."""
    path = data_args.val_data_path if is_eval else data_args.data_path
    if path is None:
        raise ValueError(
            f"Data path for {'validation' if is_eval else 'training'} is not specified."
        )

    dataset = HMEDataset(
        data_path=path,
        task_type=data_args.task_type,
        data_type=data_args.data_type,
        emb_dict_mol=data_args.emb_dict_mol,
        emb_dict_protein=data_args.emb_dict_protein,
        val=is_eval,
    )

    # Log dataset info only on the main process
    if os.environ.get("LOCAL_RANK") in [None, "0"]:
        sample = dataset[0]
        logging.info(
            f"First {'eval' if is_eval else 'train'} sample: "
            f"Q: '{sample[0][:80]}...'; A: '{str(sample[1])[:80]}...'; "
            f"2D: {sample[2] is not None}; 3D: {sample[3] is not None}; Protein: {sample[4] is not None}"
        )
        logging.info(
            f"Dataset length ({'eval' if is_eval else 'train'}): {len(dataset)}"
        )

    return dataset


def train():
    """
    The main training function.

    It performs the following steps:
    1. Parses command-line arguments for model, data, and training settings.
    2. Loads the model, tokenizer, and processor, applying LoRA configuration.
    3. Loads the training and optional evaluation datasets.
    4. Initializes the `Seq2SeqTrainer`.
    5. Starts the training process.
    6. Saves the final model, tokenizer, and trainer state.
    7. Optionally, merges the LoRA adapter into the base model and saves the merged version.
    """
    # 1. Parse arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, Seq2SeqTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Load model and processor
    model, processor, config = load_model_processor_for_training(model_args, data_args)

    # 3. Load datasets
    train_dataset = prepare_dataset(data_args, is_eval=False)
    eval_dataset = (
        prepare_dataset(data_args, is_eval=True) if training_args.do_eval else None
    )

    # 4. Initialize Trainer
    data_collator = TrainHMECollator(processor=processor, config=config)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # 5. Start training
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # 6. Save final model and state
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    processor.tokenizer.save_pretrained(training_args.output_dir)

    # 7. Optionally, merge and save the final model
    if model_args.merge_when_finished and (os.environ.get("LOCAL_RANK") in [None, "0"]):
        output_dir = training_args.output_dir.rstrip("/")
        merged_model_path = f"{output_dir}_merged"
        os.makedirs(merged_model_path, exist_ok=True)

        logging.info(f"Merging LoRA adapter and saving to {merged_model_path}...")
        merged_model = model.merge_and_unload()

        torch.save(
            merged_model.feature_fuser.state_dict(),
            os.path.join(merged_model_path, "feature_fuser.pth"),
        )
        merged_model.language_model.save_pretrained(merged_model_path)
        processor.tokenizer.save_pretrained(merged_model_path)

        logging.info(f"Saved the merged model to {merged_model_path}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()
