import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import transformers
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)
from transformers import Trainer, TrainingArguments

from hme.data import (HMEDataset, HMEProcessor,
                  TrainHMECollatorClassification)
from hme.util import load_hme, print_trainable_parameters, set_seed

# --- Setup ---
set_seed(42)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""
    model_name_or_path: Optional[str] = field(default="test_model/model001")
    lora_r: Optional[int] = field(default=None, metadata={"help": "LoRA attention dimension."})
    lora_alpha: Optional[int] = field(default=None, metadata={"help": "LoRA scaling factor."})
    lora_targets: Optional[str] = field(default=None, metadata={"help": "Modules to apply LoRA to."})
    modules_to_save: Optional[str] = field(default=None, metadata={"help": "Modules to keep trainable."})
    merge_when_finished: Optional[bool] = field(default=True, metadata={"help": "Merge LoRA on completion."})


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    data_path: str = field(metadata={"help": "Path to the training data file."})
    data_type: str = field(metadata={"help": "Comma-separated string of data modalities."})
    emb_dict_mol: str = field(metadata={"help": "Path to the molecule embeddings dictionary."})
    emb_dict_protein: str = field(metadata={"help": "Path to the protein embeddings dictionary."})
    task_type: str = field(metadata={"help": "The type of task (e.g., 'moleculenet_cls')."})
    val_data_path: Optional[str] = field(default=None, metadata={"help": "Path to the validation data file."})
    max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})


def load_model_processor_for_training(
    model_args: ModelArguments, data_args: DataArguments
) -> tuple[torch.nn.Module, HMEProcessor, "HMEConfig"]:
    """Loads the model, tokenizer, and processor, and applies PEFT (LoRA) configuration."""
    model, tokenizer, config, _ = load_hme(
        model_args, data_args, add_frg_vocab=True
    )
    processor = HMEProcessor(tokenizer=tokenizer, max_length=data_args.max_length)
    
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=model_args.lora_targets.split(','),
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        modules_to_save=(
            model_args.modules_to_save.split(',')
            if model_args.modules_to_save is not None
            else None
        ),
    )
    model = get_peft_model(model, lora_config)
    
    print_trainable_parameters(model)
    return model, processor, config


def prepare_dataset(data_args: DataArguments, is_eval: bool = False) -> HMEDataset:
    """Initializes and returns the custom HMEDataset for training or evaluation."""
    path = data_args.val_data_path if is_eval else data_args.data_path
    if path is None:
        raise ValueError(f"Data path for {'validation' if is_eval else 'training'} is not specified.")
        
    dataset = HMEDataset(
        data_path=path,
        task_type=data_args.task_type,
        data_type=data_args.data_type,
        emb_dict_mol=data_args.emb_dict_mol,
        emb_dict_protein=data_args.emb_dict_protein,
        val=is_eval,
    )
    
    if os.environ.get("LOCAL_RANK") in [None, "0"]:
        logger.info(f"Loaded {'eval' if is_eval else 'train'} dataset with {len(dataset)} samples.")
    return dataset


def compute_metrics(eval_preds: tuple) -> Dict[str, float]:
    """
    Compute evaluation metrics for a binary classification task.

    Calculates accuracy, F1 score, precision, recall, and AUC.
    
    Parameters
    ----------
    eval_preds : tuple
        A tuple (logits, labels). Logits are expected to be of shape (batch_size, 1).
        
    Returns
    -------
    Dict[str, float]
        A dictionary containing the computed metrics.
    """
    logits, labels = eval_preds
    
    # Squeeze logits and get probabilities via sigmoid
    positive_class_scores = 1 / (1 + np.exp(-np.squeeze(logits)))
    predictions = (positive_class_scores >= 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', pos_label=1, zero_division=0
    )
    
    try:
        auc = roc_auc_score(labels, positive_class_scores)
    except ValueError:  # Handles cases where only one class is present in a batch
        auc = float('nan')

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
    }


def train():
    """Main training function for classification tasks."""
    # 1. Parse arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Load model and processor
    model, processor, config = load_model_processor_for_training(model_args, data_args)
    
    # 3. Load datasets
    train_dataset = prepare_dataset(data_args, is_eval=False)
    eval_dataset = prepare_dataset(data_args, is_eval=True) if training_args.do_eval else None
    
    # 4. Initialize Trainer
    data_collator = TrainHMECollatorClassification(processor=processor, config=config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
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
        
        logger.info(f"Merging LoRA adapter and saving to {merged_model_path}...")
        merged_model = model.merge_and_unload()
        
        torch.save(
            merged_model.feature_fuser.state_dict(),
            os.path.join(merged_model_path, "feature_fuser.pth"),
        )
        torch.save(
            merged_model.classification_head.state_dict(),
            os.path.join(merged_model_path, "classification_head.pth"),
        )
        merged_model.language_model.save_pretrained(merged_model_path)
        processor.tokenizer.save_pretrained(merged_model_path)
        
        logger.info(f"Saved the merged model to {merged_model_path}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()