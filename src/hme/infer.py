import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from tqdm import tqdm
from transformers import GenerationConfig, HfArgumentParser

from hme.data import HMEProcessor, TrainHMECollator
from hme.run_clm import prepare_dataset  # Renamed from load_dataset to avoid conflict
from hme.util import load_hme, set_seed

# --- Setup ---
set_seed(42)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to use for inference."""

    model_name_or_path: Optional[str] = field(default="test_model/model001")
    do_sample: bool = field(
        default=False, metadata={"help": "Enable sampling; otherwise greedy decoding."}
    )
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature."})
    top_k: int = field(default=50, metadata={"help": "Top-k filtering parameter."})
    top_p: float = field(
        default=0.95, metadata={"help": "Top-p (nucleus) filtering parameter."}
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search (1 = no beam search)."},
    )


@dataclass
class DataArguments:
    """Arguments pertaining to the data used for inference."""

    data_path: str = field(metadata={"help": "Path to the inference data file."})
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
        metadata={"help": "The type of task (e.g., 'qa', 'caption')."}
    )
    output_path: str = field(
        default="output.jsonl", metadata={"help": "Path to save the output JSONL file."}
    )
    max_length: Optional[int] = field(
        default=512, metadata={"help": "Maximum sequence length."}
    )


def _parse_generated_output(text: str, model_name: str) -> str:
    """Parses the raw generated text to extract the clean response."""
    if "llama2" in model_name.lower() or "llama-2" in model_name.lower():
        return text.split("[/INST]")[-1].strip().split("</s>")[0]
    else:  # Assumes Llama-3 or similar chat format
        return (
            text.split("assistant")[-1].split("\n\n")[-1].strip().split("<|eot_id|>")[0]
        )


def run():
    """
    Main function to run the inference pipeline.

    Steps:
    1. Parses command-line arguments.
    2. Loads the pre-trained/merged HME model and tokenizer.
    3. Loads and prepares the inference dataset.
    4. Sets up the generation configuration.
    5. Iterates through the dataset, generates responses for each sample.
    6. Parses the generated output and saves it along with the ground truth
       to a JSONL file in an append-only fashion.
    """
    # 1. Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # 2. Load model and tokenizer
    # Note: `add_frg_vocab` is False because we assume the model is already trained/merged.
    model, tokenizer, config, _ = load_hme(model_args, data_args, add_frg_vocab=False)
    model.to("cuda")
    model = torch.compile(model=model)

    # 3. Load dataset
    processor = HMEProcessor(tokenizer=tokenizer, max_length=data_args.max_length)
    data_collator = TrainHMECollator(processor=processor, config=config)
    test_dataset = prepare_dataset(
        data_args, is_eval=False
    )  # Use `is_eval=False` to load from `data_path`

    # 4. Set up generation config
    generation_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=model_args.do_sample,
        top_k=model_args.top_k,
        top_p=model_args.top_p,
        temperature=model_args.temperature,
        num_beams=model_args.num_beams,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # 5. Run inference loop
    if os.path.exists(data_args.output_path):
        logging.warning(
            f"Output file {data_args.output_path} already exists. Appending to it."
        )

    for i, item in enumerate(tqdm(test_dataset, desc="Generating responses")):
        # Prepare a single data point for the model
        data = data_collator([item])
        # Isolate the prompt part (where labels are -100)
        prompt_mask = data["labels"] == -100
        data["input_ids"] = data["input_ids"][prompt_mask].unsqueeze(0)
        data["attention_mask"] = data["attention_mask"][prompt_mask].unsqueeze(0)

        # Remove labels and move data to GPU
        data.pop("labels")
        for k, v in data.items():
            if v is not None:
                data[k] = v.to("cuda")

        # Generate response
        with torch.no_grad():
            output_tokens = model.generate(**data, generation_config=generation_config)

        # Decode and parse the output
        response_tokens = output_tokens[:, data["input_ids"].size(1) :]
        raw_gen_text = tokenizer.decode(response_tokens[0], skip_special_tokens=False)
        clean_gen_text = _parse_generated_output(
            raw_gen_text, model_args.model_name_or_path
        )
        ground_truth = item[1].strip()

        # Save the result
        with open(data_args.output_path, "a", encoding="utf-8") as f:
            json.dump(
                {"gen": clean_gen_text, "gt": ground_truth}, f, ensure_ascii=False
            )
            f.write("\n")

    logger.info(f"Inference complete. Results saved to {data_args.output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run()
