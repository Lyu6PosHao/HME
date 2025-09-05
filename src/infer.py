from util import set_seed

set_seed(42)
import logging
import os
import torch, json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
from tqdm import tqdm
from transformers import HfArgumentParser
from tqdm import tqdm
from transformers import GenerationConfig

# from frg import get_frg_from_one_smiles
from data import TrainMoLlamaCollator, MoLlamaProcessor
from util import load_mollama
from run_clm import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="test_model/model001")

    do_sample: bool = field(
        default=False,
        metadata={"help": "Whether to use sampling; use greedy decoding otherwise."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "The value used to module the next token probabilities."},
    )
    top_k: int = field(
        default=50,
        metadata={
            "help": "The number of highest probability vocabulary tokens to keep for top-k-filtering."
        },
    )
    top_p: float = field(
        default=0.95,
        metadata={
            "help": "If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation."
        },
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."},
    )


@dataclass
class DataArguments:
    data_path: str = field()
    data_type: str = field()

    emb_dict_mol: str = field()

    emb_dict_protein: str = field()

    task_type: str = field()

    val_data_path: str = field(default=None)

    output_path: str = field(default="output.txt")

    max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Truncate the input/output sequence to this length if specified."
        },
    )


# #暂时只实现向2d和3d加噪
# def add_noise(raw_data,noise:float):
#     if noise==0:
#         return raw_data
#     #给raw_data添加高斯噪声,raw_data是tensor
#     noise_data=torch.normal(mean=0,std=noise,size=raw_data.size()).to(raw_data)
#     return raw_data+noise_data


def replace_random_chars(input_str, num_replacements=5):
    import random
    import string

    # 将字符串转换为列表，以便于修改
    str_list = list(input_str)

    # 获取字符串的长度
    str_len = len(str_list)

    # 随机替换字符
    for _ in range(num_replacements):
        # 随机选择一个位置
        random_index = random.randint(0, str_len - 1)

        # 随机选择一个字符进行替换
        random_char = random.choice(["N", "C", "O", "=", "c", ")", "("])

        # 进行替换
        str_list[random_index] = random_char

    # 将列表转换回字符串
    return "".join(str_list)


def run():
    # load the arguments
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # load the merged model and tokenizer
    model, tokenizer, config, _ = load_mollama(
        model_args, data_args, add_frg_vocab=False
    )
    model.to("cuda")
    model = torch.compile(model=model)

    # load the dataset
    processor = MoLlamaProcessor(tokenizer=tokenizer, max_length=data_args.max_length)
    data_collator = TrainMoLlamaCollator(processor=processor, config=config)
    test_dataset = load_dataset(data_args, val=False)

    # generation config
    generation_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=model_args.do_sample,
        top_k=model_args.top_k,
        top_p=model_args.top_p,
        temperature=model_args.temperature,
        num_beams=model_args.num_beams,
    )
    # -------------------------

    # -------------------------
    # inference
    i = 0
    inner_loop = 1
    print(f"Pay attention that the inner_loop maybe not 1. It is {inner_loop}!!!")
    bar = tqdm(total=len(test_dataset))
    while i < len(test_dataset):
        # 处理输入的数据
        item = test_dataset[i]

        if i == 0:
            print("!!!!", item[0])

        data = data_collator([item])
        if i == 0:
            print(tokenizer.tokenize(item[0] + item[1]))
            print("-" * 100)
            print(data["input_ids"][0])
            print(tokenizer.decode(data["input_ids"][0]))
            print("-" * 100)
        # data中labels为-100的部分，即为不计算损失的部分
        data["input_ids"] = data["input_ids"][data["labels"] == -100].unsqueeze(dim=0)
        data["attention_mask"] = data["attention_mask"][
            data["labels"] == -100
        ].unsqueeze(dim=0)

        data.pop("labels")
        for k, v in data.items():
            if v is not None:
                data[k] = v.to("cuda")

        if i == 0:
            print(tokenizer.decode(data["input_ids"][0]))
            print("-" * 100)

        # 内部循环，对同一个输入，产生inner_loop个不同的输出

        for _ in range(inner_loop):
            # 模型推理
            with torch.no_grad():
                output = model.generate(
                    **data,
                    generation_config=generation_config,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            # 进行输出的处理
            output = output[:, data["input_ids"].size(1) :]
            if (
                "llama2" in model_args.model_name_or_path.lower()
                or "llama-2" in model_args.model_name_or_path.lower()
            ):
                gen = (
                    tokenizer.decode(output[0], skip_special_tokens=False)
                    .split("[/INST]")[-1]
                    .strip()
                    .split("</s>")[0]
                )
            else:
                gen = (
                    tokenizer.decode(output[0], skip_special_tokens=False)
                    .split("assistant")[-1]
                    .split("\n\n")[-1]
                    .strip()
                    .split("<|eot_id|>")[0]
                )
            gt = item[1].strip()

            # 写入jsonl文件
            with open(data_args.output_path, "a", encoding="utf-8") as f:
                json.dump({"gen": gen, "gt": gt}, f, ensure_ascii=False)
                f.write("\n")

        # 回显第一条数据的输出
        if i == 0:
            print(tokenizer.decode(output[0], skip_special_tokens=False))
            print("-" * 100)
            print(gen)
            print("-" * 100)
        # 进度显示
        bar.update(1)
        i = i + 1
        if i / float(len(test_dataset)) * 100 % 10 == 0:
            print(f"processed {i} samples; {i/len(test_dataset)}")
    bar.close()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run()
