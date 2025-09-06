import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import TensorType


class HMEDataset(Dataset):
    """
    A PyTorch Dataset for handling various multi-modal chemistry tasks.

    This dataset loads raw text data from a JSON file and corresponding molecular/protein
    embeddings from pre-computed dictionaries. It formats the data based on the specified
    task type and data modalities.

    Parameters
    ----------
    data_path : str
        Path to the main JSON file containing the dataset.
    task_type : str
        The type of task (e.g., 'qa', 'caption', 'pdbbind_reg'). This determines
        how prompts and labels are formatted.
    data_type : str
        A comma-separated string of data modalities to include (e.g., '1d,2d,3d').
        Supported types are '1d' (SMILES), '2d', '3d', and 'frg' (fragments).
    emb_dict_mol : str
        Path to the pre-computed molecule embeddings dictionary (.pt file). Can be 'none'.
    emb_dict_protein : str
        Path to the pre-computed protein embeddings dictionary (.pt file). Can be 'none'.
    val : bool, optional, default=False
        If True, use a smaller subset of the data for validation purposes.
    """

    def __init__(
        self,
        data_path: str,
        task_type: str,
        data_type: str,
        emb_dict_mol: str,
        emb_dict_protein: str,
        val: bool = False,
    ):
        super().__init__()
        self.data_path = data_path
        self.task_type = task_type
        self.data_type = data_type.split(",")

        if not self.data_type:
            raise ValueError("No data type provided.")
        for d in self.data_type:
            if d not in ["1d", "2d", "3d", "frg"]:
                raise ValueError(f"Unsupported data type: {d}")

        self.__load_raw_dataset(
            emb_dict_mol=emb_dict_mol, emb_dict_protein=emb_dict_protein, val=val
        )
        self.__clear_frg()

    def __load_raw_dataset(self, emb_dict_mol: str, emb_dict_protein: str, val: bool):
        """
        Loads the raw JSON data and embedding dictionaries, and performs filtering.

        Note on embedding paths:
        - QA/Captioning: '/path/to/pubchem_train.json.cfm.pt'
        - Pretraining: '/path/to/pretrain.json.cfm.pt'
        - CrossDocked: '/path/to/crossdocked2020/crossdocked_pocket10_train.pt'
        - ...
        """
        with open(self.data_path, "r") as f:
            self.json_list = json.load(f)

        self.emb_dict_mol = (
            torch.load(emb_dict_mol, map_location="cpu")
            if emb_dict_mol.lower() != "none"
            and ("2d" in self.data_type or "3d" in self.data_type)
            else None
        )
        self.emb_dict_protein = (
            torch.load(emb_dict_protein, map_location="cpu")
            if emb_dict_protein.lower() != "none"
            else None
        )

        if "_reg" in self.task_type:
            if "v2016" in self.data_path:
                mean = 6.486140251159668
                std = 2.174337387084961
            else:
                raise ValueError(
                    f"Unknown task_type: {self.task_type} for regression. Please provide a valid data_path."
                )
            for item in self.json_list:
                item["label"] = (item["label"] - mean) / std

        removed = 0
        if self.emb_dict_mol is not None:
            self.emb_dict_mol = {
                k: v
                for k, v in self.emb_dict_mol.items()
                if (
                    v["molecule_raw_2d_features"] is not None
                    and v["molecule_raw_3d_features"] is not None
                )
            }
            new_json_list = []
            for item in self.json_list:
                if item["smiles"] in self.emb_dict_mol:
                    new_json_list.append(item)
                else:
                    removed += 1
            self.json_list = new_json_list

        if self.emb_dict_protein is not None:
            self.emb_dict_protein = {
                k: v for k, v in self.emb_dict_protein.items() if v is not None
            }
            new_json_list = []
            for item in self.json_list:
                if (
                    "meta_data" in item
                    and item["meta_data"]["pocket_file"] in self.emb_dict_protein
                ) or (
                    "pocket_path" in item
                    and item["pocket_path"] in self.emb_dict_protein
                ):
                    new_json_list.append(item)
                else:
                    removed += 1
            self.json_list = new_json_list

        if self.emb_dict_mol is not None or self.emb_dict_protein is not None:
            print(
                f"Now the length of the dataset is {len(self.json_list)}"
            )

        if val:
            self.json_list = self.json_list[:10000]

    def __len__(self) -> int:
        return len(self.json_list)

    def __input_sequence_format(self, seq: str, item: dict) -> str:
        """Formats the input sequence by appending modality information."""
        if "2d" in self.data_type:
            seq = f"{seq} Molecular 2D features are: " + "<molecule_2d>" * 8 + "."
        if "3d" in self.data_type:
            seq = f"{seq} Molecular 3D features are: " + "<molecule_3d>" * 16 + "."
        if "1d" in self.data_type:
            smiles = item["smiles"]
            seq = f"{seq} Molecular SMILES is: {smiles}."
        if "frg" in self.data_type:
            fragments = item["fragments"]
            seq = f"{seq} Molecular fragments are: {fragments}."
        return seq

    def __clear_frg(self):
        """Cleans and standardizes fragment strings in the dataset."""
        if not self.json_list or "fragments" not in self.json_list[0]:
            return
        for item in self.json_list:
            frg_without_ffix = re.findall(r"(?<=<\|).+?(?=\|>)", item["fragments"])
            frg_without_ffix = list(set(frg_without_ffix))
            frg_with_ffix = [f"<|{i}|>" for i in frg_without_ffix]
            frg_with_ffix.sort()
            item["fragments"] = "".join(frg_with_ffix)
            if item["fragments"] == "":
                item["fragments"] = "None"

    @torch.no_grad()
    def __getitem__(self, idx: int) -> List:
        """
        Retrieves a single data point from the dataset.

        Returns
        -------
        List
            A list containing [question, answer, molecule_2d_features,
            molecule_3d_features, protein_features].
        """
        item = self.json_list[idx]
        item_2d, item_3d = None, None

        if self.emb_dict_mol is not None:
            if "2d" in self.data_type:
                item_2d = self.emb_dict_mol[item["smiles"]]["molecule_raw_2d_features"]
            if "3d" in self.data_type:
                item_3d = self.emb_dict_mol[item["smiles"]]["molecule_raw_3d_features"]

        if self.task_type == "qa":
            (
                question,
                answer,
                molecule_raw_2d_features,
                molecule_raw_3d_features,
                protein_raw_features,
            ) = (
                self.__input_sequence_format(item["instruction"], item),
                item["output"],
                item_2d,
                item_3d,
                None,
            )
        elif self.task_type == "caption":
            (
                question,
                answer,
                molecule_raw_2d_features,
                molecule_raw_3d_features,
                protein_raw_features,
            ) = (
                self.__input_sequence_format("Please describe the molecule:", item),
                item["description"],
                item_2d,
                item_3d,
                None,
            )
        elif self.task_type == "pretrain-s2":
            (
                question,
                answer,
                molecule_raw_2d_features,
                molecule_raw_3d_features,
                protein_raw_features,
            ) = (
                self.__input_sequence_format("Please describe the molecule:", item),
                item["description"],
                item_2d,
                item_3d,
                None,
            )
        elif self.task_type == "pretrain":
            (
                question,
                answer,
                molecule_raw_2d_features,
                molecule_raw_3d_features,
                protein_raw_features,
            ) = (
                self.__input_sequence_format("", item),
                item["description"],
                item_2d,
                item_3d,
                None,
            )
        elif self.task_type == "text2smi":
            (
                question,
                answer,
                molecule_raw_2d_features,
                molecule_raw_3d_features,
                protein_raw_features,
            ) = (
                f"Please give me molecular SMILES based on the description: {item['description']}",
                f"{item['smiles']}",
                item_2d,
                item_3d,
                None,
            )
        elif self.task_type == "text2frgsmi":
            (
                question,
                answer,
                molecule_raw_2d_features,
                molecule_raw_3d_features,
                protein_raw_features,
            ) = (
                f"Please give me molecular fragments based on the description. And then give me the molecular SMILES based on both the fragments and the description. The description is: {item['description']}",
                f"Molecular fragments are: {item['fragments']} Molecular SMILES is: {item['smiles']}",
                item_2d,
                item_3d,
                None,
            )
        elif self.task_type == "textfrg2smi":
            (
                question,
                answer,
                molecule_raw_2d_features,
                molecule_raw_3d_features,
                protein_raw_features,
            ) = (
                f"There are some conditions, including logp (the hydrophobicity and solubility balance), qed (the drug-likeness), sas (the synthetic accessibility score), and the fragments (include specific fragments). Now please design a molecule under the given constraints: {item['description']}",
                f"{item['output']}",
                item_2d,
                item_3d,
                None,
            )
        elif self.task_type == "textprotein2frgsmi":
            (
                question,
                answer,
                molecule_raw_2d_features,
                molecule_raw_3d_features,
                protein_raw_features,
            ) = (
                f"Your task is to design a small-molecule ligand. You should first design the molecular fragments and then design the molecular SMILES. If any property constraints are provided (e.g., logP: octanol-water partition coefficient; QED: Quantitative Estimation of Drug-likeness; SAS: Synthetic Accessibility Score； Affinity: binding affinity predicted by AutoDock Vina), the designed molecule should satisfy them.\n\nNow please design a molecule under the given constraints:\nThe ligand should bind to this protein pocket: {'<protein>'*32}. {item['description']}",
                f"Molecular fragments are: {item['fragments']}. Molecular SMILES is: {item['smiles']}",
                item_2d,
                item_3d,
                self.emb_dict_protein[item["meta_data"]["pocket_file"]],
            )
        elif self.task_type == "pdbbind_reg":
            (
                question,
                answer,
                molecule_raw_2d_features,
                molecule_raw_3d_features,
                protein_raw_features,
            ) = (
                self.__input_sequence_format(
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a chemist.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nYour task is to predict the affinity of the molecule with the protein pocket. The protein pocket is: {'<protein>'*32}.",
                    item,
                )
                + " Based on the given information, the affinity of the molecule with the protein pocket is:<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
                item["label"],
                item_2d,
                item_3d,
                self.emb_dict_protein[item["pocket_path"]],
            )
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        return [
            question,
            answer,
            (
                molecule_raw_2d_features.bfloat16()
                if molecule_raw_2d_features is not None
                else None
            ),
            (
                molecule_raw_3d_features.bfloat16()
                if molecule_raw_3d_features is not None
                else None
            ),
            (
                protein_raw_features.bfloat16()
                if protein_raw_features is not None
                else None
            ),
        ]


class HMEProcessor:
    """
    A processor that combines a tokenizer with handling for multi-modal features.

    This class wraps a Hugging Face tokenizer and extends its functionality to carry
    through molecular and protein features alongside tokenized text inputs.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        A pre-trained tokenizer instance.
    max_length : int
        The maximum sequence length for tokenization.
    """

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    @torch.no_grad()
    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        molecule_raw_2d_features: Optional[torch.FloatTensor] = None,
        molecule_raw_3d_features: Optional[torch.FloatTensor] = None,
        protein_raw_features: Optional[torch.FloatTensor] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = None,
        add_special_tokens: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        if text is None:
            raise ValueError("text is required")
        if max_length is None:
            max_length = self.max_length

        text_inputs = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )

        return BatchFeature(
            data={
                **text_inputs,
                "molecule_raw_2d_features": molecule_raw_2d_features,
                "molecule_raw_3d_features": molecule_raw_3d_features,
                "protein_raw_features": protein_raw_features,
            }
        )

    @torch.no_grad()
    def batch_decode(self, *args, **kwargs):
        """Wraps the tokenizer's batch_decode method."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    @torch.no_grad()
    def decode(self, *args, **kwargs):
        """Wraps the tokenizer's decode method."""
        return self.tokenizer.decode(*args, **kwargs)


@dataclass
class MoleculeQAObject:
    """A data class to hold processed question/answer pairs and modal features."""

    q_input_ids: torch.Tensor
    a_input_ids: torch.Tensor
    molecule_raw_2d_features: Optional[torch.Tensor]
    molecule_raw_3d_features: Optional[torch.Tensor]
    protein_raw_features: Optional[torch.Tensor]


@torch.no_grad()
def build_molecule_qa_input(
    processor: HMEProcessor,
    question: str,
    answer: str,
    molecule_raw_2d_features: Optional[torch.Tensor] = None,
    molecule_raw_3d_features: Optional[torch.Tensor] = None,
    protein_raw_features: Optional[torch.Tensor] = None,
) -> MoleculeQAObject:
    """
    Constructs a MoleculeQAObject by processing raw data with the HMEProcessor.
    """
    messages = [
        {"role": "system", "content": "You are a chemist."},
        {"role": "user", "content": question},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_special_tokens=False
    )
    inputs = processor(
        text=prompt,
        molecule_raw_2d_features=molecule_raw_2d_features,
        molecule_raw_3d_features=molecule_raw_3d_features,
        protein_raw_features=protein_raw_features,
        return_tensors="pt",
    )
    outputs = processor(text=answer, return_tensors="pt")

    return MoleculeQAObject(
        q_input_ids=inputs["input_ids"],
        a_input_ids=outputs["input_ids"],
        molecule_raw_2d_features=inputs["molecule_raw_2d_features"],
        molecule_raw_3d_features=inputs["molecule_raw_3d_features"],
        protein_raw_features=inputs["protein_raw_features"],
    )


class TrainHMECollator:
    """
    Data collator for training HME models on sequence-to-sequence tasks.

    This collator processes a batch of samples from HMEDataset, tokenizes them,
    constructs labels for language modeling, and pads all tensors to form a batch.

    Parameters
    ----------
    processor : HMEProcessor
        The processor for tokenizing text and handling modal features.
    config : object
        A configuration object containing `ignore_index` and `modal_padding` values.
    """

    def __init__(self, processor: HMEProcessor, config: Any) -> None:
        self.processor = processor
        self.ignore_index = config.ignore_index
        self.modal_padding = config.modal_padding

    def convert_one_piece(self, mqa_obj: MoleculeQAObject) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Converts a single processed sample into input_ids and labels."""
        input_ids = torch.concat(
            [
                mqa_obj.q_input_ids,
                mqa_obj.a_input_ids,
                torch.tensor(
                    self.processor.tokenizer.eos_token_id, dtype=torch.int64
                ).view(1, -1),
            ],
            dim=1,
        )
        labels = torch.concat(
            [
                torch.full(mqa_obj.q_input_ids.shape, self.ignore_index),
                mqa_obj.a_input_ids,
                torch.tensor(
                    self.processor.tokenizer.eos_token_id, dtype=torch.int64
                ).view(1, -1),
            ],
            dim=1,
        )
        return (
            input_ids.squeeze(0),
            labels.squeeze(0),
            mqa_obj.molecule_raw_2d_features,
            mqa_obj.molecule_raw_3d_features,
            mqa_obj.protein_raw_features,
        )

    @torch.no_grad()
    def __call__(self, features: List[List]) -> Dict[str, torch.Tensor]:
        """Collates a list of samples into a single batch dictionary."""
        input_ids_list = []
        labels_list = []
        molecule_raw_2d_features_list = []
        molecule_raw_3d_features_list = []
        protein_raw_features_list = []

        for feature in features:
            mqa_obj = build_molecule_qa_input(
                processor=self.processor,
                question=feature[0],
                answer=feature[1],
                molecule_raw_2d_features=feature[2],
                molecule_raw_3d_features=feature[3],
                protein_raw_features=feature[4],
            )
            (
                temp_input_ids,
                temp_labels,
                temp_molecule_raw_2d_features,
                temp_molecule_raw_3d_features,
                temp_protein_raw_features,
            ) = self.convert_one_piece(mqa_obj)

            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            if temp_molecule_raw_2d_features is not None:
                molecule_raw_2d_features_list.append(temp_molecule_raw_2d_features)
            if temp_molecule_raw_3d_features is not None:
                molecule_raw_3d_features_list.append(temp_molecule_raw_3d_features)
            if temp_protein_raw_features is not None:
                protein_raw_features_list.append(temp_protein_raw_features)

        final_input_ids = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        final_labels = pad_sequence(
            labels_list, batch_first=True, padding_value=self.ignore_index
        )
        attention_mask = final_input_ids.ne(
            self.processor.tokenizer.pad_token_id
        ).long()

        final_molecule_raw_2d_features = (
            pad_sequence(
                molecule_raw_2d_features_list,
                batch_first=True,
                padding_value=self.modal_padding,
            )
            if molecule_raw_2d_features_list
            else None
        )
        final_molecule_raw_3d_features = (
            pad_sequence(
                molecule_raw_3d_features_list,
                batch_first=True,
                padding_value=self.modal_padding,
            )
            if molecule_raw_3d_features_list
            else None
        )
        final_protein_raw_features = (
            pad_sequence(
                protein_raw_features_list,
                batch_first=True,
                padding_value=self.modal_padding,
            )
            if protein_raw_features_list
            else None
        )

        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "molecule_raw_2d_features": final_molecule_raw_2d_features,
            "molecule_raw_3d_features": final_molecule_raw_3d_features,
            "protein_raw_features": final_protein_raw_features,
            "attention_mask": attention_mask,
        }


class TrainHMECollatorRegression:
    """
    Data collator for training HME models on sequence-level REGRESSION tasks.

    Parameters
    ----------
    processor : HMEProcessor
        The processor for tokenizing text.
    config : object
        A configuration object with a `modal_padding` attribute.
    """

    def __init__(self, processor: HMEProcessor, config: Any) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.modal_padding = config.modal_padding
        self.pad_token_id = self.tokenizer.pad_token_id
        assert self.pad_token_id is not None, "Tokenizer must have a pad_token_id."

    @torch.no_grad()
    def __call__(self, features: List[List]) -> Dict[str, torch.Tensor]:
        """
        Processes a list of features for a regression task.

        Each feature is a list: [text_prompt, regression_label, mol_2d, mol_3d, protein].
        """
        input_ids_list = []
        labels_list = []
        molecule_raw_2d_features_list = []
        molecule_raw_3d_features_list = []
        protein_raw_features_list = []

        for feature_tuple in features:
            text_prompt = feature_tuple[0]
            regression_label = feature_tuple[1]

            tokenized_output = self.tokenizer(
                text_prompt, return_tensors="pt", max_length=self.processor.max_length
            )
            input_ids = tokenized_output.input_ids.squeeze(0)
            input_ids_list.append(input_ids)

            if isinstance(regression_label, (float, int)):
                labels_list.append(
                    torch.tensor([regression_label], dtype=torch.bfloat16)
                )
            elif isinstance(regression_label, torch.Tensor):
                labels_list.append(
                    regression_label.float().unsqueeze(0)
                    if regression_label.ndim == 0
                    else regression_label.bfloat16()
                )
            else:
                raise TypeError(f"Unsupported label type: {type(regression_label)}")

            mol2d_feat = feature_tuple[2]
            if mol2d_feat is not None:
                molecule_raw_2d_features_list.append(mol2d_feat)
            mol3d_feat = feature_tuple[3]
            if mol3d_feat is not None:
                molecule_raw_3d_features_list.append(mol3d_feat)
            prot_feat = feature_tuple[4]
            if prot_feat is not None:
                protein_raw_features_list.append(prot_feat)

        final_input_ids = pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.pad_token_id
        )
        attention_mask = final_input_ids.ne(self.pad_token_id).long()
        final_labels = torch.stack(labels_list, dim=0)
        if final_labels.ndim == 1:
            final_labels = final_labels.unsqueeze(-1)

        final_molecule_raw_2d_features = None
        if molecule_raw_2d_features_list:
            if any(f is None for f in molecule_raw_2d_features_list):
                raise ValueError(
                    "Found None in non-empty molecule_raw_2d_features_list."
                )
            final_molecule_raw_2d_features = pad_sequence(
                molecule_raw_2d_features_list,
                batch_first=True,
                padding_value=self.modal_padding,
            )

        final_molecule_raw_3d_features = None
        if molecule_raw_3d_features_list:
            if any(f is None for f in molecule_raw_3d_features_list):
                raise ValueError(
                    "Found None in non-empty molecule_raw_3d_features_list."
                )
            final_molecule_raw_3d_features = pad_sequence(
                molecule_raw_3d_features_list,
                batch_first=True,
                padding_value=self.modal_padding,
            )

        final_protein_raw_features = None
        if protein_raw_features_list:
            if any(f is None for f in protein_raw_features_list):
                raise ValueError("Found None in non-empty protein_raw_features_list.")
            final_protein_raw_features = pad_sequence(
                protein_raw_features_list,
                batch_first=True,
                padding_value=self.modal_padding,
            )

        return {
            "input_ids": final_input_ids,
            "attention_mask": attention_mask,
            "labels": final_labels,
            "molecule_raw_2d_features": final_molecule_raw_2d_features,
            "molecule_raw_3d_features": final_molecule_raw_3d_features,
            "protein_raw_features": final_protein_raw_features,
        }


class TrainHMECollatorClassification:
    """
    Data collator for training HME models on sequence-level CLASSIFICATION tasks.

    Parameters
    ----------
    processor : HMEProcessor
        The processor for tokenizing text.
    config : object
        A configuration object with a `modal_padding` attribute.
    """

    def __init__(self, processor: HMEProcessor, config: Any) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.modal_padding = config.modal_padding
        self.pad_token_id = self.tokenizer.pad_token_id
        assert self.pad_token_id is not None, "Tokenizer must have a pad_token_id."

    @torch.no_grad()
    def __call__(self, features: List[List]) -> Dict[str, torch.Tensor]:
        """
        Processes a list of features for a classification task.

        Each feature is a list: [text_prompt, class_label, mol_2d, mol_3d, protein].
        """
        input_ids_list = []
        labels_list = []
        molecule_raw_2d_features_list = []
        molecule_raw_3d_features_list = []
        protein_raw_features_list = []

        for feature_tuple in features:
            text_prompt = feature_tuple[0]
            classification_label = feature_tuple[1]

            tokenized_output = self.tokenizer(
                text_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.processor.max_length,
            )
            input_ids = tokenized_output.input_ids.squeeze(0)
            input_ids_list.append(input_ids)

            if isinstance(classification_label, (int, float)):
                labels_list.append(
                    torch.tensor(int(classification_label), dtype=torch.long)
                )
            elif isinstance(classification_label, torch.Tensor):
                labels_list.append(classification_label.long().squeeze())
            else:
                raise TypeError(
                    f"Unsupported label type for classification: {type(classification_label)}"
                )

            mol2d_feat = feature_tuple[2]
            if mol2d_feat is not None:
                molecule_raw_2d_features_list.append(mol2d_feat)
            mol3d_feat = feature_tuple[3]
            if mol3d_feat is not None:
                molecule_raw_3d_features_list.append(mol3d_feat)
            prot_feat = feature_tuple[4]
            if prot_feat is not None:
                protein_raw_features_list.append(prot_feat)

        final_input_ids = pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.pad_token_id
        )
        attention_mask = final_input_ids.ne(self.pad_token_id).long()
        final_labels = torch.stack(labels_list, dim=0)

        final_molecule_raw_2d_features = None
        if molecule_raw_2d_features_list:
            final_molecule_raw_2d_features = pad_sequence(
                molecule_raw_2d_features_list,
                batch_first=True,
                padding_value=self.modal_padding,
            )
        final_molecule_raw_3d_features = None
        if molecule_raw_3d_features_list:
            final_molecule_raw_3d_features = pad_sequence(
                molecule_raw_3d_features_list,
                batch_first=True,
                padding_value=self.modal_padding,
            )
        final_protein_raw_features = None
        if protein_raw_features_list:
            final_protein_raw_features = pad_sequence(
                protein_raw_features_list,
                batch_first=True,
                padding_value=self.modal_padding,
            )

        return {
            "input_ids": final_input_ids,
            "attention_mask": attention_mask,
            "labels": final_labels,
            "molecule_raw_2d_features": final_molecule_raw_2d_features,
            "molecule_raw_3d_features": final_molecule_raw_3d_features,
            "protein_raw_features": final_protein_raw_features,
        }
