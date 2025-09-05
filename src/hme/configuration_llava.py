"""HME model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama import LlamaConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class HMEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a `HMEForConditionalGeneration`.
    It is used to instantiate an HME model according to the specified arguments, defining the
    model architecture.

    Configuration objects inherit from `PretrainedConfig` and can be used to control the model outputs.
    Read the documentation from `PretrainedConfig` for more information.

    Parameters
    ----------
    text_config : Union[AutoConfig, dict], optional, default=LlamaConfig()
        The config object or dictionary of the text backbone.
    molecule_2d_hidden_size : int, optional, default=300
        The hidden size of the 2D molecule features.
    molecule_3d_hidden_size : int, optional, default=512
        The hidden size of the 3D molecule features.
    protein_hidden_size : int, optional, default=128
        The hidden size of the protein features.
    ignore_index : int, optional, default=-100
        The ignore index for the loss function.
    molecule_2d_token_index : int, optional
        The token index in the vocabulary to represent 2D molecule features.
    molecule_3d_token_index : int, optional
        The token index in the vocabulary to represent 3D molecule features.
    protein_token_index : int, optional
        The token index in the vocabulary to represent protein features.
    projector_hidden_act : str, optional, default="gelu"
        The activation function used by the multimodal projector.
    modal_padding : int, optional, default=-100
        The padding value used for multi-modal features.
    kwargs : dict, optional
        Additional keyword arguments passed to the `PretrainedConfig` parent class.

    """

    model_type = "HME"
    is_composition = False

    def __init__(
        self,
        text_config=None,
        molecule_2d_hidden_size: int = 300,
        molecule_3d_hidden_size: int = 512,
        protein_hidden_size: int = 128,
        ignore_index: int = -100,
        molecule_2d_token_index: int = None,
        molecule_3d_token_index: int = None,
        protein_token_index: int = None,
        projector_hidden_act: str = "gelu",
        modal_padding: int = -100,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = LlamaConfig()
            logger.info("text_config is None. Initializing with a default LlamaConfig.")
        self.text_config = text_config

        self.molecule_3d_hidden_size = molecule_3d_hidden_size
        self.molecule_2d_hidden_size = molecule_2d_hidden_size
        self.protein_hidden_size = protein_hidden_size
        self.ignore_index = ignore_index
        self.molecule_2d_token_index = molecule_2d_token_index
        self.molecule_3d_token_index = molecule_3d_token_index
        self.protein_token_index = protein_token_index
        self.projector_hidden_act = projector_hidden_act
        self.modal_padding = modal_padding
        self.tie_word_embeddings = False