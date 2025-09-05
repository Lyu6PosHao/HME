"""PyTorch HME model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging

from hme.configuration_llava import HMEConfig

logger = logging.get_logger(__name__)


@dataclass
class HMECausalLMOutputWithPast(ModelOutput):
    """
    Output type of `HMEForConditionalGeneration`.

    Parameters
    ----------
    loss : torch.FloatTensor of shape (1,), optional
        Language modeling loss (for next-token prediction).
    logits : torch.FloatTensor of shape (batch_size, sequence_length, vocab_size)
        Prediction scores of the language modeling head.
    past_key_values : List[torch.FloatTensor], optional
        Contains pre-computed hidden-states (key and values in the attention blocks)
        that can be used to speed up sequential decoding.
    hidden_states : Tuple[torch.FloatTensor], optional
        Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
        one for the output of each layer). of shape `(batch_size, sequence_length, hidden_size)`.
    attentions : Tuple[torch.FloatTensor], optional
        Tuple of `torch.FloatTensor` (one for each layer) of shape
        `(batch_size, num_heads, sequence_length, sequence_length)`.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class HMERegressionOutput(ModelOutput):
    """
    Output type for `HMEForSequenceRegression`.

    Parameters
    ----------
    loss : torch.FloatTensor of shape (1,), optional
        The regression loss, typically Mean Squared Error.
    logits : torch.FloatTensor of shape (batch_size, 1)
        The predicted regression values.
    hidden_states : Tuple[torch.FloatTensor], optional
        Hidden states of the model.
    attentions : Tuple[torch.FloatTensor], optional
        Attentions weights of the model.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class HMESequenceClassificationOutput(ModelOutput):
    """
    Output type for `HMEForSequenceClassification`.

    Parameters
    ----------
    loss : torch.FloatTensor of shape (1,), optional
        The classification loss.
    logits : torch.FloatTensor of shape (batch_size, num_labels)
        The classification scores (before SoftMax).
    hidden_states : Tuple[torch.FloatTensor], optional
        Hidden states of the model.
    attentions : Tuple[torch.FloatTensor], optional
        Attentions weights of the model.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) module.

    This module implements GQA, where the number of query heads is a multiple of the
    number of key/value heads. This is designed as a drop-in replacement for standard
    multi-head attention.

    Parameters
    ----------
    embed_dim : int
        The embedding dimension of the input.
    num_heads : int
        The number of query heads.
    num_kv_heads : int
        The number of key/value heads. `num_heads` must be divisible by `num_kv_heads`.
    batch_first : bool, optional, default=True
        If `True`, then the input and output tensors are provided as (batch, seq, feature).
        Currently only `True` is supported.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        batch_first: bool = True,
    ):
        super().__init__()
        if not batch_first:
            raise NotImplementedError("batch_first=False is not supported.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})."
            )

        self.head_dim = embed_dim // num_heads
        self.num_groups = num_heads // num_kv_heads

        self.qq_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kk_proj = nn.Linear(embed_dim, self.head_dim * num_kv_heads, bias=False)
        self.vv_proj = nn.Linear(embed_dim, self.head_dim * num_kv_heads, bias=False)
        self.oo_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Grouped Query Attention.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape `(batch_size, seq_len_q, embed_dim)`.
        key : torch.Tensor
            Key tensor of shape `(batch_size, seq_len_kv, embed_dim)`.
        value : torch.Tensor
            Value tensor of shape `(batch_size, seq_len_kv, embed_dim)`.
        key_padding_mask : Optional[torch.Tensor]
            Mask for keys of shape `(batch_size, seq_len_kv)`. `True` indicates a masked position.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - The attention output tensor of shape `(batch_size, seq_len_q, embed_dim)`.
            - The attention weights tensor of shape `(batch_size, num_heads, seq_len_q, seq_len_kv)`.
        """
        batch_size, q_len, _ = query.shape
        kv_len = key.shape[1]

        q = (
            self.qq_proj(query)
            .view(batch_size, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.kk_proj(key)
            .view(batch_size, kv_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.vv_proj(value)
            .view(batch_size, kv_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, q_len, self.embed_dim)
        )
        output = self.oo_proj(attn_output)

        return output, attn_weights


class QueryLearning(nn.Module):
    """
    Projects raw features from a modality into an embedding space and then samples
    them into a fixed number of tokens using a learned set of queries.

    This module combines input projection with a perceiver-style sampler.

    Parameters
    ----------
    config : HMEConfig
        The model configuration.
    modal : str
        The modality name, one of "2d", "3d", or "protein".
    embed_dim : int
        The dimension of the intermediate embedding space.
    num_queries : int
        The number of learnable queries to use for sampling, determining the output token count.
    num_heads : int
        The number of attention heads for the sampler.
    num_kv_heads : int
        The number of key/value heads for the sampler's GQA.
    """

    def __init__(
        self,
        config: HMEConfig,
        modal: str,
        embed_dim: int,
        num_queries: int,
        num_heads: int,
        num_kv_heads: int,
    ):
        super().__init__()
        self.modal_padding = config.modal_padding
        if modal == "2d":
            self.xd_dim = config.molecule_2d_hidden_size
        elif modal == "3d":
            self.xd_dim = config.molecule_3d_hidden_size
        elif modal == "protein":
            self.xd_dim = config.protein_hidden_size
        else:
            raise ValueError(f"Invalid modal: {modal}")

        self.input_projection = nn.Linear(self.xd_dim, embed_dim, bias=True)
        self.query = nn.Parameter(torch.Tensor(num_queries, embed_dim))
        nn.init.normal_(self.query, std=embed_dim**-0.5)

        self.attn = GroupedQueryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            batch_first=True,
        )
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)

    def forward(
        self, modal_raw_xd_features: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Processes raw modal features.

        Parameters
        ----------
        modal_raw_xd_features : Optional[torch.Tensor]
            Raw input features for the modality of shape `(batch_size, seq_len, xd_dim)`.
            If `None`, returns `None`.

        Returns
        -------
        Optional[torch.Tensor]
            The sampled feature tokens of shape `(batch_size, num_queries, embed_dim)`,
            or `None` if the input was `None`.
        """
        if modal_raw_xd_features is None:
            return None

        key_padding_mask = modal_raw_xd_features.eq(self.modal_padding).all(dim=-1)
        projected_xd_features = self.input_projection(modal_raw_xd_features)
        x_kv = self.ln_kv(projected_xd_features)

        q_base = self.ln_q(self.query)
        q = q_base.unsqueeze(0).repeat(x_kv.size(0), 1, 1)

        attn_output, _ = self.attn(q, x_kv, x_kv, key_padding_mask=key_padding_mask)
        return attn_output


class ModalCrossFuser(nn.Module):
    """
    A multi-layer cross-attention module to fuse features from two modalities.

    Each layer consists of two cross-attention blocks: one where modality A attends to B,
    and another where modality B attends to the updated modality A. This allows for
    bidirectional information flow.

    Parameters
    ----------
    embed_dim : int
        The embedding dimension for both modalities.
    num_heads : int
        The number of attention heads.
    num_layers : int
        The number of fusion layers to stack.
    num_kv_heads : int
        The number of key/value heads for Grouped Query Attention.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, num_layers: int, num_kv_heads: int
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "cross_attn_2d_to_3d": GroupedQueryAttention(
                            embed_dim, num_heads, num_kv_heads, batch_first=True
                        ),
                        "norm_2d_after_attn": nn.LayerNorm(embed_dim),
                        "cross_attn_3d_to_2d": GroupedQueryAttention(
                            embed_dim, num_heads, num_kv_heads, batch_first=True
                        ),
                        "norm_3d_after_attn": nn.LayerNorm(embed_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, tokens_2d: torch.Tensor, tokens_3d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for ModalCrossFuser.

        Parameters
        ----------
        tokens_2d : torch.Tensor
            Tokens from the first modality (e.g., 2D features) of shape `(batch_size, seq_len_2d, embed_dim)`.
        tokens_3d : torch.Tensor
            Tokens from the second modality (e.g., 3D features) of shape `(batch_size, seq_len_3d, embed_dim)`.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the fused tokens for both modalities.
        """
        if tokens_2d is None or tokens_3d is None:
            return tokens_2d, tokens_3d

        feat_2d, feat_3d = tokens_2d, tokens_3d
        for layer_module in self.layers:
            fused_2d_from_3d, _ = layer_module["cross_attn_2d_to_3d"](
                query=feat_2d, key=feat_3d, value=feat_3d, key_padding_mask=None
            )
            feat_2d = feat_2d + fused_2d_from_3d
            feat_2d = layer_module["norm_2d_after_attn"](feat_2d)

            fused_3d_from_2d, _ = layer_module["cross_attn_3d_to_2d"](
                query=feat_3d, key=feat_2d, value=feat_2d, key_padding_mask=None
            )
            feat_3d = feat_3d + fused_3d_from_2d
            feat_3d = layer_module["norm_3d_after_attn"](feat_3d)

        return feat_2d, feat_3d


class FeatureFuser(nn.Module):
    """
    A comprehensive module to process, fuse, and project features from multiple modalities
    (2D molecule, 3D molecule, protein) into the language model's embedding space.

    Parameters
    ----------
    config : HMEConfig
        The model configuration.
    lightweight_embed_dim : int, optional, default=4096
        The intermediate embedding dimension for tokenizers.
    tokenizer_num_attn_heads : int, optional, default=32
        Number of attention heads for the tokenizer samplers.
    tokenizer_num_kv_heads : int, optional, default=8
        Number of key/value heads for the tokenizer samplers (GQA).
    num_queries_2d : int, optional, default=8
        Number of output tokens for the 2D molecule tokenizer.
    num_queries_3d : int, optional, default=16
        Number of output tokens for the 3D molecule tokenizer.
    num_queries_protein : int, optional, default=32
        Number of output tokens for the protein tokenizer.
    fusion_num_layers : int, optional, default=1
        Number of layers in the molecule cross-fuser. If 0, no fusion is performed.
    fusion_num_attn_heads : int, optional, default=32
        Number of attention heads for the cross-fuser.
    fusion_num_kv_heads : int, optional, default=8
        Number of key/value heads for the cross-fuser (GQA).
    """

    def __init__(
        self,
        config: HMEConfig,
        lightweight_embed_dim: int = 4096,
        tokenizer_num_attn_heads: int = 32,
        tokenizer_num_kv_heads: int = 8,
        num_queries_2d: int = 8,
        num_queries_3d: int = 16,
        num_queries_protein: int = 32,
        fusion_num_layers: int = 1,
        fusion_num_attn_heads: int = 32,
        fusion_num_kv_heads: int = 8,
    ):
        super().__init__()
        self.target_hidden_size = config.text_config.hidden_size

        self.tokenizer_2d = QueryLearning(
            config,
            modal="2d",
            embed_dim=lightweight_embed_dim,
            num_queries=num_queries_2d,
            num_heads=tokenizer_num_attn_heads,
            num_kv_heads=tokenizer_num_kv_heads,
        )
        self.tokenizer_3d = QueryLearning(
            config,
            modal="3d",
            embed_dim=lightweight_embed_dim,
            num_queries=num_queries_3d,
            num_heads=tokenizer_num_attn_heads,
            num_kv_heads=tokenizer_num_kv_heads,
        )
        self.tokenizer_protein = QueryLearning(
            config,
            modal="protein",
            embed_dim=lightweight_embed_dim,
            num_queries=num_queries_protein,
            num_heads=tokenizer_num_attn_heads,
            num_kv_heads=tokenizer_num_kv_heads,
        )

        self.molecule_fuser = (
            ModalCrossFuser(
                embed_dim=lightweight_embed_dim,
                num_heads=fusion_num_attn_heads,
                num_layers=fusion_num_layers,
                num_kv_heads=fusion_num_kv_heads,
            )
            if fusion_num_layers > 0
            else None
        )

        self.out_projection_2d = nn.Linear(
            lightweight_embed_dim, self.target_hidden_size, bias=True
        )
        self.out_projection_3d = nn.Linear(
            lightweight_embed_dim, self.target_hidden_size, bias=True
        )
        self.out_projection_protein = nn.Linear(
            lightweight_embed_dim, self.target_hidden_size, bias=True
        )

    def forward(
        self,
        molecule_raw_2d_features: Optional[torch.Tensor],
        molecule_raw_3d_features: Optional[torch.Tensor],
        protein_raw_features: Optional[torch.Tensor],
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """
        Processes and fuses features from all modalities.

        Parameters
        ----------
        molecule_raw_2d_features : Optional[torch.Tensor]
            Raw 2D molecule features.
        molecule_raw_3d_features : Optional[torch.Tensor]
            Raw 3D molecule features.
        protein_raw_features : Optional[torch.Tensor]
            Raw protein features.

        Returns
        -------
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
            A tuple containing the projected tokens for 2D molecule, 3D molecule,
            and protein modalities, respectively. Each element has a shape of
            (batch_size, num_queries, target_hidden_size) or is `None`.
        """
        lightweight_tokens_2d = self.tokenizer_2d(molecule_raw_2d_features)
        lightweight_tokens_3d = self.tokenizer_3d(molecule_raw_3d_features)
        lightweight_tokens_protein = self.tokenizer_protein(protein_raw_features)

        if (
            self.molecule_fuser is not None
            and lightweight_tokens_2d is not None
            and lightweight_tokens_3d is not None
        ):
            fused_lightweight_2d, fused_lightweight_3d = self.molecule_fuser(
                lightweight_tokens_2d, lightweight_tokens_3d
            )
        else:
            fused_lightweight_2d = lightweight_tokens_2d
            fused_lightweight_3d = lightweight_tokens_3d

        projected_2d_tokens = (
            self.out_projection_2d(fused_lightweight_2d)
            if fused_lightweight_2d is not None
            else None
        )
        projected_3d_tokens = (
            self.out_projection_3d(fused_lightweight_3d)
            if fused_lightweight_3d is not None
            else None
        )
        projected_protein_tokens = (
            self.out_projection_protein(lightweight_tokens_protein)
            if lightweight_tokens_protein is not None
            else None
        )

        return (
            projected_2d_tokens,
            projected_3d_tokens,
            projected_protein_tokens,
        )


class HMEPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for
    downloading and loading pretrained models.
    """

    config_class = HMEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module: nn.Module):
        """
        Initializes the weights of the model.

        Note: This implementation is intended for fine-tuning, not training from scratch.
        """
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        elif hasattr(self.config.text_config, "initializer_range"):
            std = self.config.text_config.initializer_range
        else:
            std = 0.02

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """
        Checks if the underlying language model supports Scaled Dot Product Attention.
        """
        return self.language_model._supports_sdpa


class HMEForConditionalGeneration(HMEPreTrainedModel):
    """
    The main HME model for conditional text generation, integrating multi-modal
    features into a large language model.

    Parameters
    ----------
    config : HMEConfig
        The model configuration.
    language_model : PreTrainedModel
        The underlying large language model backbone (e.g., Llama).
    """

    def __init__(self, config: HMEConfig, language_model: PreTrainedModel):
        super().__init__(config)
        self.language_model = language_model
        self.vocab_size = config.text_config.vocab_size
        self.feature_fuser = FeatureFuser(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of
        )
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        molecule_raw_2d_features: Optional[torch.FloatTensor] = None,
        molecule_raw_3d_features: Optional[torch.FloatTensor] = None,
        protein_raw_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, HMECausalLMOutputWithPast]:
        """
        Forward pass of the model.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if input_ids.shape[1] != 1:
                molecule_2d_features, molecule_3d_features, protein_features = (
                    self.feature_fuser(
                        molecule_raw_2d_features,
                        molecule_raw_3d_features,
                        protein_raw_features,
                    )
                )

                if molecule_2d_features is not None:
                    special_molecule_2d_mask = (
                        input_ids == self.config.molecule_2d_token_index
                    ).unsqueeze(-1)
                    special_molecule_2d_mask = special_molecule_2d_mask.expand_as(
                        inputs_embeds
                    ).to(inputs_embeds.device)
                    if (
                        inputs_embeds[special_molecule_2d_mask].numel()
                        != molecule_2d_features.numel()
                    ):
                        n_molecule_2d_tokens = (
                            input_ids == self.config.molecule_2d_token_index
                        ).sum()
                        n_molecule_2d_features = (
                            molecule_2d_features.shape[0]
                            * molecule_2d_features.shape[1]
                        )
                        raise ValueError(
                            f"mol features and mol tokens do not match: tokens: {n_molecule_2d_tokens}, features {n_molecule_2d_features}"
                        )
                    molecule_2d_features = molecule_2d_features.to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(
                        special_molecule_2d_mask, molecule_2d_features
                    )

                if molecule_3d_features is not None:
                    special_molecule_3d_mask = (
                        input_ids == self.config.molecule_3d_token_index
                    ).unsqueeze(-1)
                    special_molecule_3d_mask = special_molecule_3d_mask.expand_as(
                        inputs_embeds
                    ).to(inputs_embeds.device)
                    if (
                        inputs_embeds[special_molecule_3d_mask].numel()
                        != molecule_3d_features.numel()
                    ):
                        n_molecule_3d_tokens = (
                            input_ids == self.config.molecule_3d_token_index
                        ).sum()
                        n_molecule_3d_features = (
                            molecule_3d_features.shape[0]
                            * molecule_3d_features.shape[1]
                        )
                        raise ValueError(
                            f"mol features and mol tokens do not match: tokens: {n_molecule_3d_tokens}, features {n_molecule_3d_features}"
                        )
                    molecule_3d_features = molecule_3d_features.to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(
                        special_molecule_3d_mask, molecule_3d_features
                    )

                if protein_features is not None:
                    special_protein_mask = (
                        input_ids == self.config.protein_token_index
                    ).unsqueeze(-1)
                    special_protein_mask = special_protein_mask.expand_as(
                        inputs_embeds
                    ).to(inputs_embeds.device)
                    if (
                        inputs_embeds[special_protein_mask].numel()
                        != protein_features.numel()
                    ):
                        n_protein_tokens = (
                            input_ids == self.config.protein_token_index
                        ).sum()
                        n_protein_features = (
                            protein_features.shape[0] * protein_features.shape[1]
                        )
                        raise ValueError(
                            f"mol features and mol tokens do not match: tokens: {n_protein_tokens}, features {n_protein_features}"
                        )
                    protein_features = protein_features.to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(
                        special_protein_mask, protein_features
                    )

            elif past_key_values is not None and input_ids.shape[1] == 1:
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]
                batch_index, non_attended_tokens = torch.where(
                    first_layer_past_key_value.float().sum(-2) == 0
                )
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]
                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0
                attention_mask = torch.cat(
                    (extended_attention_mask, attention_mask[:, -target_length:]), dim=1
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]
        loss = None
        if labels is not None:
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = labels[..., 1:][
                    shift_attention_mask.to(labels.device) != 0
                ].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return HMECausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Union[Cache, Tuple]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        molecule_raw_2d_features: Optional[torch.FloatTensor] = None,
        molecule_raw_3d_features: Optional[torch.FloatTensor] = None,
        protein_raw_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            elif (
                self.config.molecule_2d_token_index in input_ids
                or self.config.molecule_3d_token_index in input_ids
                or self.config.protein_token_index in input_ids
            ):
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[
                    :, -(cache_length + input_ids.shape[1]) :
                ]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        model_inputs = (
            {"inputs_embeds": inputs_embeds}
            if inputs_embeds is not None and past_key_values is None
            else {"input_ids": input_ids}
        )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "molecule_raw_2d_features": molecule_raw_2d_features,
                "molecule_raw_3d_features": molecule_raw_3d_features,
                "protein_raw_features": protein_raw_features,
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)


class HMEForSequenceRegression(nn.Module):
    """
    HME model adapted for sequence-level regression tasks.

    This model uses the HME backbone to process multi-modal inputs and adds a
    regression head on top of the pooled output sequence representation.

    Parameters
    ----------
    config : HMEConfig
        The model configuration.
    language_model : nn.Module
        The pre-loaded language model backbone.
    """

    def __init__(self, config: HMEConfig, language_model: nn.Module):
        super().__init__()
        self.config = config
        self.language_model = language_model
        self.feature_fuser = FeatureFuser(config)
        self.regression_head = nn.Sequential(
            nn.LayerNorm(config.text_config.hidden_size),
            nn.Linear(
                config.text_config.hidden_size,
                config.text_config.hidden_size // 2,
                bias=True,
            ),
            nn.GELU(),
            nn.Linear(config.text_config.hidden_size // 2, 1, bias=True),
        )

    def get_output_embeddings(self) -> nn.Linear:
        return self.language_model.get_output_embeddings()

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of
        )
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        molecule_raw_2d_features: Optional[torch.FloatTensor] = None,
        molecule_raw_3d_features: Optional[torch.FloatTensor] = None,
        protein_raw_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, HMERegressionOutput]:
        """Forward pass for sequence regression."""
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        molecule_2d_features, molecule_3d_features, protein_features = (
            self.feature_fuser(
                molecule_raw_2d_features, molecule_raw_3d_features, protein_raw_features
            )
        )

        if molecule_2d_features is not None:
            special_molecule_2d_mask = (
                input_ids == self.config.molecule_2d_token_index
            ).unsqueeze(-1)
            special_molecule_2d_mask = special_molecule_2d_mask.expand_as(
                inputs_embeds
            ).to(inputs_embeds.device)
            if (
                inputs_embeds[special_molecule_2d_mask].numel()
                != molecule_2d_features.numel()
            ):
                n_molecule_2d_tokens = (
                    input_ids == self.config.molecule_2d_token_index
                ).sum()
                n_molecule_2d_features = (
                    molecule_2d_features.shape[0] * molecule_2d_features.shape[1]
                )
                raise ValueError(
                    f"mol features and mol tokens do not match: tokens: {n_molecule_2d_tokens}, features {n_molecule_2d_features}"
                )
            molecule_2d_features = molecule_2d_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_molecule_2d_mask, molecule_2d_features
            )

        if molecule_3d_features is not None:
            special_molecule_3d_mask = (
                input_ids == self.config.molecule_3d_token_index
            ).unsqueeze(-1)
            special_molecule_3d_mask = special_molecule_3d_mask.expand_as(
                inputs_embeds
            ).to(inputs_embeds.device)
            if (
                inputs_embeds[special_molecule_3d_mask].numel()
                != molecule_3d_features.numel()
            ):
                n_molecule_3d_tokens = (
                    input_ids == self.config.molecule_3d_token_index
                ).sum()
                n_molecule_3d_features = (
                    molecule_3d_features.shape[0] * molecule_3d_features.shape[1]
                )
                raise ValueError(
                    f"mol features and mol tokens do not match: tokens: {n_molecule_3d_tokens}, features {n_molecule_3d_features}"
                )
            molecule_3d_features = molecule_3d_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_molecule_3d_mask, molecule_3d_features
            )

        if protein_features is not None:
            special_protein_mask = (
                input_ids == self.config.protein_token_index
            ).unsqueeze(-1)
            special_protein_mask = special_protein_mask.expand_as(inputs_embeds).to(
                inputs_embeds.device
            )
            if inputs_embeds[special_protein_mask].numel() != protein_features.numel():
                n_protein_tokens = (input_ids == self.config.protein_token_index).sum()
                n_protein_features = (
                    protein_features.shape[0] * protein_features.shape[1]
                )
                raise ValueError(
                    f"mol features and mol tokens do not match: tokens: {n_protein_tokens}, features {n_protein_features}"
                )
            protein_features = protein_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_protein_mask, protein_features
            )

        llm_outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        if (
            hasattr(llm_outputs, "last_hidden_state")
            and llm_outputs.last_hidden_state is not None
        ):
            last_hidden_state = llm_outputs.last_hidden_state
        elif (
            hasattr(llm_outputs, "hidden_states")
            and llm_outputs.hidden_states is not None
        ):
            last_hidden_state = llm_outputs.hidden_states[-1]
        else:
            raise RuntimeError(
                "Could not retrieve last hidden state from the language model. "
                "Ensure 'output_hidden_states=True' and the model output format is as expected."
            )

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).bfloat16()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        pooled_output = sum_embeddings / sum_mask.clamp(min=1e-9)

        predictions = self.regression_head(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            labels = labels.to(predictions.device, predictions.dtype)
            loss = loss_fct(predictions.squeeze(-1), labels.squeeze(-1))

        return HMERegressionOutput(loss=loss, logits=predictions)






class HMEForSequenceClassification(nn.Module):
    """
    HME model adapted for sequence-level classification tasks.

    This model uses the HME backbone and adds a classification head on top of the
    pooled output sequence representation.

    Parameters
    ----------
    config : HMEConfig
        The model configuration.
    language_model : nn.Module
        The pre-loaded language model backbone.
    """

    def __init__(self, config: HMEConfig, language_model: nn.Module):
        super().__init__()
        self.config = config
        self.language_model = language_model
        self.feature_fuser = FeatureFuser(config, lightweight_embed_dim=4096)
        self.classification_head = nn.Sequential(
            nn.LayerNorm(config.text_config.hidden_size),
            nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.text_config.hidden_size, 1),
        )

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of
        )
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Linear:
        return self.language_model.get_output_embeddings()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        molecule_raw_2d_features: Optional[torch.FloatTensor] = None,
        molecule_raw_3d_features: Optional[torch.FloatTensor] = None,
        protein_raw_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, HMESequenceClassificationOutput]:
        """Forward pass for sequence classification."""
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        molecule_2d_features, molecule_3d_features, protein_features = (
            self.feature_fuser(
                molecule_raw_2d_features, molecule_raw_3d_features, protein_raw_features
            )
        )

        if molecule_2d_features is not None:
            special_molecule_2d_mask = (
                (input_ids == self.config.molecule_2d_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
            )
            molecule_2d_features = molecule_2d_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_molecule_2d_mask, molecule_2d_features
            )

        if molecule_3d_features is not None:
            special_molecule_3d_mask = (
                (input_ids == self.config.molecule_3d_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
            )
            molecule_3d_features = molecule_3d_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_molecule_3d_mask, molecule_3d_features
            )

        if protein_features is not None:
            special_protein_mask = (
                (input_ids == self.config.protein_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
            )
            protein_features = protein_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_protein_mask, protein_features
            )

        llm_outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        if (
            hasattr(llm_outputs, "last_hidden_state")
            and llm_outputs.last_hidden_state is not None
        ):
            last_hidden_state = llm_outputs.last_hidden_state
        elif (
            hasattr(llm_outputs, "hidden_states")
            and llm_outputs.hidden_states is not None
        ):
            last_hidden_state = llm_outputs.hidden_states[-1]
        else:
            raise RuntimeError(
                "Could not retrieve last hidden state from the language model."
            )

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).bfloat16()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        pooled_output = sum_embeddings / sum_mask.clamp(min=1e-9)

        logits = self.classification_head(pooled_output)
        loss = None
        if labels is not None:
            labels_for_bce = labels.float()

            bce_loss_fct = nn.BCEWithLogitsLoss()
            bce_loss = bce_loss_fct(logits.squeeze(-1), labels_for_bce)
            loss = bce_loss 

        return HMESequenceClassificationOutput(loss=loss, logits=logits)