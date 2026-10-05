import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, SiLU
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    MaskedLMOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from dataclasses import dataclass
from typing import Any, Dict

import torch.distributions as dists
from torch.nn import functional as F
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import ModelOutput

try:
    from tqdm import trange
except ImportError:
    def trange(n, **kwargs):
        return range(n)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)


def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class RotaryEmbedding(torch.nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=2):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(
                self.inv_freq
            )
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k, seq_dimension=-2
        )

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


def create_position_ids_from_input_ids(
    input_ids, padding_idx, past_key_values_length=0
):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (
        torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
    ) * mask
    return incremental_indices.long() + padding_idx


class EsmEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        if config.emb_layer_norm_before:
            self.layer_norm = nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )
        else:
            self.layer_norm = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )
        self.token_dropout = config.token_dropout
        self.mask_token_id = config.mask_token_id

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(
                    inputs_embeds
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # Note that if we want to support ESM-1 (not 1b!) in future then we need to support an
        # embedding_scale factor here.
        embeddings = inputs_embeds

        # Matt: ESM has the option to handle masking in MLM in a slightly unusual way. If the token_dropout
        # flag is False then it is handled in the same was as BERT/RoBERTa. If it is set to True, however,
        # masked tokens are treated as if they were selected for input dropout and zeroed out.
        # This "mask-dropout" is compensated for when masked tokens are not present, by scaling embeddings by
        # a factor of (fraction of unmasked tokens during training) / (fraction of unmasked tokens in sample).
        # This is analogous to the way that dropout layers scale down outputs during evaluation when not
        # actually dropping out values (or, equivalently, scale up their un-dropped outputs in training).
        if self.token_dropout:
            embeddings.masked_fill_(
                (input_ids == self.mask_token_id).unsqueeze(-1), 0.0
            )
            mask_ratio_train = (
                0.15 * 0.8
            )  # Hardcoded as the ratio used in all ESM model training runs
            src_lengths = attention_mask.sum(-1)
            mask_ratio_observed = (input_ids == self.mask_token_id).sum(
                -1
            ).float() / src_lengths
            embeddings = (
                embeddings
                * (1 - mask_ratio_train)
                / (1 - mask_ratio_observed)[:, None, None]
            ).to(embeddings.dtype)

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
        if attention_mask is not None:
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).to(
                embeddings.dtype
            )
        # Matt: I think this line was copied incorrectly from BERT, disabling it for now.
        # embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1,
            sequence_length + self.padding_idx + 1,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class EsmSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        self.rotary_embeddings = None
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )
        elif self.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.

        if past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Matt: Our BERT model (which this code was derived from) scales attention logits down by sqrt(head_dim).
        # ESM scales the query down by the same factor instead. Modulo numerical stability these are equivalent,
        # but not when rotary embeddings get involved. Therefore, we scale the query here to match the original
        # ESM code and fix rotary embeddings.
        query_layer = query_layer * self.attention_head_size**-0.5

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in EsmModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


class EsmSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += input_tensor
        return hidden_states


class EsmAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = EsmSelfAttention(config)
        self.output = EsmSelfOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        hidden_states_ln = self.LayerNorm(hidden_states)
        self_outputs = self.self(
            hidden_states_ln,
            attention_mask,
            head_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class AdaLNEsmAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = EsmSelfAttention(config)
        self.output = EsmSelfOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
          # AdaLN scale/shift from protein_cond before attention
        self.adaln_modulation = nn.Linear(config.hidden_size, 2 *config.hidden_size)

    def forward(
        self,
        hidden_states,
        protein_cond=None,
        attention_mask=None,
        head_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # protein_cond should be one vector per sample, shape about [batch, hidden_size].
        # AdaLN uses the pooled protein condition, not the full protein token sequence.
        if protein_cond is None:
              raise ValueError("protein_cond is required when AdaLN is enabled.")
        hidden_states_ln = self.LayerNorm(hidden_states)

        scale, shift = self.adaln_modulation(protein_cond).chunk(2,dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        hidden_states_ln = hidden_states_ln * (1.0 + scale) + shift

        self_outputs = self.self(
            hidden_states_ln,
            attention_mask,
            head_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class ProteinConditioningAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /config.num_attention_heads)
        self.all_head_size = self.num_attention_heads *self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.rna_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.protein_norm = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)

        self.gate_proj = nn.Linear(config.hidden_size,config.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        rna_hidden_states,
        protein_hidden_states,
        protein_attention_mask=None,
        output_attentions=False,
    ):
        rna_hidden_states_ln = self.rna_norm(rna_hidden_states)
        protein_hidden_states_ln = self.protein_norm(protein_hidden_states)

        query_layer = self.transpose_for_scores(self.query(rna_hidden_states_ln))
        key_layer =self.transpose_for_scores(self.key(protein_hidden_states_ln))
        value_layer =self.transpose_for_scores(self.value(protein_hidden_states_ln))

        query_layer = query_layer * self.attention_head_size**-0.5
        attention_scores = torch.matmul(query_layer,
        key_layer.transpose(-1, -2))

        if protein_attention_mask is not None:
            protein_attention_mask = protein_attention_mask[:, None,None, :].to(dtype=attention_scores.dtype,
                device=attention_scores.device)
            attention_scores = attention_scores + (
                1.0 - protein_attention_mask
            ) * torch.finfo(attention_scores.dtype).min

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] +(self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        protein_update = self.out_proj(context_layer)

        gate = torch.sigmoid(self.gate_proj(rna_hidden_states_ln))
        conditioned_hidden_states = rna_hidden_states + gate * protein_update

        outputs = (
            (conditioned_hidden_states, attention_probs)
            if output_attentions
            else (conditioned_hidden_states,)
        )
        return outputs
    

class EsmIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(
            config.hidden_size,
            int(config.intermediate_size * 2),
            bias=config.add_bias_fnn,
        )
        self.activation_fn = SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)

        # GLU
        x1, x2 = hidden_states.split(int(hidden_states.size(-1) / 2), -1)
        hidden_states = self.activation_fn(x1) * x2

        return hidden_states


class EsmOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.add_bias_fnn
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += input_tensor
        return hidden_states


class EsmLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = EsmAttention(config)
        self.AdaLN_attention = AdaLNEsmAttention(config)
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.use_FiLM = config.use_FiLM
        self.use_AdaLN = config.use_AdaLN
        self.use_gated_bias = config.use_gated_bias
        self.use_protein_conditioning_attention = config.use_protein_conditioning_attention
        protein_dim = getattr(config, "protein_dim", config.hidden_size)
        if protein_dim == config.hidden_size:
            self.protein_proj = nn.Identity()
        else:
            self.protein_proj = nn.Linear(protein_dim, config.hidden_size)

        self.ffn_adaln_modulation = nn.Linear(config.hidden_size, 2 *config.hidden_size)

        if self.use_protein_conditioning_attention:
            self.protein_conditioning_attention = ProteinConditioningAttention(config)

        self.film_modulation = nn.Linear(config.hidden_size, 2 * config.hidden_size)


        self.gated_bias_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.gated_bias_gate = nn.Linear(config.hidden_size, config.hidden_size)



    def forward(
        self,
        hidden_states,
        protein_cond=None,
        attention_mask=None,
        head_mask=None,
        protein_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        
        # protein_cond default shape: [B, Lp, Dp]
        # protein_hidden_states: token-level protein condition, shape [B, Lp, H]
        # pooled_protein_cond: pooled protein condition for AdaLN / FiLM / gated bias, shape [B, H]

        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )

        protein_hidden_states = None
        pooled_protein_cond = None

        if protein_cond is not None:
            protein_hidden_states = self.protein_proj(protein_cond)   # [B, Lp, H]

            if protein_hidden_states.dim() == 3:


                if protein_attention_mask is None:
                    pooled_protein_cond = protein_hidden_states.mean(dim=1)   #[B, H]
                else:
                    # protein_attention_mask: [B, Lp], padding = 0, valid = 1
                    mask = protein_attention_mask.to(protein_hidden_states.dtype).unsqueeze(-1)   #[B, Lp, 1]
                    denom = mask.sum(dim=1).clamp_min(1.0)   # [B, 1]
                    pooled_protein_cond = (protein_hidden_states * mask).sum(dim=1) / denom

            elif protein_hidden_states.dim() == 2:
                pooled_protein_cond = protein_hidden_states
                protein_hidden_states = protein_hidden_states.unsqueeze(1)
            else:
                raise ValueError("protein_cond must have shape [B, Lp, Dp] or [B, Dp].")

        if self.use_AdaLN:
            self_attention_outputs = self.AdaLN_attention(
                hidden_states,
                protein_cond=pooled_protein_cond,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                past_key_value=self_attn_past_key_value,
            )
            attention_output = self_attention_outputs[0]

        else:
            self_attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
                past_key_value=self_attn_past_key_value,
            )
            attention_output = self_attention_outputs[0]

        if self.use_protein_conditioning_attention:
            if protein_hidden_states is None:
                raise ValueError("protein_hidden_states is required when protein conditioning attention is enabled."
                )
            protein_attention_outputs = self.protein_conditioning_attention(
                attention_output,
                protein_hidden_states,
                protein_attention_mask=protein_attention_mask,
                output_attentions=output_attentions,
            )
            attention_output = protein_attention_outputs[0]


        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        # gated_bias
        if self.use_gated_bias:
            if pooled_protein_cond is None:
                raise ValueError("pooled_protein_cond is required when gated bias is enabled.")
            bias = self.gated_bias_proj(pooled_protein_cond)
            gate = torch.sigmoid(self.gated_bias_gate(pooled_protein_cond))
            bias = bias.unsqueeze(1)
            gate = gate.unsqueeze(1)
            attention_output = attention_output + gate * bias

        # FiLM
        if self.use_FiLM:
            if pooled_protein_cond is None:
                raise ValueError("pooled_protein_cond is required when FiLM is enabled.")
            gamma, beta = self.film_modulation(pooled_protein_cond).chunk(2,dim=-1)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            attention_output = attention_output * (1.0 + gamma) + beta

        # FFN
        layer_output = self.feed_forward_chunk(attention_output, protein_cond=pooled_protein_cond)

        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output, protein_cond=None):
        attention_output_ln = self.LayerNorm(attention_output)

        if self.use_AdaLN:
            if protein_cond is None:
                raise ValueError("protein_cond is required when AdaLN is enabled.")
            scale, shift = self.ffn_adaln_modulation(protein_cond).chunk(2,dim=-1)
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
            attention_output_ln = attention_output_ln * (1.0 + scale) + shift

        intermediate_output = self.intermediate(attention_output_ln)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class EsmEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [EsmLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.emb_layer_norm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states,
        protein_cond=None,
        attention_mask=None,
        head_mask=None,
        protein_attention_mask=None,
        past_key_values=None,
        return_dict=True,
    ):
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                protein_cond=protein_cond,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                protein_attention_mask=protein_attention_mask,
                past_key_value=past_key_value,
                output_attentions=False
            )
            hidden_states = layer_outputs[0]
        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
        )


class EsmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ESMModel`]. It is used to instantiate a ESM model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ESM
    [facebook/esm-1b](https://huggingface.co/facebook/esm-1b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*):
            Vocabulary size of the ESM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ESMModel`].
        mask_token_id (`int`, *optional*):
            The index of the mask token in the vocabulary. This must be included in the config because of the
            "mask-dropout" scaling trick, which will scale the inputs depending on the number of masked tokens.
        pad_token_id (`int`, *optional*):
            The index of the padding token in the vocabulary. This must be included in the config because certain parts
            of the ESM code use this instead of the attention mask.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 1026):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query", "rotary"`.
            For positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        emb_layer_norm_before (`bool`, *optional*):
            Whether to apply layer normalization after embeddings but before the main stem of the network.
        token_dropout (`bool`, defaults to `False`):
            When this is enabled, masked tokens are treated as if they had been dropped out by input dropout.

    Examples:

    ```python
    >>> from transformers import EsmModel, EsmConfig

    >>> # Initializing a ESM facebook/esm-1b style configuration >>> configuration = EsmConfig()

    >>> # Initializing a model from the configuration >>> model = ESMModel(configuration)

    >>> # Accessing the model configuration >>> configuration = model.config
    ```"""
    model_type = "esm"


class EsmPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EsmConfig
    base_model_prefix = "esm"
    _no_split_modules = ["EsmLayer", "EsmFoldTriangularSelfAttentionBlock"]

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class EsmModel(EsmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = EsmEmbeddings(config)
        self.encoder = EsmEncoder(config)
        self.post_init()
        self._reset_conditioning_parameters()

    def _reset_conditioning_parameters(self):
        for layer in self.encoder.layer:
            nn.init.zeros_(layer.AdaLN_attention.adaln_modulation.weight)
            nn.init.zeros_(layer.AdaLN_attention.adaln_modulation.bias)

            nn.init.zeros_(layer.ffn_adaln_modulation.weight)
            nn.init.zeros_(layer.ffn_adaln_modulation.bias)

            nn.init.zeros_(layer.film_modulation.weight)
            nn.init.zeros_(layer.film_modulation.bias)

            nn.init.zeros_(layer.gated_bias_proj.weight)
            nn.init.zeros_(layer.gated_bias_proj.bias)
            nn.init.zeros_(layer.gated_bias_gate.weight)
            nn.init.zeros_(layer.gated_bias_gate.bias)

            if hasattr(layer, "protein_conditioning_attention"):
                nn.init.zeros_(layer.protein_conditioning_attention.out_proj.weight)
                nn.init.zeros_(layer.protein_conditioning_attention.out_proj.bias)

    def forward(
        self,
        protein_cond: Optional[torch.Tensor] = None,
        protein_attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[BaseModelOutputWithPastAndCrossAttentions]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length)), device=device
            )

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape
        )
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            protein_cond=protein_cond,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            protein_attention_mask=protein_attention_mask,
        )
        sequence_output = encoder_outputs[0]

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class EsmLMHead(nn.Module):
    """ESM Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias
        return x


class EsmForMaskedLM(EsmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.esm = EsmModel(config)
        self.lm_head = EsmLMHead(config)
        self.lm_head.apply(self._init_weights)
    def forward(
        self,
        protein_cond: Optional[torch.Tensor] = None,
        protein_attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        outputs = self.esm(
            protein_cond=protein_cond,
            protein_attention_mask=protein_attention_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        masked_lm_loss = None
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ==================== Masked diffusion generation ====================
# The following block is broken. Do not use it.
# The following block is broken. Do not use it.
# The following block is broken. Do not use it.
# Repeated on purpose.
'''

def _top_p_logits(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def _top_k_logits(logits, top_k):
    if top_k is None or top_k == 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def _sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, alg="origin"):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = _top_p_logits(logits, top_p)
    if top_k is not None:
        logits = _top_k_logits(logits, top_k)
    probs = torch.softmax(logits.float(), dim=-1)
    if temperature > 0:
        x0 = dists.Categorical(probs=probs).sample()
    else:
        _, x0 = probs.max(dim=-1)
    confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

    if alg == "topk_margin":
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[..., 0] - sorted_probs[..., 1]
    elif alg == "entropy":
        log_probs = torch.log(probs.clamp(min=1e-10))
        confidence = (probs * log_probs).sum(dim=-1)

    return confidence, x0


@dataclass
class DiffusionOutput(ModelOutput):
    """Output type for discrete diffusion generation."""
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.LongTensor]] = None


class MDMGenerationConfig(GenerationConfig):
    """
    Configuration for Masked Diffusion Model generation.

    Args:
        steps (`int`, defaults to 50):
            Number of diffusion denoising steps. More steps generally yields higher quality.
        alg (`str`, defaults to `"random"`):
            Token unmasking order algorithm. One of:
            - `"random"`: randomly select positions to unmask at each step.
            - `"maskgit_plus"`: unmask positions with highest prediction confidence.
            - `"entropy"`: unmask positions with lowest prediction entropy.
            - `"topk_margin"`: unmask positions with highest top-1 vs top-2 probability margin.
            - `"origin"`: stochastically unmask each position with probability proportional to schedule.
            - `"p2"`: progressive prediction with remasking based on confidence.
        temperature (`float`, defaults to 1.0):
            Sampling temperature applied to the model logits. Higher values produce more diverse outputs.
        top_p (`float`, defaults to 0.9):
            Top-p (nucleus) sampling cutoff. Set to 1.0 to disable.
        top_k (`int`, defaults to 0):
            Top-k sampling cutoff. Set to 0 to disable.
        alg_temp (`float`, defaults to 0.9):
            Temperature for the confidence-based unmasking order (Gumbel-TopK). Set to 0 for deterministic order.
        output_history (`bool`, defaults to `False`):
            Whether to return the sequence state at each diffusion step.
    """
    def __init__(self, **kwargs):
        if "do_sample" not in kwargs:
            kwargs["do_sample"] = True
        super().__init__(**kwargs)
        self.temperature: float = kwargs.pop("temperature", 1.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", 0.9)
        self.top_k: Optional[int] = kwargs.pop("top_k", 0)
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 50)
        self.alg: str = kwargs.pop("alg", "random")
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", 0.9)
        self.output_history: bool = kwargs.pop("output_history", False)
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)


class MDMGenerationMixin:
    """Mixin that adds masked diffusion generation to any MaskedLM model."""

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[MDMGenerationConfig] = None,
        **kwargs,
    ) -> Union[DiffusionOutput, torch.LongTensor]:
        """
        Generate RNA sequences using masked diffusion.

        Args:
            inputs (`torch.LongTensor`):
                Input token IDs, typically all `<mask>` tokens representing the desired output length.
                Shape: `(1, sequence_length)`.
            generation_config (`MDMGenerationConfig`, *optional*):
                Generation configuration. If not provided, defaults are used.

        Returns:
            `DiffusionOutput` with `sequences` tensor of shape `(num_return_sequences, sequence_length)`.
        """
        if generation_config is None:
            generation_config = MDMGenerationConfig()
        generation_config.update(**kwargs)

        input_ids = inputs
        attention_mask = kwargs.get("attention_mask", None)

        if input_ids is None:
            raise ValueError("`inputs` must be provided for diffusion generation.")

        if generation_config.max_new_tokens is not None:
            generation_config.max_length = (
                input_ids.shape[-1] + generation_config.max_new_tokens
            )
        elif not hasattr(generation_config, "max_length") or generation_config.max_length is None:
            generation_config.max_length = input_ids.shape[-1]

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        mask_token_id = generation_config.mask_token_id
        if mask_token_id is None:
            raise ValueError("`mask_token_id` must be set in the generation config.")

        x = F.pad(
            input_ids,
            (0, generation_config.max_length - input_ids.shape[1]),
            value=mask_token_id,
        )

        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if generation_config.output_history else None
        fix_mask = (x != mask_token_id)
        gen_attention_mask = (
            (x != self.config.pad_token_id).long()
            if self.config.pad_token_id is not None
            else None
        )
        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        for i in trange(steps, desc="Diffusion"):
            mask_index = (x == mask_token_id)
            if not mask_index.any():
                break

            outputs = self(input_ids=x, attention_mask=gen_attention_mask)
            logits = outputs.logits
            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]

            if alg == "origin":
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = torch.full_like(
                    x[mask_index], fill_value=mask_token_id, device=x.device, dtype=torch.long
                )
                transfer_index = torch.rand(*x0.shape, device=x.device) < p_transfer
                _, sampled = _sample_tokens(
                    mask_logits[transfer_index], temperature=temperature,
                    top_p=top_p, top_k=top_k, alg=alg
                )
                x0[transfer_index] = sampled
                x[mask_index] = x0

            elif alg == "p2":
                kappa_t = (i + 1) / steps
                confidence, x0 = _sample_tokens(
                    mask_logits, temperature, top_p, top_k, alg=alg
                )
                full_conf = torch.full_like(x, float("inf"), dtype=confidence.dtype)
                full_conf[mask_index] = confidence
                full_conf[fix_mask] = float("inf")
                num_positions = (~fix_mask).sum(dim=1, keepdim=True)
                num_to_mask = (num_positions.float() * (1 - kappa_t)).long()
                sorted_idx = torch.argsort(full_conf, dim=-1, descending=False)
                max_mask = num_to_mask.max()
                arange_mask = torch.arange(max_mask, device=x.device).unsqueeze(0) < num_to_mask
                to_mask_idx = sorted_idx[:, :max_mask][arange_mask]
                to_mask = torch.zeros_like(x, dtype=torch.bool)
                batch_idx = (
                    torch.arange(x.size(0), device=x.device)
                    .unsqueeze(1).expand(-1, max_mask)[arange_mask]
                )
                to_mask[batch_idx, to_mask_idx] = True
                x[to_mask] = mask_token_id
                mask_candidates = mask_index & ~to_mask
                x_proposals = torch.full_like(x, fill_value=mask_token_id)
                x_proposals[mask_index] = x0
                x[mask_candidates] = x_proposals[mask_candidates]

            elif alg in ["maskgit_plus", "entropy", "topk_margin"]:
                confidence, x0 = _sample_tokens(
                    mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, alg=alg
                )
                confidence = confidence.to(mask_logits.dtype)
                num_mask_tokens = mask_index.sum(dim=1)
                if i < steps - 1:
                    n_transfer = (num_mask_tokens.float() * (1 - s / t)).long()
                else:
                    n_transfer = num_mask_tokens
                full_confidence = torch.full_like(x, -torch.inf, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                max_transfer = n_transfer.max().item()
                if max_transfer > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, all_indices = torch.topk(full_confidence, max_transfer, dim=1)
                    else:
                        scaled = full_confidence / alg_temp
                        uniform = torch.rand_like(scaled).clamp_(1e-20, 1 - 1e-20)
                        scores = scaled + (-torch.log(-torch.log(uniform)))
                        _, all_indices = torch.topk(scores, max_transfer, dim=1)
                    valid_mask = (
                        torch.arange(max_transfer, device=x.device).unsqueeze(0)
                        < n_transfer.unsqueeze(1)
                    )
                    valid_indices = all_indices[valid_mask]
                    valid_batch = (
                        torch.arange(x.size(0), device=x.device)
                        .unsqueeze(1).expand_as(all_indices)[valid_mask]
                    )
                    x_ = torch.full_like(x, fill_value=mask_token_id)
                    x_[mask_index] = x0.clone()
                    x[valid_batch, valid_indices] = x_[valid_batch, valid_indices]

            elif alg == "random":
                _, x0 = _sample_tokens(
                    mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, alg=alg
                )
                num_mask_tokens = mask_index.sum(dim=1)
                if i < steps - 1:
                    n_transfer = (num_mask_tokens.float() * (1 - s / t)).long()
                else:
                    n_transfer = num_mask_tokens
                max_transfer = n_transfer.max().item()
                if max_transfer > 0:
                    x_ = torch.full_like(x, fill_value=mask_token_id)
                    x_[mask_index] = x0.clone()
                    for b in range(x.size(0)):
                        positions = mask_index[b].nonzero(as_tuple=True)[0]
                        n = n_transfer[b].item()
                        if len(positions) > 0 and n > 0:
                            n = min(n, len(positions))
                            sel = torch.randperm(len(positions), device=x.device)[:n]
                            x[b, positions[sel]] = x_[b, positions[sel]]
            else:
                raise NotImplementedError(f"Algorithm '{alg}' is not implemented.")

            if histories is not None:
                histories.append(x.clone())

        if generation_config.return_dict_in_generate:
            return DiffusionOutput(sequences=x, history=histories)
        return x


class DiffusionForMaskedLM(EsmForMaskedLM, MDMGenerationMixin):
    """
    Discrete diffusion language model on RNA tokens (ESM backbone).

    Subclass of `EsmForMaskedLM` with `diffusion_generate()` for iterative unmasking.
    `diffusion_generate()` samples RNA; `forward()` is standard MLM.
    """

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

'''
