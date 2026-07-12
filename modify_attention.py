import math
import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def llama_new_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )

    # Vision-attention modifications are applied to the pre-softmax logits of
    # the newest token only.  This is the same location used by the paper.
    if hasattr(self, "aggregation"):
        img_start_idx = self.img_start_idx
        img_end_idx = self.img_end_idx
        aggregation = self.aggregation

        if aggregation == "mean":
            attn_weights[:, :, -1, img_start_idx:img_end_idx] = (
                attn_weights[:, :, -1, img_start_idx:img_end_idx]
                + self.alpha * attn_weights[:, :, -1, img_start_idx:img_end_idx].abs().mean(dim=1, keepdim=True)
            )

    if hasattr(self, "foreground_mask"):
        img_start_idx = self.img_start_idx
        img_end_idx = self.img_end_idx
        visual_scores = attn_weights[:, :, -1, img_start_idx:img_end_idx]
        mask = self.foreground_mask.to(device=visual_scores.device, dtype=visual_scores.dtype)
        if mask.ndim == 1:
            mask = mask.view(1, 1, -1)
        if mask.shape[-1] != visual_scores.shape[-1]:
            raise ValueError(
                "foreground mask length must equal the number of image tokens "
                f"({mask.shape[-1]} != {visual_scores.shape[-1]})"
            )
        # The scale is deliberately identical to Heads Guided Attention.  The
        # per-token boost is ``base + alpha * mask``: ``base`` enriches every
        # visual token (as head-guide does) and ``alpha`` adds extra weight on
        # foreground patches.  base=0 recovers pure foreground-only guidance;
        # alpha=0 recovers plain head-guide.
        shared_score_scale = visual_scores.abs().mean(dim=1, keepdim=True)
        boost = self.foreground_base + self.foreground_alpha * mask
        attn_weights[:, :, -1, img_start_idx:img_end_idx] = (
            visual_scores + shared_score_scale * boost
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_head_guide(model, guided_layer_range, aggregation, alpha, img_start_idx, img_end_idx):
    layer_list = guided_layer_range if len(guided_layer_range) == 1 else list(range(guided_layer_range[0], guided_layer_range[1]))

    for i in layer_list:
        model.model.layers[i].self_attn.img_start_idx = img_start_idx
        model.model.layers[i].self_attn.img_end_idx = img_end_idx
        model.model.layers[i].self_attn.aggregation = aggregation
        model.model.layers[i].self_attn.alpha = alpha
        model.model.layers[i].self_attn.forward = types.MethodType(llama_new_forward, model.model.layers[i].self_attn)


def llama_foreground_guide(
    model, guided_layer_range, foreground_mask, alpha, img_start_idx, img_end_idx, base=0.0
):
    """Boost visual patches with a ``base + alpha * mask`` per-token schedule.

    ``foreground_mask`` is a length-576, row-major 24x24 CLIP-patch mask with
    values in [0, 1].  It is an oracle intervention when sourced from COCO GT.

    ``base`` is applied to every visual token (the head-guide enrichment) and
    ``alpha`` adds the extra foreground weight.  ``base=0`` gives pure
    foreground-only guidance; ``alpha=0`` reproduces plain head-guide.
    """
    layer_list = (
        guided_layer_range
        if len(guided_layer_range) == 1
        else list(range(guided_layer_range[0], guided_layer_range[1]))
    )
    for i in layer_list:
        attn = model.model.layers[i].self_attn
        attn.img_start_idx = img_start_idx
        attn.img_end_idx = img_end_idx
        attn.foreground_mask = foreground_mask
        attn.foreground_alpha = alpha
        attn.foreground_base = base
        attn.forward = types.MethodType(llama_new_forward, attn)
