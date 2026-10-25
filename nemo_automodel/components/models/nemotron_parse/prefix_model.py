# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainable Nemotron 3.5 Parse (``RadioMBartPrefixForCausalLM``).

Nemotron 3.5 Parse is a decoder-only model: C-RADIOv4-SO400M features of each
NaViT-packed page form a causal prefix in front of the text tokens of a
10-layer GQA mBART decoder. Parameter names match the released checkpoint.
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)

PATCH_SIZE = 16
PATCH_DIM = 3 * PATCH_SIZE * PATCH_SIZE


@dataclass(frozen=True)
class VisionSpec:
    """Shape of the RADIO vision transformer; defaults are C-RADIOv4-SO400M."""

    width: int = 1152
    depth: int = 27
    heads: int = 16
    mlp_width: int = 4304
    registers: int = 10
    position_grid: int = 128
    summary_tokens: int = 2


def content_normalized_positions(
    pos_embed: torch.Tensor, rows: int, cols: int, content_height: int, content_width: int
) -> torch.Tensor:
    """Sample the learned square position field for one padded page.

    The field is resized to the page's content area and then read at each
    patch centre in content-normalized coordinates, so padding patches reuse
    the border positions. Mirrors the released serving geometry.
    """
    if content_height <= 0 or content_width <= 0:
        raise ValueError(f"content size must be positive, got {(content_height, content_width)}")
    if content_height > rows * PATCH_SIZE or content_width > cols * PATCH_SIZE:
        raise ValueError(f"content {(content_height, content_width)} exceeds the padded page {(rows, cols)} patches")
    side = math.isqrt(pos_embed.shape[1])
    grid = pos_embed.reshape(1, side, side, -1).permute(0, 3, 1, 2).float()
    content_rows = math.ceil(content_height / PATCH_SIZE)
    content_cols = math.ceil(content_width / PATCH_SIZE)
    if min(content_rows, content_cols) == 1:
        field = F.adaptive_avg_pool2d(grid, (content_rows, content_cols))
    else:
        field = F.interpolate(
            grid, size=(content_rows, content_cols), mode="bilinear", align_corners=False, antialias=True
        )
    device = pos_embed.device
    row_starts = torch.arange(rows, device=device, dtype=torch.float32) * PATCH_SIZE
    col_starts = torch.arange(cols, device=device, dtype=torch.float32) * PATCH_SIZE
    row_extents = (content_height - row_starts).clamp(min=0.0, max=float(PATCH_SIZE))
    col_extents = (content_width - col_starts).clamp(min=0.0, max=float(PATCH_SIZE))
    normalized_y = (row_starts + 0.5 * row_extents).clamp(max=float(content_height)) / float(content_height)
    normalized_x = (col_starts + 0.5 * col_extents).clamp(max=float(content_width)) / float(content_width)
    grid_y, grid_x = torch.meshgrid(normalized_y, normalized_x, indexing="ij")
    sampling = torch.stack((2.0 * grid_x - 1.0, 2.0 * grid_y - 1.0), dim=-1).unsqueeze(0)
    sampled = F.grid_sample(field, sampling, mode="bilinear", padding_mode="border", align_corners=False)
    return sampled.flatten(2).transpose(1, 2).to(pos_embed.dtype)


class ClassTokens(nn.Module):
    """Learned RADIO summary and register tokens prepended to the patch sequence."""

    def __init__(self, spec: VisionSpec) -> None:
        super().__init__()
        self.token = nn.Parameter(torch.empty(spec.registers, spec.width))


class PatchGenerator(nn.Module):
    """Patch embedding, class tokens and the learned 2D position field."""

    def __init__(self, spec: VisionSpec) -> None:
        super().__init__()
        self.cls_token = ClassTokens(spec)
        self.embedder = nn.Linear(PATCH_DIM, spec.width, bias=False)
        self.pos_embed = nn.Parameter(torch.empty(1, spec.position_grid * spec.position_grid, spec.width))


class VisionTransformer(nn.Module):
    """RADIO ViT trunk run over one page at a time."""

    def __init__(self, spec: VisionSpec) -> None:
        super().__init__()
        from timm.models.vision_transformer import Block

        self.patch_generator = PatchGenerator(spec)
        self.blocks = nn.ModuleList(
            [
                Block(
                    spec.width,
                    spec.heads,
                    mlp_ratio=spec.mlp_width / spec.width,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                )
                for _ in range(spec.depth)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self, patches: torch.Tensor, rows: int, cols: int, content_height: int, content_width: int
    ) -> torch.Tensor:
        generator = self.patch_generator
        hidden = generator.embedder(patches).unsqueeze(0)
        hidden = hidden + content_normalized_positions(generator.pos_embed, rows, cols, content_height, content_width)
        hidden = torch.cat((generator.cls_token.token.unsqueeze(0), hidden), dim=1)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(block, hidden, use_reentrant=False)
                continue
            hidden = block(hidden)
        return hidden


class InputConditioner(nn.Module):
    """Pixel normalization buffers kept for checkpoint compatibility; the processor normalizes."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("norm_mean", torch.zeros(3, 1, 1))
        self.register_buffer("norm_std", torch.ones(3, 1, 1))


class RadioModel(nn.Module):
    """RADIO model holder matching the released checkpoint layout."""

    def __init__(self, spec: VisionSpec) -> None:
        super().__init__()
        self.register_buffer("summary_idxs", torch.arange(spec.summary_tokens))
        self.input_conditioner = InputConditioner()
        self.model = VisionTransformer(spec)


class RadioWrapper(nn.Module):
    """Outer RADIO wrapper matching the released checkpoint layout."""

    def __init__(self, spec: VisionSpec) -> None:
        super().__init__()
        self.radio_model = RadioModel(spec)


class PrefixVisionEncoder(nn.Module):
    """C-RADIOv4-SO400M plus the neck that turns each page into decoder prefix tokens."""

    def __init__(self, d_model: int, spec: VisionSpec) -> None:
        super().__init__()
        self.model_encoder = RadioWrapper(spec)
        self.conv1 = nn.Conv1d(spec.width, d_model, 1)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.conv2 = nn.Conv2d(d_model, d_model, (1, 4), stride=(1, 4), bias=False)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.sum_proj = nn.Linear(spec.summary_tokens * spec.width, d_model)
        self.layer_norm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model
        self.registers = spec.registers

    def encode_page(
        self, patches: torch.Tensor, rows: int, cols: int, content_height: int, content_width: int
    ) -> torch.Tensor:
        radio = self.model_encoder.radio_model
        hidden = radio.model(patches, rows, cols, content_height, content_width)
        summary = hidden[:, radio.summary_idxs].flatten(1)
        features = hidden[:, self.registers :]
        features = self.layer_norm1(self.conv1(features.transpose(1, 2)).transpose(1, 2))
        features = features.transpose(1, 2).reshape(1, self.d_model, rows, cols)
        features = self.layer_norm2(self.conv2(features).flatten(2).transpose(1, 2))
        summary = self.layer_norm3(self.sum_proj(summary)).unsqueeze(1)
        return torch.cat((features, summary), dim=1)[0]

    def forward(
        self, pixel_values: torch.Tensor, image_sizes: torch.Tensor, content_sizes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode packed raw patches ``[patches, 768]`` into right-padded prefixes and their mask."""
        if pixel_values.ndim == 3 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values[0]
        if pixel_values.ndim != 2 or pixel_values.shape[1] != PATCH_DIM:
            raise ValueError(f"expected packed raw patches [patches, {PATCH_DIM}], got {tuple(pixel_values.shape)}")
        pixel_values = pixel_values.to(self.conv1.weight.dtype)
        pages = []
        cursor = 0
        for (height, width), (content_height, content_width) in zip(image_sizes.tolist(), content_sizes.tolist()):
            rows, cols = int(height) // PATCH_SIZE, int(width) // PATCH_SIZE
            count = rows * cols
            pages.append(
                self.encode_page(
                    pixel_values[cursor : cursor + count], rows, cols, int(content_height), int(content_width)
                )
            )
            cursor += count
        if cursor != len(pixel_values):
            raise ValueError(f"image_sizes consumed {cursor} of {len(pixel_values)} packed patches")
        longest = max(len(page) for page in pages)
        prefix = pages[0].new_zeros(len(pages), longest, self.d_model)
        mask = torch.zeros(len(pages), longest, device=prefix.device, dtype=torch.bool)
        for index, page in enumerate(pages):
            prefix[index, : len(page)] = page
            mask[index, : len(page)] = True
        return prefix, mask


class ScaledEmbedding(nn.Embedding):
    """Token embedding multiplied by ``sqrt(d_model)`` as in mBART."""

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int, embed_scale: float) -> None:
        super().__init__(vocab_size, d_model, padding_idx=padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale


class GQAAttention(nn.Module):
    """Causal grouped-query self-attention over the visual prefix and text."""

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, layer_index: int) -> None:
        super().__init__()
        if d_model % num_heads or num_heads % num_kv_heads:
            raise ValueError(f"d_model={d_model}, heads={num_heads}, kv_heads={num_kv_heads} do not divide")
        self.layer_index = layer_index
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, hidden: torch.Tensor, key_mask: torch.Tensor, cache: Cache | None) -> torch.Tensor:
        batch, length, _ = hidden.shape
        query = self.q_proj(hidden).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if cache is not None:
            key, value = cache.update(key, value, self.layer_index)
        past = key.shape[-2] - length
        positions = torch.arange(key.shape[-2], device=hidden.device)
        causal = positions[None, :] <= positions[past:, None]
        allowed = causal[None, None] & key_mask[:, None, None, :]
        repeat = self.num_heads // self.num_kv_heads
        output = F.scaled_dot_product_attention(
            query, key.repeat_interleave(repeat, dim=1), value.repeat_interleave(repeat, dim=1), attn_mask=allowed
        )
        return self.out_proj(output.transpose(1, 2).reshape(batch, length, -1))


class DecoderLayer(nn.Module):
    """Pre-norm mBART decoder layer without cross-attention."""

    def __init__(self, d_model: int, ffn_dim: int, num_heads: int, num_kv_heads: int, layer_index: int) -> None:
        super().__init__()
        self.self_attn = GQAAttention(d_model, num_heads, num_kv_heads, layer_index)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)

    def forward(self, hidden: torch.Tensor, key_mask: torch.Tensor, cache: Cache | None) -> torch.Tensor:
        hidden = hidden + self.self_attn(self.self_attn_layer_norm(hidden), key_mask, cache)
        return hidden + self.fc2(F.gelu(self.fc1(self.final_layer_norm(hidden))))


class PrefixDecoder(nn.Module):
    """mBART decoder stack plus the bundled MTP projection heads."""

    def __init__(self, config: PretrainedConfig, num_extra_heads: int) -> None:
        super().__init__()
        d_model = config.d_model
        embed_scale = math.sqrt(d_model) if config.scale_embedding else 1.0
        self.embed_tokens = ScaledEmbedding(config.vocab_size, d_model, config.pad_token_id, embed_scale)
        num_kv_heads = getattr(config, "self_attention_kv_heads", None) or config.decoder_attention_heads
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, config.decoder_ffn_dim, config.decoder_attention_heads, num_kv_heads, index)
                for index in range(config.decoder_layers)
            ]
        )
        self.layernorm_embedding = nn.LayerNorm(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.extra_heads = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_extra_heads)])
        self.extra_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_extra_heads)])
        self.gradient_checkpointing = False

    def forward(self, embeddings: torch.Tensor, key_mask: torch.Tensor, cache: Cache | None) -> torch.Tensor:
        hidden = self.layernorm_embedding(embeddings)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and cache is None:
                hidden = checkpoint(layer, hidden, key_mask, None, use_reentrant=False)
                continue
            hidden = layer(hidden, key_mask, cache)
        return self.layer_norm(hidden)


@dataclass
class NemotronParsePrefixOutput(CausalLMOutputWithPast):
    """Causal LM output plus the one-step MTP logits the VLM recipe trains as a drafter."""

    drafter_logits: list[torch.Tensor] | None = None
    drafter_loss_weight: float | None = None


class NemotronParsePrefixForCausalLM(HFCheckpointingMixin, PreTrainedModel, GenerationMixin):
    """Nemotron 3.5 Parse with a training forward.

    ``labels`` are not consumed here: the VLM recipe pops them and applies its
    configured loss to ``logits``, which cover the text positions only
    (``logits[:, t]`` predicts ``input_ids[:, t + 1]``). With ``mtp_loss_weight > 0``
    the bundled one-step MTP head is trained alongside, so vLLM speculative
    decoding stays effective after fine-tuning.
    """

    config_class = PretrainedConfig
    base_model_prefix = ""
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _no_split_modules = ["DecoderLayer"]
    _tied_weights_keys = {"lm_head.weight": "decoder.embed_tokens.weight"}
    tie_word_embeddings_support: TieSupport = TieSupport.TIED_ONLY

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False

    def __init__(self, config: PretrainedConfig, mtp_loss_weight: float = 0.0, **kwargs: Any) -> None:
        reject_unsupported_tie_word_embeddings(type(self), config)
        if not getattr(config, "legacy_causal_vision_prefix", False):
            raise ValueError("NemotronParsePrefixForCausalLM needs a causal visual-prefix (Nemotron 3.5 Parse) config")
        config.is_encoder_decoder = False
        super().__init__(config)
        decoder_config = config.decoder
        self.encoder = PrefixVisionEncoder(
            decoder_config.d_model, VisionSpec(**(getattr(config, "vision_spec", None) or {}))
        )
        self.decoder = PrefixDecoder(decoder_config, getattr(config, "num_extra_heads", 0))
        self.lm_head = nn.Linear(decoder_config.d_model, decoder_config.vocab_size, bias=False)
        self.mtp_loss_weight = mtp_loss_weight
        if mtp_loss_weight and not self.decoder.extra_heads:
            raise ValueError("mtp_loss_weight > 0 needs num_extra_heads >= 1 in the config")
        self.post_init()
        self.tie_weights()

    @property
    def vision_tower(self) -> PrefixVisionEncoder:
        return self.encoder

    def tie_weights(self, *args: object, **kwargs: object) -> None:
        self.lm_head.weight = self.decoder.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Module:
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.decoder.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.lm_head = value

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict | None = None) -> None:
        self.decoder.gradient_checkpointing = True
        self.encoder.model_encoder.radio_model.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.decoder.gradient_checkpointing = False
        self.encoder.model_encoder.radio_model.model.gradient_checkpointing = False

    def mtp_logits(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Logits of the MTP head aligned to ``labels``: position ``t`` reads ``h[t-1]`` and ``x[t]``."""
        previous = torch.cat((hidden.new_zeros(hidden[:, :1].shape), hidden[:, :-1]), dim=1)
        draft = self.decoder.extra_heads[0](previous + self.decoder.extra_proj[0](self.decoder.embed_tokens(input_ids)))
        return self.lm_head(draft)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_sizes: torch.Tensor | None = None,
        content_sizes: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> NemotronParsePrefixOutput:
        if input_ids is None:
            input_ids = decoder_input_ids
        if input_ids is None:
            raise ValueError("input_ids are required")
        if use_cache is None:
            use_cache = not self.training
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        cache = past_key_values if use_cache else None
        past_length = cache.get_seq_length() if cache is not None else 0
        text = self.decoder.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if past_length == 0:
            if pixel_values is None or image_sizes is None or content_sizes is None:
                raise ValueError("pixel_values, image_sizes and content_sizes are required for the prefix step")
            prefix, prefix_mask = self.encoder(pixel_values, image_sizes, content_sizes)
            embeddings = torch.cat((prefix.to(text.dtype), text), dim=1)
            text_offset = prefix.shape[1]
            key_mask = torch.cat((prefix_mask, attention_mask.bool()), dim=1)
            if cache is not None:
                cache.nemotron_key_mask = key_mask
        else:
            embeddings = text
            text_offset = 0
            key_mask = torch.cat((cache.nemotron_key_mask, attention_mask[:, -input_ids.shape[1] :].bool()), dim=1)
            cache.nemotron_key_mask = key_mask
        if key_mask.shape[1] > self.config.max_sequence_length:
            raise ValueError(f"prefix plus text is {key_mask.shape[1]} tokens, over {self.config.max_sequence_length}")
        hidden = self.decoder(embeddings, key_mask, cache)[:, text_offset:]
        kept = hidden if not logits_to_keep else hidden[:, -logits_to_keep:]
        output = NemotronParsePrefixOutput(
            logits=self.lm_head(kept),
            past_key_values=cache,
            hidden_states=hidden if output_hidden_states else None,
        )
        if self.training and self.mtp_loss_weight:
            output.drafter_logits = [self.mtp_logits(hidden, input_ids)]
            output.drafter_loss_weight = self.mtp_loss_weight
        return output

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_sizes: torch.Tensor | None = None,
        content_sizes: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            input_ids = input_ids[:, -1:]
            pixel_values = None
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "content_sizes": content_sizes,
            "use_cache": True,
        }


ModelClass = NemotronParsePrefixForCausalLM
