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

import pytest
import torch
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache

pytest.importorskip("timm")

from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.models.nemotron_parse.prefix_model import (
    PATCH_DIM,
    PATCH_SIZE,
    NemotronParsePrefixForCausalLM,
    content_normalized_positions,
)
from nemo_automodel.components.utils.model_utils import apply_parameter_freezing

VOCAB = 120


def tiny_config(num_extra_heads: int = 1) -> PretrainedConfig:
    decoder = PretrainedConfig(
        d_model=32,
        decoder_layers=2,
        decoder_attention_heads=4,
        self_attention_kv_heads=2,
        decoder_ffn_dim=64,
        vocab_size=VOCAB,
        pad_token_id=1,
        scale_embedding=True,
    )
    return PretrainedConfig(
        decoder=decoder,
        legacy_causal_vision_prefix=True,
        num_extra_heads=num_extra_heads,
        max_sequence_length=4096,
        tie_word_embeddings=True,
        vision_spec={"width": 32, "depth": 2, "heads": 4, "mlp_width": 64, "registers": 2, "position_grid": 8},
    )


def tiny_model(mtp_loss_weight: float = 0.0) -> NemotronParsePrefixForCausalLM:
    torch.manual_seed(0)
    model = NemotronParsePrefixForCausalLM(tiny_config(), mtp_loss_weight=mtp_loss_weight)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.05)
    return model.eval()


def packed_pages(sizes: list[tuple[int, int]]) -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    count = sum((height // PATCH_SIZE) * (width // PATCH_SIZE) for height, width in sizes)
    return {
        "pixel_values": torch.randn(count, PATCH_DIM),
        "image_sizes": torch.tensor(sizes),
        "content_sizes": torch.tensor([(height - 8, width - 8) for height, width in sizes]),
    }


def pages_subset(pages: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    counts = [(int(h) // PATCH_SIZE) * (int(w) // PATCH_SIZE) for h, w in pages["image_sizes"].tolist()]
    start = sum(counts[:index])
    return {
        "pixel_values": pages["pixel_values"][start : start + counts[index]],
        "image_sizes": pages["image_sizes"][index : index + 1],
        "content_sizes": pages["content_sizes"][index : index + 1],
    }


def test_architecture_is_registered():
    assert ModelRegistry.has_custom_model("RadioMBartPrefixForCausalLM")


def test_rejects_encoder_decoder_config():
    config = tiny_config()
    config.legacy_causal_vision_prefix = False
    with pytest.raises(ValueError, match="causal visual-prefix"):
        NemotronParsePrefixForCausalLM(config)


def test_lm_head_is_tied_to_embeddings():
    model = tiny_model()
    assert model.lm_head.weight is model.decoder.embed_tokens.weight


def test_logits_cover_text_positions_only():
    model = tiny_model()
    pages = packed_pages([(64, 64), (32, 128)])
    input_ids = torch.randint(3, VOCAB, (2, 7))
    output = model(input_ids=input_ids, **pages)
    assert output.logits.shape == (2, 7, VOCAB)


def test_batch_padding_does_not_change_logits():
    model = tiny_model()
    pages = packed_pages([(64, 64), (32, 128)])
    input_ids = torch.randint(3, VOCAB, (2, 7))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, 5:] = 0
    batched = model(input_ids=input_ids, attention_mask=attention_mask, **pages).logits
    for row, length in ((0, 7), (1, 5)):
        alone = model(input_ids=input_ids[row : row + 1, :length], **pages_subset(pages, row)).logits
        torch.testing.assert_close(batched[row, :length], alone[0], atol=1e-5, rtol=1e-4)


def test_decoder_is_causal():
    model = tiny_model()
    pages = packed_pages([(64, 64)])
    input_ids = torch.randint(3, VOCAB, (1, 6))
    changed = input_ids.clone()
    changed[0, 4] = (changed[0, 4] + 1) % VOCAB
    before = model(input_ids=input_ids, **pages).logits
    after = model(input_ids=changed, **pages).logits
    torch.testing.assert_close(before[:, :4], after[:, :4])
    assert not torch.allclose(before[:, 4:], after[:, 4:])


def test_cached_decoding_matches_full_forward():
    model = tiny_model()
    pages = packed_pages([(64, 64)])
    input_ids = torch.randint(3, VOCAB, (1, 6))
    full = model(input_ids=input_ids, use_cache=False, **pages).logits
    cache = DynamicCache()
    steps = [model(input_ids=input_ids[:, :3], past_key_values=cache, use_cache=True, **pages).logits]
    for position in range(3, 6):
        steps.append(
            model(
                input_ids=input_ids[:, position : position + 1],
                attention_mask=torch.ones(1, position + 1, dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
            ).logits
        )
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-5, rtol=1e-4)


def test_training_forward_builds_no_cache():
    model = tiny_model().train()
    pages = packed_pages([(64, 64)])
    output = model(input_ids=torch.randint(3, VOCAB, (1, 5)), **pages)
    assert output.past_key_values is None


def test_mtp_drafter_logits_only_when_weighted_and_training():
    pages = packed_pages([(64, 64)])
    input_ids = torch.randint(3, VOCAB, (1, 5))
    assert tiny_model().train()(input_ids=input_ids, **pages).drafter_logits is None
    assert tiny_model(0.3)(input_ids=input_ids, **pages).drafter_logits is None
    output = tiny_model(0.3).train()(input_ids=input_ids, **pages)
    assert output.drafter_loss_weight == 0.3
    assert output.drafter_logits[0].shape == output.logits.shape


def test_mtp_reads_previous_hidden_and_current_token():
    model = tiny_model(0.3)
    hidden = torch.randn(1, 4, 32)
    input_ids = torch.randint(3, VOCAB, (1, 4))
    logits = model.mtp_logits(hidden, input_ids)
    decoder = model.decoder
    expected = model.lm_head(
        decoder.extra_heads[0](hidden[:, 1] + decoder.extra_proj[0](decoder.embed_tokens(input_ids[:, 2])))
    )
    torch.testing.assert_close(logits[:, 2], expected)


def test_mtp_weight_needs_an_extra_head():
    with pytest.raises(ValueError, match="num_extra_heads"):
        NemotronParsePrefixForCausalLM(tiny_config(num_extra_heads=0), mtp_loss_weight=0.3)


def test_freeze_vision_tower_freezes_only_the_encoder():
    model = tiny_model()
    apply_parameter_freezing(model, {"freeze_vision_tower": True, "freeze_language_model": False})
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.decoder.parameters())


def test_positions_reuse_border_on_padding():
    pos_embed = torch.randn(1, 64, 4)
    positions = content_normalized_positions(pos_embed, rows=2, cols=4, content_height=16, content_width=32)
    assert positions.shape == (1, 8, 4)
    torch.testing.assert_close(positions[0, 4:], positions[0, :4])


def test_positions_reject_content_larger_than_page():
    with pytest.raises(ValueError, match="exceeds"):
        content_normalized_positions(torch.randn(1, 64, 4), rows=1, cols=1, content_height=32, content_width=16)
