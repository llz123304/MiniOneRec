"""Tests for decoder position-specific projections."""

import unittest

import torch
import torch.nn as nn

from lazy_onerec.model.configuration import LazyOneRecConfig
from lazy_onerec.model.modeling import (
    LazyOneRecForCausalLM,
    PerTokenLinear,
    PerTokenSwiGLU,
)


class PerTokenDecoderTest(unittest.TestCase):
    def test_linear_full_sequence_matches_incremental_steps(self):
        torch.manual_seed(0)
        inputs = torch.randn(2, 4, 8)
        projection = PerTokenLinear(4, 8, 6)

        full = projection(inputs)
        incremental = torch.cat(
            [
                projection(inputs[:, position : position + 1], position)
                for position in range(4)
            ],
            dim=1,
        )

        self.assertTrue(torch.allclose(full, incremental, atol=1e-6))

    def test_swiglu_full_sequence_matches_incremental_steps(self):
        torch.manual_seed(1)
        inputs = torch.randn(2, 4, 8)
        ffn = PerTokenSwiGLU(
            d_model=8,
            d_ff=16,
            max_positions=4,
        )

        full = ffn(inputs)
        incremental = torch.cat(
            [
                ffn(inputs[:, position : position + 1], position)
                for position in range(4)
            ],
            dim=1,
        )

        self.assertTrue(torch.allclose(full, incremental, atol=1e-6))

    def test_decoder_defaults_to_per_token_modules(self):
        config = LazyOneRecConfig(
            codebook_sizes=[8, 8, 8],
            d_model=32,
            d_ff=64,
            n_layers=1,
            n_context_layers=1,
            n_heads=4,
            n_kv_heads=2,
        )
        model = LazyOneRecForCausalLM(config)
        layer = model.layers[0]

        self.assertIsInstance(layer.self_attn.q_proj, PerTokenLinear)
        self.assertIsInstance(layer.self_attn.k_proj, PerTokenLinear)
        self.assertIsInstance(layer.self_attn.v_proj, PerTokenLinear)
        self.assertIsInstance(layer.cross_attn.q_proj, PerTokenLinear)
        self.assertIsInstance(layer.ffn, PerTokenSwiGLU)

        shared_config = LazyOneRecConfig(
            codebook_sizes=[8, 8, 8],
            d_model=32,
            d_ff=64,
            n_layers=1,
            n_context_layers=1,
            n_heads=4,
            n_kv_heads=2,
            use_per_token_qkv=False,
            use_per_token_ffn=False,
        )
        shared_model = LazyOneRecForCausalLM(shared_config)
        shared_layer = shared_model.layers[0]
        self.assertIsInstance(shared_layer.self_attn.q_proj, nn.Linear)
        self.assertNotIsInstance(shared_layer.ffn, PerTokenSwiGLU)

    def test_decoder_kv_cache_matches_full_sequence(self):
        torch.manual_seed(2)
        config = LazyOneRecConfig(
            codebook_sizes=[8, 8, 8],
            d_model=32,
            d_ff=64,
            n_layers=2,
            n_context_layers=1,
            n_heads=4,
            n_kv_heads=2,
        )
        model = LazyOneRecForCausalLM(config).eval()
        context = torch.randn(2, 5, config.d_model)
        context_mask = torch.ones(2, 5, dtype=torch.long)
        target = torch.tensor(
            [
                [config.bos_token_id, 3, 12],
                [config.bos_token_id, 4, 13],
            ],
            dtype=torch.long,
        )

        with torch.inference_mode():
            full = model(
                context_inputs_embeds=context,
                context_attention_mask=context_mask,
                target_input_ids=target,
            )
            context_kv_blocks, encoded_mask = model.context_processor(
                context,
                context_mask,
            )
            past_key_values = None
            incremental_logits = []
            for position in range(target.size(1)):
                step = model(
                    context_kv_blocks=context_kv_blocks,
                    context_attention_mask=encoded_mask,
                    target_input_ids=target[:, position : position + 1],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                incremental_logits.append(step.logits[:, -1])
                past_key_values = step.past_key_values

        incremental = torch.stack(incremental_logits, dim=1)
        self.assertTrue(torch.allclose(full.logits, incremental, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
