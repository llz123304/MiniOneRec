"""Tests for native SID generation metrics."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from lazy_onerec.data.schema import (
    GID_SEQUENCE_NAMES,
    N_LONG_VIEW_DURATION_BUCKETS,
)
from lazy_onerec.data.user_features import CATEGORICAL_USER_FIELDS
from lazy_onerec.model.configuration import LazyOneRecConfig
from lazy_onerec.model.inference import sid_beam_search
from lazy_onerec.model.kuairand import KuaiRandLazyOneRecForCausalLM
from lazy_onerec.sid.evaluation import SidRankingMetrics, SidValidityIndex
from lazy_onerec.sid.layout import N_SPECIAL
from lazy_onerec.sid.token_codec import SidTokenCodec
from lazy_onerec.src.evaluate_kuairand import validate_checkpoint_dataset


class _FakeGenerationModel:
    def __init__(self, codebook_sizes):
        self.config = SimpleNamespace(codebook_sizes=list(codebook_sizes))
        self.vocab_size = N_SPECIAL + sum(codebook_sizes)

    def __call__(self, target_input_ids, **_context):
        batch_size, target_length = target_input_ids.shape
        logits = torch.full(
            (batch_size, target_length, self.vocab_size),
            -100.0,
        )
        logits[:, :, :] = torch.arange(
            self.vocab_size,
            dtype=torch.float32,
        )
        return SimpleNamespace(logits=logits)


class _FakeCachedGenerationModel(_FakeGenerationModel):
    def __init__(self, codebook_sizes):
        super().__init__(codebook_sizes)
        self.encode_calls = 0
        self.decode_input_lengths = []

    def encode_context(self, context_inputs):
        self.encode_calls += 1
        batch_size = next(iter(context_inputs.values())).size(0)
        key = torch.zeros(batch_size, 1, 1, 1)
        value = torch.zeros_like(key)
        mask = torch.ones(batch_size, 1)
        return ((key, value),), mask

    def decode_with_context(
        self,
        target_input_ids,
        context_kv_blocks,
        context_attention_mask,
        past_key_values=None,
        use_cache=True,
    ):
        if not use_cache:
            raise AssertionError("cached fake expects use_cache=True")
        if context_attention_mask.size(0) != target_input_ids.size(0):
            raise AssertionError("context mask batch does not match target")
        if context_kv_blocks[0][0].size(0) != target_input_ids.size(0):
            raise AssertionError("context KV batch does not match target")
        self.decode_input_lengths.append(target_input_ids.size(1))
        outputs = self(target_input_ids)
        past_length = (
            0
            if past_key_values is None
            else past_key_values[0][0].size(2)
        )
        present_length = past_length + target_input_ids.size(1)
        batch_size = target_input_ids.size(0)
        key = torch.zeros(batch_size, 1, present_length, 1)
        value = torch.zeros_like(key)
        outputs.past_key_values = ((key, value),)
        return outputs


class SidEvaluationTest(unittest.TestCase):
    def test_dataset_gid_vocabulary_must_fit_checkpoint(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                user_categorical_cardinalities=[4, 5],
            ),
            context_feature_embedding=SimpleNamespace(
                gid_embedding=SimpleNamespace(num_embeddings=10),
            ),
        )
        corpus = SimpleNamespace(
            num_gid_embeddings=11,
            user_features=SimpleNamespace(
                categorical_cardinalities=(4, 5),
            ),
        )

        with self.assertRaisesRegex(ValueError, "checkpoint provides 10"):
            validate_checkpoint_dataset(model, corpus)

    def test_hr_mrr_level_accuracy_and_invalid_rate(self):
        targets = np.asarray([[1, 2, 3], [0, 1, 2]])
        predictions = np.zeros((2, 10, 3), dtype=np.int64)
        predictions[0] = np.asarray(
            [[1, 2, 4], [1, 2, 3]] + [[2, index, 0] for index in range(8)]
        )
        predictions[1] = np.asarray(
            [[0, 1, 1]] + [[3, index, 0] for index in range(9)]
        )
        predictions[1, 9] = [9, 9, 9]

        valid_rows = [
            tuple(row)
            for sample in predictions
            for row in sample
            if tuple(row) != (9, 9, 9)
        ]
        valid_rows.extend(tuple(row) for row in targets)
        validity_index = SidValidityIndex(valid_rows, (10, 10, 10))
        metrics = SidRankingMetrics(validity_index)

        metrics.update(predictions, targets)

        self.assertEqual(
            metrics.compute(),
            {
                "sid0_hr_at_10": 1.0,
                "sid0_mrr_at_10": 1.0,
                "sid1_hr_at_10": 1.0,
                "sid1_mrr_at_10": 1.0,
                "sid2_hr_at_10": 0.5,
                "sid2_mrr_at_10": 0.25,
                "overall_hr_at_10": 0.5,
                "overall_mrr_at_10": 0.25,
                "invalid_sid_rate": 0.05,
            },
        )

    def test_unconstrained_beam_search_can_return_invalid_sids(self):
        codebook_sizes = (12, 12, 12)
        valid_codes = [(first, first % 2, 0) for first in range(12)]
        validity_index = SidValidityIndex(valid_codes, codebook_sizes)
        model = _FakeGenerationModel(codebook_sizes)
        context_inputs = {"dummy": torch.ones(2, 1)}

        predictions, scores = sid_beam_search(
            model=model,
            context_inputs=context_inputs,
            beam_size=10,
        )

        self.assertEqual(tuple(predictions.shape), (2, 10, 3))
        self.assertEqual(tuple(scores.shape), (2, 10))
        self.assertTrue(
            any(
                not validity_index.contains(row)
                for sample in predictions.tolist()
                for row in sample
            )
        )

    def test_cached_beam_search_matches_uncached_search(self):
        codebook_sizes = (12, 12, 12)
        context_inputs = {"dummy": torch.ones(2, 1)}

        uncached_predictions, uncached_scores = sid_beam_search(
            model=_FakeGenerationModel(codebook_sizes),
            context_inputs=context_inputs,
            beam_size=10,
            use_kv_cache=False,
        )
        cached_model = _FakeCachedGenerationModel(codebook_sizes)
        cached_predictions, cached_scores = sid_beam_search(
            model=cached_model,
            context_inputs=context_inputs,
            beam_size=10,
            use_kv_cache=True,
        )

        self.assertTrue(torch.equal(cached_predictions, uncached_predictions))
        self.assertTrue(torch.equal(cached_scores, uncached_scores))
        self.assertEqual(cached_model.encode_calls, 1)
        self.assertEqual(cached_model.decode_input_lengths, [1, 1, 1])

    def test_real_kuairand_model_cache_matches_uncached_search(self):
        torch.manual_seed(3)
        codebook_sizes = (12, 12, 12)
        history_lengths = {
            "click": 4,
            "long_view": 4,
            "like": 3,
            "deep_interact": 2,
            "hate": 2,
        }
        qformer_query_counts = {
            "click": 2,
            "long_view": 2,
            "long_view_duration": 2,
            "like": 2,
            "deep_interact": 1,
            "hate": 1,
        }
        config = LazyOneRecConfig(
            codebook_sizes=list(codebook_sizes),
            d_model=16,
            d_ff=32,
            n_layers=2,
            n_context_layers=1,
            n_heads=4,
            n_kv_heads=2,
            max_context_len=16,
            num_gid_embeddings=32,
            user_categorical_cardinalities=tuple(
                4 for _ in CATEGORICAL_USER_FIELDS
            ),
            history_lengths=history_lengths,
            qformer_query_counts=qformer_query_counts,
        )
        model = KuaiRandLazyOneRecForCausalLM(config).eval()
        context_inputs = {}
        for name in GID_SEQUENCE_NAMES:
            gid_ids = torch.randint(
                1,
                32,
                (2, history_lengths[name]),
            )
            context_inputs[f"{name}_gid_ids"] = gid_ids
            context_inputs[f"{name}_attention_mask"] = torch.ones_like(
                gid_ids
            )
        context_inputs["long_view_duration_bucket"] = torch.randint(
            0,
            N_LONG_VIEW_DURATION_BUCKETS,
            (2, history_lengths["long_view"]),
        )
        context_inputs["long_view_duration_attention_mask"] = torch.ones(
            2,
            history_lengths["long_view"],
            dtype=torch.long,
        )
        context_inputs["user_categorical_features"] = torch.randint(
            0,
            4,
            (2, len(CATEGORICAL_USER_FIELDS)),
        )
        context_inputs["user_continuous_features"] = torch.randn(2, 4)
        context_inputs["request_categorical_features"] = torch.zeros(
            2,
            4,
            dtype=torch.long,
        )
        uncached_predictions, uncached_scores = sid_beam_search(
            model=model,
            context_inputs=context_inputs,
            beam_size=10,
            use_kv_cache=False,
        )
        cached_predictions, cached_scores = sid_beam_search(
            model=model,
            context_inputs=context_inputs,
            beam_size=10,
            use_kv_cache=True,
        )

        self.assertTrue(torch.equal(cached_predictions, uncached_predictions))
        self.assertTrue(
            torch.allclose(cached_scores, uncached_scores, atol=1e-5)
        )

    def test_token_codec_decodes_level_specific_tokens(self):
        codec = SidTokenCodec((4, 5, 6))
        codes = [2, 3, 4]
        tokens = codec.encode_codes(codes)
        self.assertEqual(codec.decode_tokens(tokens), codes)


if __name__ == "__main__":
    unittest.main()
