"""Tests for constrained SID generation metrics."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from lazy_onerec.model.inference import constrained_sid_beam_search
from lazy_onerec.sid.evaluation import SidPrefixIndex, SidRankingMetrics
from lazy_onerec.sid.layout import N_SPECIAL
from lazy_onerec.sid.token_codec import SidTokenCodec


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


class SidEvaluationTest(unittest.TestCase):
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
        prefix_index = SidPrefixIndex(valid_rows, (10, 10, 10))
        metrics = SidRankingMetrics(prefix_index)

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

    def test_constrained_beam_search_only_returns_valid_sids(self):
        codebook_sizes = (12, 12, 12)
        valid_codes = [(first, first % 2, 0) for first in range(12)]
        prefix_index = SidPrefixIndex(valid_codes, codebook_sizes)
        model = _FakeGenerationModel(codebook_sizes)
        context_inputs = {"dummy": torch.ones(2, 1)}

        predictions, scores = constrained_sid_beam_search(
            model=model,
            context_inputs=context_inputs,
            prefix_index=prefix_index,
            beam_size=10,
        )

        self.assertEqual(tuple(predictions.shape), (2, 10, 3))
        self.assertEqual(tuple(scores.shape), (2, 10))
        self.assertTrue(
            all(
                prefix_index.contains(row)
                for sample in predictions.tolist()
                for row in sample
            )
        )

    def test_token_codec_decodes_level_specific_tokens(self):
        codec = SidTokenCodec((4, 5, 6))
        codes = [2, 3, 4]
        tokens = codec.encode_codes(codes)
        self.assertEqual(codec.decode_tokens(tokens), codes)


if __name__ == "__main__":
    unittest.main()
