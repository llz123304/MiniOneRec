"""KuaiRand model configuration and training-builder regressions."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from lazy_onerec.data.user_features import CATEGORICAL_USER_FIELDS
from lazy_onerec.model import LazyOneRecConfig, KuaiRandLazyOneRecForCausalLM
from lazy_onerec.src.train_kuairand import (
    build_context_layout,
    build_model,
    build_training_arguments,
)


def _arguments(output_dir: str) -> SimpleNamespace:
    return SimpleNamespace(
        click_history_length=4,
        long_view_history_length=4,
        like_history_length=3,
        deep_interact_history_length=2,
        hate_history_length=2,
        click_query_tokens=2,
        long_view_query_tokens=2,
        long_view_duration_query_tokens=2,
        like_query_tokens=2,
        deep_interact_query_tokens=1,
        hate_query_tokens=1,
        d_model=16,
        d_ff=32,
        n_layers=2,
        n_context_layers=1,
        n_heads=4,
        n_kv_heads=2,
        per_token_qkv=True,
        per_token_ffn=True,
        kv_sharing=True,
        kv_share_every=2,
        position_encoding="rope",
        gid_dim=8,
        user_id_dim=8,
        categorical_dim=4,
        continuous_dim=4,
        duration_dim=4,
        qformer_layers=1,
        positive_target="click",
        num_train_epochs=2,
        output_dir=output_dir,
        batch_size=8,
        micro_batch_size=2,
        num_workers=0,
        prefetch_factor=2,
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_steps=1,
        optimizer="adamw_torch",
        logging_steps=1,
        bf16=False,
    )


class KuaiRandModelConfigTest(unittest.TestCase):
    def test_builders_put_all_model_state_in_config(self):
        with tempfile.TemporaryDirectory() as output_dir:
            args = _arguments(output_dir)
            history_lengths, query_counts = build_context_layout(args)
            artifact = SimpleNamespace(codebook_sizes=(8, 8, 8))
            corpus = SimpleNamespace(
                num_gid_embeddings=32,
                user_features=SimpleNamespace(
                    categorical_cardinalities=tuple(
                        4 for _ in CATEGORICAL_USER_FIELDS
                    )
                ),
            )

            model = build_model(
                args,
                artifact,
                corpus,
                history_lengths,
                query_counts,
            )
            training_args = build_training_arguments(args)

        self.assertEqual(model.config.num_gid_embeddings, 32)
        self.assertEqual(model.config.history_lengths, history_lengths)
        self.assertEqual(model.config.qformer_query_counts, query_counts)
        self.assertEqual(model.config.positive_target, "click")
        self.assertEqual(model.config.num_train_epochs, 2)
        self.assertEqual(training_args.gradient_accumulation_steps, 4)
        self.assertEqual(training_args.num_train_epochs, 2)

    def test_model_save_load_round_trip_uses_config_only(self):
        config = LazyOneRecConfig(
            codebook_sizes=[8, 8, 8],
            d_model=16,
            d_ff=32,
            n_layers=1,
            n_context_layers=1,
            n_heads=4,
            n_kv_heads=2,
            max_context_len=12,
            num_gid_embeddings=32,
            user_categorical_cardinalities=[
                4 for _ in CATEGORICAL_USER_FIELDS
            ],
            history_lengths={
                "click": 4,
                "long_view": 4,
                "like": 3,
                "deep_interact": 2,
                "hate": 2,
            },
            qformer_query_counts={
                "click": 2,
                "long_view": 2,
                "long_view_duration": 2,
                "like": 2,
                "deep_interact": 1,
                "hate": 1,
            },
        )
        model = KuaiRandLazyOneRecForCausalLM(config)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            model.save_pretrained(checkpoint)
            restored = KuaiRandLazyOneRecForCausalLM.from_pretrained(checkpoint)

        for name in (
            "codebook_sizes",
            "max_target_len",
            "num_gid_embeddings",
            "user_categorical_cardinalities",
            "gid_dim",
            "user_id_dim",
            "categorical_dim",
            "continuous_dim",
            "duration_dim",
            "history_lengths",
            "qformer_query_counts",
            "qformer_layers",
            "positive_target",
            "num_train_epochs",
        ):
            self.assertEqual(
                getattr(restored.config, name),
                getattr(model.config, name),
            )
        self.assertEqual(
            restored.context_feature_embedding.gid_embedding.num_embeddings,
            32,
        )

    def test_target_length_must_match_sid_levels(self):
        with self.assertRaisesRegex(ValueError, "does not match 2 SID levels"):
            LazyOneRecConfig(
                codebook_sizes=[8, 8],
                max_target_len=4,
            )


if __name__ == "__main__":
    unittest.main()
