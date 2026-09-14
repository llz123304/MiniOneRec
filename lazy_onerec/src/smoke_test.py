"""Smoke test for the KuaiRand LazyOneRec training path.

Run on a machine with PyTorch installed:
    python -m lazy_onerec.src.smoke_test

It builds a tiny model, runs one forward with aligned exposure features and
target SIDs, and checks loss, logits, and gradients.
"""

import torch

from ..data.schema import (
    GID_SEQUENCE_NAMES,
    N_LONG_VIEW_DURATION_BUCKETS,
    REQUEST_CATEGORICAL_CARDINALITIES,
    total_context_length,
)
from ..data.user_features import (
    CATEGORICAL_USER_FIELDS,
    CONTINUOUS_USER_FIELDS,
)
from ..model import LazyOneRecConfig, KuaiRandLazyOneRecForCausalLM
from ..model.modeling import PerTokenLinear, PerTokenSwiGLU


def main():
    torch.manual_seed(0)
    for pe in ("rope", "learned"):
        history_lengths = {
            "click": 6,
            "long_view": 5,
            "like": 4,
            "deep_interact": 3,
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
        cfg = LazyOneRecConfig(
            vocab_size=3 + 3 * 128,       # 3 specials + 3 levels * 128
            codebook_sizes=[128, 128, 128],
            d_model=64,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            n_context_layers=2,
            d_ff=128,
            kv_share_every=2,
            max_target_len=4,            # BOS + 3 codes
            max_context_len=total_context_length(
                history_lengths,
                qformer_query_counts,
            ),
            position_encoding=pe,
        )
        model = KuaiRandLazyOneRecForCausalLM(
            cfg,
            num_gid_embeddings=100,
            user_categorical_cardinalities=(
                8,
            )
            * len(CATEGORICAL_USER_FIELDS),
            history_lengths=history_lengths,
            qformer_query_counts=qformer_query_counts,
        )
        model.train()
        assert len(model.embed_tokens.level_emb) == 3
        assert cfg.use_per_token_qkv
        assert cfg.use_per_token_ffn
        for layer in model.layers:
            assert isinstance(layer.self_attn.q_proj, PerTokenLinear)
            assert isinstance(layer.self_attn.k_proj, PerTokenLinear)
            assert isinstance(layer.self_attn.v_proj, PerTokenLinear)
            assert isinstance(layer.cross_attn.q_proj, PerTokenLinear)
            assert isinstance(layer.ffn, PerTokenSwiGLU)
        feature_embedding = model.context_feature_embedding
        assert feature_embedding.gid_embedding.embedding_dim == 64
        assert (
            feature_embedding.user_categorical_embeddings[0].embedding_dim
            == 128
        )
        assert all(
            embedding.embedding_dim == 8
            for embedding in feature_embedding.user_categorical_embeddings[1:]
        )
        assert all(
            embedding.embedding_dim == 8
            for embedding in feature_embedding.request_categorical_embeddings
        )
        assert (
            feature_embedding.long_view_duration_embedding.embedding_dim == 8
        )

        batch_size, target_length = 3, 4
        behavior_inputs = {}
        for name in GID_SEQUENCE_NAMES:
            gid_ids = torch.randint(
                1,
                100,
                (batch_size, history_lengths[name]),
            )
            behavior_inputs[f"{name}_gid_ids"] = gid_ids
            behavior_inputs[f"{name}_attention_mask"] = torch.ones_like(
                gid_ids
            )
        behavior_inputs["hate_gid_ids"][0].zero_()
        behavior_inputs["hate_attention_mask"][0].zero_()
        behavior_inputs["long_view_duration_bucket"] = torch.randint(
            0,
            N_LONG_VIEW_DURATION_BUCKETS,
            (batch_size, history_lengths["long_view"]),
        )
        behavior_inputs["long_view_duration_attention_mask"] = torch.ones(
            batch_size,
            history_lengths["long_view"],
            dtype=torch.long,
        )
        target_codes = torch.stack(
            [
                torch.randint(0, 128, (batch_size,)) + 3 + level * 128
                for level in range(3)
            ],
            dim=-1,
        )
        target_input_ids = torch.cat(
            [
                torch.full((batch_size, 1), cfg.bos_token_id),
                target_codes,
            ],
            dim=-1,
        )
        labels = target_input_ids.clone()
        labels[:, 0] = -100
        out = model(
            user_categorical_features=torch.randint(
                0,
                8,
                (
                    batch_size,
                    len(CATEGORICAL_USER_FIELDS),
                ),
                dtype=torch.long,
            ),
            user_continuous_features=torch.randn(
                batch_size,
                len(CONTINUOUS_USER_FIELDS),
            ),
            request_categorical_features=torch.stack(
                [
                    torch.randint(0, cardinality, (batch_size,))
                    for cardinality in REQUEST_CATEGORICAL_CARDINALITIES
                ],
                dim=-1,
            ),
            target_input_ids=target_input_ids,
            labels=labels,
            **behavior_inputs,
        )

        assert out.logits.shape == (
            batch_size,
            target_length,
            cfg.vocab_size,
        ), out.logits.shape
        assert (
            out.context_kv_blocks[0][0].shape[2]
            == total_context_length(
                history_lengths,
                qformer_query_counts,
            )
        )
        assert out.loss.dim() == 0 and torch.isfinite(out.loss), out.loss
        out.loss.backward()
        assert model.context_feature_embedding.gid_embedding.weight.grad is not None
        assert (
            model.context_feature_embedding.user_categorical_embeddings[
                0
            ].weight.grad
            is not None
        )
        assert (
            model.context_feature_embedding.request_categorical_embeddings[
                0
            ].weight.grad
            is not None
        )
        assert (
            model.context_feature_embedding.sequence_qformers[
                "click"
            ].query_tokens.grad
            is not None
        )
        per_token_gradient = model.layers[0].self_attn.q_proj.weight.grad
        assert per_token_gradient is not None
        assert torch.count_nonzero(per_token_gradient[:3]).item() > 0
        n_grad = sum(
            1 for parameter in model.parameters() if parameter.grad is not None
        )
        print(
            f"OK [pe={pe}] loss={out.loss.item():.4f} "
            f"logits={tuple(out.logits.shape)} "
            f"params_with_grad={n_grad} "
            f"n_kv_blocks={model.context_processor.n_kv_blocks}"
        )


if __name__ == "__main__":
    main()
