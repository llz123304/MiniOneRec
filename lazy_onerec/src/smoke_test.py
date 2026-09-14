"""Smoke test for the KuaiRand LazyOneRec training path.

Run on a machine with PyTorch installed:
    python -m lazy_onerec.src.smoke_test

It builds a tiny model, runs one forward with aligned exposure features and
target SIDs, and checks loss, logits, and gradients.
"""

import torch

from ..data.schema import (
    BINARY_EVENT_FIELDS,
    N_PLAY_RATIO_BUCKETS,
    N_TAB_EMBEDDINGS,
    N_TIME_GAP_BUCKETS,
    REQUEST_CATEGORICAL_FIELDS,
)
from ..data.user_features import (
    CATEGORICAL_USER_FIELDS,
    CONTINUOUS_USER_FIELDS,
)
from ..model import LazyOneRecConfig, KuaiRandLazyOneRecForCausalLM


def main():
    torch.manual_seed(0)
    for pe in ("rope", "learned"):
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
            position_encoding=pe,
        )
        model = KuaiRandLazyOneRecForCausalLM(
            cfg,
            num_gid_embeddings=100,
            gid_dim=16,
        )
        model.train()
        assert len(model.embed_tokens.level_emb) == 3

        batch_size, history_length, target_length = 3, 6, 4
        context_gid_ids = torch.randint(
            1, 100, (batch_size, history_length)
        )
        context_attention_mask = torch.ones_like(context_gid_ids)
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
        binary_context = {
            f"context_{field}": torch.randint(
                0, 2, (batch_size, history_length)
            )
            for field in BINARY_EVENT_FIELDS
        }

        out = model(
            context_gid_ids=context_gid_ids,
            context_attention_mask=context_attention_mask,
            user_categorical_features=torch.zeros(
                batch_size,
                len(CATEGORICAL_USER_FIELDS),
                dtype=torch.long,
            ),
            user_continuous_features=torch.zeros(
                batch_size,
                len(CONTINUOUS_USER_FIELDS),
            ),
            request_categorical_features=torch.zeros(
                batch_size,
                len(REQUEST_CATEGORICAL_FIELDS),
                dtype=torch.long,
            ),
            context_play_ratio_bucket=torch.randint(
                0,
                N_PLAY_RATIO_BUCKETS,
                (batch_size, history_length),
            ),
            context_time_gap_bucket=torch.randint(
                0,
                N_TIME_GAP_BUCKETS,
                (batch_size, history_length),
            ),
            context_tab=torch.randint(
                0,
                N_TAB_EMBEDDINGS,
                (batch_size, history_length),
            ),
            target_input_ids=target_input_ids,
            labels=labels,
            **binary_context,
        )

        assert out.logits.shape == (
            batch_size,
            target_length,
            cfg.vocab_size,
        ), out.logits.shape
        assert out.loss.dim() == 0 and torch.isfinite(out.loss), out.loss
        out.loss.backward()
        assert model.context_feature_embedding.gid_embedding.weight.grad is not None
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
