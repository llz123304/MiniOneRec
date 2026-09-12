"""Smoke test for the Lazy Decoder-Only skeleton.

Run on a machine with PyTorch installed:
    python -m lazy_onerec.src.smoke_test

It builds a tiny model, runs one forward with random context/target ids, and
checks that loss/logits shapes are sane and backprop works. This is a STRUCTURE
check (shapes + gradients), not a training run.
"""

import torch

from ..model import LazyOneRecConfig, LazyOneRecForCausalLM


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
        model = LazyOneRecForCausalLM(cfg)
        model.train()
        assert len(model.embed_tokens.level_emb) == 3

        B, n_items, Lt = 3, 6, 4
        level_codes = [
            torch.randint(0, 128, (B, n_items)) + 3 + level * 128
            for level in range(3)
        ]
        context_input_ids = torch.stack(level_codes, dim=-1).reshape(B, -1)
        context_attention_mask = torch.ones_like(context_input_ids)
        target_codes = torch.stack(
            [torch.randint(0, 128, (B,)) + 3 + level * 128 for level in range(3)],
            dim=-1,
        )
        target_input_ids = torch.cat(
            [torch.full((B, 1), cfg.bos_token_id), target_codes], dim=-1
        )
        labels = target_input_ids.clone()
        labels[:, 0] = -100

        out = model(
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            target_input_ids=target_input_ids,
            labels=labels,
        )

        assert out.logits.shape == (B, Lt, cfg.vocab_size), out.logits.shape
        assert out.loss.dim() == 0 and torch.isfinite(out.loss), out.loss
        out.loss.backward()
        n_grad = sum(1 for p in model.parameters() if p.grad is not None)
        print(f"OK [pe={pe}]  loss={out.loss.item():.4f}  logits={tuple(out.logits.shape)}  "
              f"tied_codebooks={model.n_sid_levels}  params_with_grad={n_grad}  "
              f"n_kv_blocks={model.context_processor.n_kv_blocks}")


if __name__ == "__main__":
    main()
