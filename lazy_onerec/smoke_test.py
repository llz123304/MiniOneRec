"""Smoke test for the Lazy Decoder-Only skeleton.

Run on a machine with PyTorch installed:
    python -m lazy_onerec.smoke_test

It builds a tiny model, runs one forward with random context/target ids, and
checks that loss/logits shapes are sane and backprop works. This is a STRUCTURE
check (shapes + gradients), not a training run.
"""

import torch

from .configuration_lazy_onerec import LazyOneRecConfig
from .modeling_lazy_onerec import LazyOneRecForCausalLM


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
        assert len(model.level_heads) == 3, len(model.level_heads)

        B, Lc, Lt = 3, 20, 4  # batch, context len, target len (BOS + 3 codes)
        context_input_ids = torch.randint(0, cfg.vocab_size, (B, Lc))
        context_attention_mask = torch.ones(B, Lc)
        target_input_ids = torch.randint(0, cfg.vocab_size, (B, Lt))
        labels = target_input_ids.clone()

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
              f"heads={len(model.level_heads)}  params_with_grad={n_grad}  "
              f"n_kv_blocks={model.context_processor.n_kv_blocks}")


if __name__ == "__main__":
    main()
