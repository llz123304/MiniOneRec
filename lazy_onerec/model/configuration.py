"""Configuration for the Lazy Decoder-Only recommender (OneRec-V2 style).

This is a *skeleton* config for a from-scratch backbone. It intentionally has no
dependency on any pretrained checkpoint: every field below is a design knob you
set yourself, and the weights are randomly initialized in the model.

Key idea (OneRec-V2, arXiv:2508.20900):
  - The user context (history / profile) is encoded ONCE into a set of static
    key/value tensors (the "Context Processor").
  - The decoder generates the target SID tokens and only *reads* the context via
    a lightweight cross-attention ("Lazy Cross-Attention": KV-sharing + GQA,
    no separate KV projection on the context side).
So computation concentrates on target decoding rather than re-encoding context.
"""

from transformers import PretrainedConfig


class LazyOneRecConfig(PretrainedConfig):
    model_type = "lazy_onerec"

    def __init__(
        self,
        vocab_size: int = 32000,          # = 3 specials + sum(codebook_sizes)
        codebook_sizes: list = None,      # per-level K, e.g. [256,256,256] (must match SID generation)
        d_model: int = 768,
        n_layers: int = 6,
        n_heads: int = 12,                # query heads (self-attention)
        n_kv_heads: int = 2,              # GQA: key/value head groups (< n_heads)
        n_context_layers: int = 2,        # depth of the (lazy) context encoder; 0 = pure projection
        d_ff: int = 3072,
        max_target_len: int = 4,          # BOS + 3-level codebook SID
        max_context_len: int = 3000,      # OneRec-V2 scales context up to ~3000
        # --- Lazy cross-attention knobs ---
        kv_sharing: bool = True,          # reuse one KV set across several layers
        kv_share_every: int = 2,          # S_kv: one KV block feeds this many layers
        # --- misc ---
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        position_encoding: str = "rope",  # "rope" | "learned"; applies to both encoder & decoder
        pad_token_id: int = 0,            # placeholder; set from tokenizer at build time
        bos_token_id: int = 1,            # placeholder; set from tokenizer at build time
        eos_token_id: int = 2,            # placeholder; set from tokenizer at build time
        # Per-level SID input/output weights are tied manually in the model.
        # Keep HF's global tying disabled because there is no single LM head.
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.codebook_sizes = codebook_sizes if codebook_sizes is not None else [256, 256, 256]
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_context_layers = n_context_layers
        self.d_ff = d_ff
        self.max_target_len = max_target_len
        self.max_context_len = max_context_len
        self.kv_sharing = kv_sharing
        self.kv_share_every = kv_share_every
        self.dropout = dropout
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        assert position_encoding in ("rope", "learned"), position_encoding
        self.position_encoding = position_encoding
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        return self.d_model // self.n_heads

    @classmethod
    def from_codec(cls, codec, **kwargs):
        """Build a config whose vocabulary matches a SidCodec exactly."""
        return cls(
            vocab_size=codec.vocab_size,
            codebook_sizes=list(codec.codebook_sizes),
            max_target_len=codec.n_levels + 1,  # BOS + one token per level
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            **kwargs,
        )
