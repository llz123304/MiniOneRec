"""Configuration for the Lazy Decoder-Only recommender (OneRec-V2 style).

The backbone has no dependency on a pretrained checkpoint; all weights are
initialized from scratch.

Key idea (OneRec-V2, arXiv:2508.20900):
  - The user context (history / profile) is encoded ONCE into a set of static
    key/value tensors (the "Context Processor").
  - The decoder generates the target SID tokens and only *reads* the context via
    a lightweight cross-attention ("Lazy Cross-Attention": KV-sharing + GQA,
    no separate KV projection on the context side).
So computation concentrates on target decoding rather than re-encoding context.
"""

from transformers import PretrainedConfig

from ..sid.layout import BOS_ID, EOS_ID, N_SPECIAL, PAD_ID


class LazyOneRecConfig(PretrainedConfig):
    model_type = "lazy_onerec"

    def __init__(
        self,
        vocab_size: int = None,           # derived from special tokens + codebooks
        codebook_sizes: list = None,      # per-level K, e.g. [256,256,256] (must match SID generation)
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,                 # query heads (self-attention)
        n_kv_heads: int = 2,              # GQA: key/value head groups (< n_heads)
        n_context_layers: int = 2,        # depth of the (lazy) context encoder; 0 = pure projection
        d_ff: int = 1024,
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
        pad_token_id: int = PAD_ID,
        bos_token_id: int = BOS_ID,
        eos_token_id: int = EOS_ID,
        # Per-level SID input/output weights are tied manually in the model.
        # Keep HF's global tying disabled because there is no single LM head.
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.codebook_sizes = (
            codebook_sizes if codebook_sizes is not None else [256, 256, 256]
        )
        if not self.codebook_sizes or any(
            int(size) <= 0 for size in self.codebook_sizes
        ):
            raise ValueError("codebook_sizes must contain positive integers")
        self.codebook_sizes = [int(size) for size in self.codebook_sizes]
        if d_model <= 0 or n_heads <= 0 or n_kv_heads <= 0:
            raise ValueError("d_model and attention head counts must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if position_encoding not in {"rope", "learned"}:
            raise ValueError("position_encoding must be 'rope' or 'learned'")
        if position_encoding == "rope" and (d_model // n_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        expected_vocab_size = N_SPECIAL + sum(self.codebook_sizes)
        if vocab_size is not None and vocab_size != expected_vocab_size:
            raise ValueError(
                f"vocab_size={vocab_size} does not match SID layout "
                f"{expected_vocab_size}"
            )
        self.vocab_size = expected_vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_context_layers = n_context_layers
        self.d_ff = d_ff
        self.max_target_len = max_target_len
        self.max_context_len = max_context_len
        self.kv_sharing = kv_sharing
        if kv_share_every <= 0:
            raise ValueError("kv_share_every must be positive")
        self.kv_share_every = kv_share_every
        self.dropout = dropout
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
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
        return self.d_model // self.n_heads

    @classmethod
    def from_codebook_sizes(cls, codebook_sizes, **kwargs):
        """Build a config directly from raw per-level codebook sizes."""
        sizes = [int(size) for size in codebook_sizes]
        kwargs.setdefault("max_target_len", len(sizes) + 1)
        return cls(
            codebook_sizes=sizes,
            **kwargs,
        )

    @property
    def kv_share_stride(self) -> int:
        return self.kv_share_every if self.kv_sharing else 1
