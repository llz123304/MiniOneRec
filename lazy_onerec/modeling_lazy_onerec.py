"""Lazy Decoder-Only recommender backbone (OneRec-V2 style) -- SKELETON.

Status: STRUCTURAL SKELETON for review. The forward math is implemented for the
standard blocks (RMSNorm, RoPE self-attention, SwiGLU FFN) so the shapes line up,
but the pieces marked `# TODO` are the OneRec-V2-specific decisions you should
review/tune before training:

  1. ContextProcessor           -> how heterogeneous user signals become static KV
  2. LazyCrossAttention         -> KV-sharing + GQA read of the context KV
  3. generate() integration     -> encode context once, cache it, decode target

Design contract (so it plugs into the existing repo):
  - Subclasses PreTrainedModel + GenerationMixin  => free resize_token_embeddings()
    and generate(); we only implement forward() and the network.
  - forward(context_input_ids, context_attention_mask, target_input_ids, labels)
    returns CausalLMOutputWithPast(loss=..., logits=...), so transformers.Trainer
    in sft.py works unchanged once the dataset yields these fields.

NOTE: weights are randomly initialized (from-scratch). There is NO from_pretrained.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_lazy_onerec import LazyOneRecConfig


# --------------------------------------------------------------------------- #
# Basic building blocks (standard; provided so shapes are concrete)
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x, cos, sin):
    # x: (B, H, T, Dh)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cos[None, None, :, :] + rot * sin[None, None, :, :]


def repeat_kv(x, n_rep: int):
    # (B, n_kv, T, Dh) -> (B, n_kv*n_rep, T, Dh)  (GQA expansion)
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, d).reshape(b, n_kv * n_rep, t, d)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# --------------------------------------------------------------------------- #
# 1. Context Processor  (TODO: the core OneRec-V2 design choice)
# --------------------------------------------------------------------------- #
class ContextEncoderLayer(nn.Module):
    """One bidirectional self-attention + FFN layer used to mix the context.

    Bidirectional (non-causal): the whole user context is available at once, so
    every context token may attend to every other one before it is projected to
    static KV. Kept intentionally light (few layers) per the OneRec-V2 design.
    """

    def __init__(self, config: LazyOneRecConfig):
        super().__init__()
        self.cfg = config
        self.ln_attn = RMSNorm(config.d_model, config.rms_norm_eps)
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.ln_ffn = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLU(config.d_model, config.d_ff)

    def forward(self, h, attn_mask=None, cos=None, sin=None):
        cfg = self.cfg
        b, t, _ = h.shape
        x = self.ln_attn(h)
        q = self.q_proj(x).view(b, t, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        # Position encoding is configurable (config.position_encoding):
        #   "learned" -> absolute PE added once in ContextProcessor (cos/sin are None)
        #   "rope"    -> applied per-layer here
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # non-causal
        out = out.transpose(1, 2).reshape(b, t, -1)
        h = h + self.o_proj(out)
        h = h + self.ffn(self.ln_ffn(h))
        return h


class SidEmbedding(nn.Module):
    """Per-level SID input embedding (aligns with sentiment/models).

    One independent table per codebook level (sid_emb_table_layer_i) plus a small
    table for the special tokens (PAD/BOS/EOS). Parameters are NOT shared across
    levels: level i has its own vocabulary of K_i codes, exactly mirroring the
    per-level output heads. Class index c at level i and at level j map to
    different embeddings.

    Call signature matches nn.Embedding (global token ids in -> vectors out), so
    both the context encoder and the decoder use it transparently. Global id
    layout (see sid_codec): [PAD, BOS, EOS] then level-0 codes, level-1, ...
    """

    def __init__(self, config: LazyOneRecConfig):
        super().__init__()
        self.n_special = 3  # PAD, BOS, EOS
        self.codebook_sizes = list(config.codebook_sizes)
        self.special_emb = nn.Embedding(self.n_special, config.d_model, config.pad_token_id)
        self.level_emb = nn.ModuleList(
            [nn.Embedding(k, config.d_model) for k in self.codebook_sizes]
        )
        offsets, acc = [], self.n_special
        for k in self.codebook_sizes:
            offsets.append(acc)
            acc += k
        self.register_buffer("level_offsets", torch.tensor(offsets), persistent=False)
        self._vocab_size = acc

    def forward(self, ids: torch.LongTensor) -> torch.Tensor:
        out = ids.new_zeros(*ids.shape, self.special_emb.embedding_dim, dtype=self.special_emb.weight.dtype)
        # specials
        sp = ids < self.n_special
        if sp.any():
            out[sp] = self.special_emb(ids[sp].clamp_min(0))
        # per-level codes
        for lvl, start in enumerate(self.level_offsets.tolist()):
            end = start + self.codebook_sizes[lvl]
            m = (ids >= start) & (ids < end)
            if m.any():
                out[m] = self.level_emb[lvl](ids[m] - start)
        return out

    @property
    def weight(self):  # some HF utilities probe .weight; expose specials' as a stand-in
        return self.special_emb.weight


class ContextProcessor(nn.Module):
    """Turns user context tokens into per-layer STATIC key/value tensors.

    OneRec-V2 idea: the context is encoded once and exposed to every decoder
    layer as (k_l, v_l). With kv_sharing, one KV block feeds `kv_share_every`
    layers, so we only produce ceil(n_layers / kv_share_every) KV blocks.

    Depth is controlled by config.n_context_layers (0 = pure projection, no
    self-attention mixing). This is the "encoder" of the lazy design, kept cheap.

    TODO(you):
      - Extend the input embedding for heterogeneous signals (profile/behavior/
        multimodal), currently per-level SID tables via SidEmbedding.
      - Whether v_l shares k_l's projection (S_kv=1) or has its own (S_kv=2).
    """

    def __init__(self, config: LazyOneRecConfig, embed_tokens: "SidEmbedding"):
        super().__init__()
        self.config = config
        self.embed_tokens = embed_tokens  # shared with decoder input embeddings
        # Learned absolute position embedding (only when position_encoding="learned"),
        # added once before the encoder -- matches sentiment/models tokenizer.py.
        self.use_learned_pe = config.position_encoding == "learned"
        if self.use_learned_pe:
            self.pos_emb = nn.Parameter(torch.zeros(config.max_context_len, config.d_model))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.encoder_layers = nn.ModuleList(
            [ContextEncoderLayer(config) for _ in range(config.n_context_layers)]
        )
        self.n_kv_blocks = (config.n_layers + config.kv_share_every - 1) // config.kv_share_every
        kv_dim = config.n_kv_heads * config.head_dim
        # One (K, V) projection per KV block. Cheap on purpose.
        self.k_proj = nn.ModuleList([nn.Linear(config.d_model, kv_dim, bias=False) for _ in range(self.n_kv_blocks)])
        self.v_proj = nn.ModuleList([nn.Linear(config.d_model, kv_dim, bias=False) for _ in range(self.n_kv_blocks)])
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(self, context_input_ids, context_attention_mask=None):
        cfg = self.config
        h = self.embed_tokens(context_input_ids)  # (B, Lc, D)
        cos = sin = None
        if self.use_learned_pe:
            h = h + self.pos_emb[: h.size(1)].unsqueeze(0)  # learned absolute PE
        else:  # rope: build cache once, share across encoder layers
            cos, sin = build_rope_cache(h.size(1), cfg.head_dim, cfg.rope_theta, h.device, h.dtype)
        enc_mask = None
        if context_attention_mask is not None:
            enc_mask = (1.0 - context_attention_mask[:, None, None, :].to(h.dtype)) * torch.finfo(h.dtype).min
        for layer in self.encoder_layers:
            h = layer(h, enc_mask, cos, sin)
        h = self.norm(h)
        kv_blocks = []
        for i in range(self.n_kv_blocks):
            k = self.k_proj[i](h)  # (B, Lc, n_kv*Dh)
            v = self.v_proj[i](h)
            k = k.view(k.size(0), k.size(1), cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
            v = v.view(v.size(0), v.size(1), cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
            kv_blocks.append((k, v))  # each: (B, n_kv, Lc, Dh)
        return kv_blocks, context_attention_mask  # mask reused for cross-attn


# --------------------------------------------------------------------------- #
# 2. Attention modules
# --------------------------------------------------------------------------- #
class SelfAttention(nn.Module):
    """Causal self-attention over the (short) target sequence, with GQA + RoPE."""

    def __init__(self, config: LazyOneRecConfig):
        super().__init__()
        self.cfg = config
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.n_rep = config.n_heads // config.n_kv_heads

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        cfg = self.cfg
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        if cos is not None:  # RoPE mode; None when learned absolute PE is used
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if past_kv is not None:  # incremental decoding
            pk, pv = past_kv
            k, v = torch.cat([pk, k], dim=2), torch.cat([pv, v], dim=2)
        present = (k, v) if use_cache else None
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=(past_kv is None))
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out), present


class LazyCrossAttention(nn.Module):
    """Query = target hidden; Key/Value = STATIC context KV (from ContextProcessor).

    "Lazy": no KV projection on the context side at inference (it was computed
    once), and GQA expansion is applied on read. This is the module that makes
    OneRec-V2 cheap. TODO: confirm whether queries also need their own norm/scale.
    """

    def __init__(self, config: LazyOneRecConfig):
        super().__init__()
        self.cfg = config
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.n_rep = config.n_heads // config.n_kv_heads

    def forward(self, x, context_kv, context_mask=None):
        cfg = self.cfg
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k, v = context_kv  # (B, n_kv, Lc, Dh) -- precomputed, static
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        attn_mask = None
        if context_mask is not None:
            # (B, Lc) -> (B, 1, 1, Lc) additive mask
            attn_mask = (1.0 - context_mask[:, None, None, :].to(q.dtype)) * torch.finfo(q.dtype).min
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out)


# --------------------------------------------------------------------------- #
# 3. Decoder block & full model
# --------------------------------------------------------------------------- #
class LazyDecoderBlock(nn.Module):
    def __init__(self, config: LazyOneRecConfig, kv_block_idx: int):
        super().__init__()
        self.kv_block_idx = kv_block_idx  # which shared context-KV block this layer reads
        self.ln_self = RMSNorm(config.d_model, config.rms_norm_eps)
        self.self_attn = SelfAttention(config)
        self.ln_cross = RMSNorm(config.d_model, config.rms_norm_eps)
        self.cross_attn = LazyCrossAttention(config)
        self.ln_ffn = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLU(config.d_model, config.d_ff)

    def forward(self, x, cos, sin, context_kv_blocks, context_mask, past_kv=None, use_cache=False):
        # Order matches sentiment/models DecoderBlock: cross-attn -> self-attn -> FFN.
        ctx_kv = context_kv_blocks[self.kv_block_idx]
        x = x + self.cross_attn(self.ln_cross(x), ctx_kv, context_mask)
        h, present = self.self_attn(self.ln_self(x), cos, sin, past_kv, use_cache)
        x = x + h
        x = x + self.ffn(self.ln_ffn(x))
        return x, present


class LazyOneRecForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = LazyOneRecConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: LazyOneRecConfig):
        super().__init__(config)
        # Per-level input embedding (independent table per codebook level), shared
        # between the context encoder and the decoder input side.
        self.embed_tokens = SidEmbedding(config)
        self.context_processor = ContextProcessor(config, self.embed_tokens)
        # Decoder target position encoding: learned absolute PE when configured;
        # otherwise RoPE is applied inside SelfAttention.
        self.use_learned_pe = config.position_encoding == "learned"
        if self.use_learned_pe:
            self.target_pos_emb = nn.Parameter(torch.zeros(config.max_target_len, config.d_model))
            nn.init.trunc_normal_(self.target_pos_emb, std=0.02)
        self.layers = nn.ModuleList(
            [LazyDecoderBlock(config, kv_block_idx=i // config.kv_share_every) for i in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)
        # Per-level output heads (aligns with sentiment/models: one head per
        # codebook level, each projecting to that level's K codes only).
        # Decoder output position i predicts codebook level i.
        self.codebook_sizes = list(config.codebook_sizes)
        self.n_sid_levels = len(self.codebook_sizes)
        self.level_heads = nn.ModuleList(
            [nn.Linear(config.d_model, k, bias=False) for k in self.codebook_sizes]
        )
        # Global-id offset where each level's code range begins (specials first).
        # Matches sid_codec: id = n_special + sum(K[:level]) + code.
        n_special = 3  # PAD, BOS, EOS
        offsets, acc = [], n_special
        for k in self.codebook_sizes:
            offsets.append(acc)
            acc += k
        self.register_buffer("level_offsets", torch.tensor(offsets), persistent=False)
        self.post_init()  # random init of all weights (from scratch)

    # -- input embedding plumbing (vocabulary is fixed by codebook_sizes; the
    #    per-level SidEmbedding is not meant to be resized like an LLM's table) --
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        # Per-level heads; no single tied output matrix. Return None so HF
        # utilities that look for one simply skip tying.
        return None

    def forward(
        self,
        context_input_ids: Optional[torch.LongTensor] = None,
        context_attention_mask: Optional[torch.Tensor] = None,
        target_input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        context_kv_blocks=None,          # reused across decoding steps (see generate)
        past_key_values=None,            # per-layer self-attn cache
        use_cache: bool = False,
        **kwargs,
    ):
        # 1) Encode context ONCE (skip if already cached during generation).
        if context_kv_blocks is None:
            context_kv_blocks, context_attention_mask = self.context_processor(
                context_input_ids, context_attention_mask
            )

        # 2) Decode the target sequence.
        x = self.embed_tokens(target_input_ids)
        seq_start = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if self.use_learned_pe:
            # learned absolute PE, sliced by decoding position (supports KV cache)
            x = x + self.target_pos_emb[seq_start : seq_start + x.size(1)].unsqueeze(0)
            cos = sin = None
        else:
            cos, sin = build_rope_cache(
                seq_start + x.size(1), self.config.head_dim, self.config.rope_theta, x.device, x.dtype
            )
            cos, sin = cos[seq_start:], sin[seq_start:]

        presents = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None
            x, present = layer(x, cos, sin, context_kv_blocks, context_attention_mask, past, use_cache)
            if use_cache:
                presents.append(present)

        x = self.norm(x)

        # Per-level heads: decoder output position t predicts codebook level t.
        # Each head outputs only its level's K codes; we scatter them into a
        # full-vocab logits tensor (other entries -inf) so downstream CE loss
        # and constrained decoding see a standard (B, T, vocab) tensor while a
        # token can only ever be predicted within its own level.
        B, T, _ = x.shape
        neg_inf = torch.finfo(x.dtype).min
        logits = x.new_full((B, T, self.config.vocab_size), neg_inf)
        for t in range(T):
            abs_pos = seq_start + t  # absolute position (supports incremental decode)
            level = abs_pos if abs_pos < self.n_sid_levels else self.n_sid_levels - 1
            head_logits = self.level_heads[level](x[:, t, :])  # (B, K_level)
            start = int(self.level_offsets[level])
            logits[:, t, start : start + head_logits.size(-1)] = head_logits

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        out = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=presents)
        # stash context KV so the next generation step can reuse it
        out.context_kv_blocks = context_kv_blocks
        return out

    # ----------------------------------------------------------------------- #
    # generate() integration.
    # TODO(you): wire prepare_inputs_for_generation so that:
    #   - on the first step, context_input_ids is encoded to context_kv_blocks;
    #   - on later steps, we pass the cached context_kv_blocks + past_key_values
    #     and only feed the last target token as target_input_ids.
    # The evaluate.py call passes input_ids positionally; you will map that to
    # target_input_ids and pass context_* via model_kwargs. Constrained decoding
    # (LogitProcessor.py / prefix_allowed_tokens_fn) then works unchanged.
    # ----------------------------------------------------------------------- #
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        raise NotImplementedError(
            "TODO: implement context-KV caching for generation (see class docstring)."
        )
