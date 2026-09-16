"""KuaiRand context-feature adapter around the generic LazyOneRec model."""

from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn as nn

from ..data.schema import (
    CONTEXT_SEQUENCE_NAMES,
    DEFAULT_GID_SEQUENCE_LENGTHS,
    DEFAULT_QFORMER_QUERY_COUNTS,
    N_LONG_VIEW_DURATION_BUCKETS,
    N_USER_REQUEST_CONTEXT_TOKENS,
    REQUEST_CATEGORICAL_CARDINALITIES,
    REQUEST_CATEGORICAL_FIELDS,
    validate_gid_sequence_lengths,
    validate_qformer_query_counts,
)
from ..data.user_features import (
    CATEGORICAL_USER_FIELDS,
    CONTINUOUS_USER_FIELDS,
)
from .configuration import LazyOneRecConfig
from .modeling import LazyOneRecForCausalLM, RMSNorm, SwiGLU


class QFormerLayer(nn.Module):
    """One normalized cross-attention and FFN query update."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.query_norm = RMSNorm(d_model, rms_norm_eps)
        self.context_norm = RMSNorm(d_model, rms_norm_eps)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.ffn_norm = RMSNorm(d_model, rms_norm_eps)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized_context = self.context_norm(context)
        attention_output, _ = self.cross_attention(
            query=self.query_norm(queries),
            key=normalized_context,
            value=normalized_context,
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        queries = queries + attention_output
        return queries + self.ffn(self.ffn_norm(queries))


class SequenceQFormer(nn.Module):
    """Compress one sequence with its own queries and Q-Former layers."""

    def __init__(
        self,
        query_count: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        rms_norm_eps: float,
    ):
        super().__init__()
        if query_count <= 0:
            raise ValueError("Q-Former query count must be positive")
        if n_layers <= 0:
            raise ValueError("qformer_layers must be positive")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by Q-Former heads")
        self.query_count = int(query_count)
        self.query_tokens = nn.Parameter(
            torch.empty(self.query_count, d_model)
        )
        self.layers = nn.ModuleList(
            [
                QFormerLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(n_layers)
            ]
        )
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)

    def forward(
        self,
        context: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_mask = attention_mask.to(dtype=torch.bool)
        has_events = valid_mask.any(dim=1)
        safe_mask = valid_mask.clone()
        safe_context = context
        if not bool(has_events.all()):
            safe_context = context.clone()
            safe_context[~has_events, 0] = 0
            safe_mask[~has_events, 0] = True

        queries = self.query_tokens.unsqueeze(0).expand(
            context.size(0),
            -1,
            -1,
        )
        for layer in self.layers:
            queries = layer(queries, safe_context, safe_mask)

        query_mask = has_events[:, None].expand(-1, self.query_count)
        queries = queries * query_mask.unsqueeze(-1).to(queries.dtype)
        return queries, query_mask.to(attention_mask.dtype)


class KuaiRandContextEmbedding(nn.Module):
    """Convert static features and behavior histories into context tokens."""

    def __init__(
        self,
        num_gid_embeddings: int,
        user_categorical_cardinalities: tuple[int, ...],
        d_model: int,
        gid_dim: int = 64,
        user_id_dim: int = 128,
        categorical_dim: int = 8,
        continuous_dim: int = 16,
        duration_dim: int = 8,
        history_lengths: dict[str, int] | None = None,
        qformer_query_counts: dict[str, int] | None = None,
        qformer_layers: int = 1,
        qformer_heads: int = 4,
        qformer_d_ff: int = 1024,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        if len(user_categorical_cardinalities) != len(
            CATEGORICAL_USER_FIELDS
        ):
            raise ValueError(
                "user categorical cardinalities must align with fields"
            )
        if min(
            gid_dim,
            user_id_dim,
            categorical_dim,
            continuous_dim,
            duration_dim,
        ) <= 0:
            raise ValueError("all feature embedding dimensions must be positive")

        self.d_model = d_model
        history_lengths = (
            DEFAULT_GID_SEQUENCE_LENGTHS
            if history_lengths is None
            else history_lengths
        )
        validate_gid_sequence_lengths(history_lengths)
        context_sequence_lengths = {
            **history_lengths,
            "long_view_duration": history_lengths["long_view"],
        }
        self.gid_embedding = nn.Embedding(
            num_gid_embeddings, gid_dim, padding_idx=0
        )
        self.gid_projection = nn.Linear(gid_dim, d_model, bias=False)
        self.user_categorical_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    cardinality,
                    user_id_dim if index == 0 else categorical_dim,
                )
                for index, cardinality in enumerate(
                    user_categorical_cardinalities
                )
            ]
        )
        self.user_continuous_projection = nn.Linear(
            len(CONTINUOUS_USER_FIELDS),
            continuous_dim,
            bias=False,
        )
        static_input_dim = (
            user_id_dim
            + categorical_dim
            * (
                len(CATEGORICAL_USER_FIELDS)
                - 1
                + len(REQUEST_CATEGORICAL_FIELDS)
            )
            + continuous_dim
        )
        self.request_categorical_embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, categorical_dim)
                for cardinality in REQUEST_CATEGORICAL_CARDINALITIES
            ]
        )
        self.user_request_projection = nn.Linear(
            static_input_dim,
            N_USER_REQUEST_CONTEXT_TOKENS * d_model,
            bias=False,
        )
        self.sequence_type_embedding = nn.Embedding(
            len(CONTEXT_SEQUENCE_NAMES),
            d_model,
        )
        self.sequence_position_embeddings = nn.ParameterDict(
            {
                name: nn.Parameter(
                    torch.empty(context_sequence_lengths[name], d_model)
                )
                for name in CONTEXT_SEQUENCE_NAMES
            }
        )
        self.long_view_duration_embedding = nn.Embedding(
            N_LONG_VIEW_DURATION_BUCKETS,
            duration_dim,
        )
        self.long_view_duration_projection = nn.Linear(
            duration_dim,
            d_model,
            bias=False,
        )
        self.user_request_norm = nn.LayerNorm(d_model)
        self.sequence_norm = nn.LayerNorm(d_model)
        qformer_query_counts = (
            DEFAULT_QFORMER_QUERY_COUNTS
            if qformer_query_counts is None
            else qformer_query_counts
        )
        validate_qformer_query_counts(qformer_query_counts)
        if any(
            int(qformer_query_counts[name])
            > int(context_sequence_lengths[name])
            for name in CONTEXT_SEQUENCE_NAMES
        ):
            raise ValueError(
                "Q-Former query count cannot exceed source sequence length"
            )
        self.sequence_qformers = nn.ModuleDict(
            {
                name: SequenceQFormer(
                    query_count=int(qformer_query_counts[name]),
                    d_model=d_model,
                    n_heads=qformer_heads,
                    d_ff=qformer_d_ff,
                    n_layers=qformer_layers,
                    rms_norm_eps=rms_norm_eps,
                )
                for name in CONTEXT_SEQUENCE_NAMES
            }
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.gid_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.gid_embedding.weight[0].zero_()
        for embedding in self.user_categorical_embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        for embedding in self.request_categorical_embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(
            self.sequence_type_embedding.weight,
            mean=0.0,
            std=0.02,
        )
        for positions in self.sequence_position_embeddings.values():
            nn.init.normal_(positions, mean=0.0, std=0.02)
        nn.init.normal_(
            self.long_view_duration_embedding.weight,
            mean=0.0,
            std=0.02,
        )
        for projection in (
            self.gid_projection,
            self.user_continuous_projection,
            self.user_request_projection,
            self.long_view_duration_projection,
        ):
            nn.init.xavier_uniform_(projection.weight)

    def forward(
        self,
        click_gid_ids: torch.LongTensor,
        click_attention_mask: torch.Tensor,
        long_view_gid_ids: torch.LongTensor,
        long_view_attention_mask: torch.Tensor,
        long_view_duration_bucket: torch.LongTensor,
        long_view_duration_attention_mask: torch.Tensor,
        like_gid_ids: torch.LongTensor,
        like_attention_mask: torch.Tensor,
        deep_interact_gid_ids: torch.LongTensor,
        deep_interact_attention_mask: torch.Tensor,
        hate_gid_ids: torch.LongTensor,
        hate_attention_mask: torch.Tensor,
        user_categorical_features: torch.LongTensor,
        user_continuous_features: torch.Tensor,
        request_categorical_features: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = click_gid_ids.size(0)
        user_parts = [
            embedding(user_categorical_features[:, index])
            for index, embedding in enumerate(
                self.user_categorical_embeddings
            )
        ]
        request_parts = [
            embedding(request_categorical_features[:, index])
            for index, embedding in enumerate(
                self.request_categorical_embeddings
            )
        ]
        static_features = torch.cat(
            [
                *user_parts,
                self.user_continuous_projection(user_continuous_features),
                *request_parts,
            ],
            dim=-1,
        )
        static_tokens = self.user_request_projection(static_features).view(
            batch_size,
            N_USER_REQUEST_CONTEXT_TOKENS,
            self.d_model,
        )
        static_tokens = self.user_request_norm(static_tokens)

        gid_sequences = {
            "click": (click_gid_ids, click_attention_mask),
            "long_view": (long_view_gid_ids, long_view_attention_mask),
            "like": (like_gid_ids, like_attention_mask),
            "deep_interact": (
                deep_interact_gid_ids,
                deep_interact_attention_mask,
            ),
            "hate": (hate_gid_ids, hate_attention_mask),
        }
        if (
            long_view_duration_bucket.shape
            != long_view_duration_attention_mask.shape
        ):
            raise ValueError(
                "long_view duration sequence and mask must align"
            )

        tokens = []
        masks = []
        for sequence_index, name in enumerate(CONTEXT_SEQUENCE_NAMES):
            if name == "long_view_duration":
                event = self.long_view_duration_embedding(
                    long_view_duration_bucket
                )
                event = self.long_view_duration_projection(event)
                mask = long_view_duration_attention_mask
            else:
                gid_ids, mask = gid_sequences[name]
                event = self.gid_projection(self.gid_embedding(gid_ids))
                if gid_ids.shape != mask.shape:
                    raise ValueError(
                        f"{name} GID sequence and mask must align"
                    )
            event = event + self.sequence_type_embedding.weight[
                sequence_index
            ].view(1, 1, -1)
            position_embedding = self.sequence_position_embeddings[name]
            if event.size(1) != position_embedding.size(0):
                raise ValueError(
                    f"{name} length {event.size(1)} does not match configured "
                    f"length {position_embedding.size(0)}"
                )
            event = event + position_embedding.unsqueeze(0)
            event = self.sequence_norm(event)
            event = event * mask.unsqueeze(-1).to(event.dtype)
            compressed, compressed_mask = self.sequence_qformers[name](
                event,
                mask,
            )
            tokens.append(compressed)
            masks.append(compressed_mask)

        static_mask = torch.ones(
            batch_size,
            N_USER_REQUEST_CONTEXT_TOKENS,
            dtype=click_attention_mask.dtype,
            device=click_attention_mask.device,
        )
        return (
            torch.cat([static_tokens, *tokens], dim=1),
            torch.cat([static_mask, *masks], dim=1),
        )


class KuaiRandLazyOneRecForCausalLM(LazyOneRecForCausalLM):
    """LazyOneRec with an independent KuaiRand GID/action context encoder."""

    accepts_loss_kwargs = True
    context_input_names = (
        "click_gid_ids",
        "click_attention_mask",
        "long_view_gid_ids",
        "long_view_attention_mask",
        "long_view_duration_bucket",
        "long_view_duration_attention_mask",
        "like_gid_ids",
        "like_attention_mask",
        "deep_interact_gid_ids",
        "deep_interact_attention_mask",
        "hate_gid_ids",
        "hate_attention_mask",
        "user_categorical_features",
        "user_continuous_features",
        "request_categorical_features",
    )

    def __init__(
        self,
        config: LazyOneRecConfig,
        num_gid_embeddings: Optional[int] = None,
        user_categorical_cardinalities: Optional[tuple[int, ...]] = None,
        gid_dim: Optional[int] = None,
        user_id_dim: Optional[int] = None,
        categorical_dim: Optional[int] = None,
        continuous_dim: Optional[int] = None,
        duration_dim: Optional[int] = None,
        history_lengths: Optional[dict[str, int]] = None,
        qformer_query_counts: Optional[dict[str, int]] = None,
        qformer_layers: Optional[int] = None,
    ):
        num_gid_embeddings = int(
            num_gid_embeddings
            if num_gid_embeddings is not None
            else getattr(config, "num_gid_embeddings")
        )
        gid_dim = int(
            gid_dim if gid_dim is not None else getattr(config, "gid_dim", 64)
        )
        user_id_dim = int(
            user_id_dim
            if user_id_dim is not None
            else getattr(config, "user_id_dim", 128)
        )
        categorical_dim = int(
            categorical_dim
            if categorical_dim is not None
            else getattr(config, "categorical_dim", 8)
        )
        continuous_dim = int(
            continuous_dim
            if continuous_dim is not None
            else getattr(config, "continuous_dim", 16)
        )
        duration_dim = int(
            duration_dim
            if duration_dim is not None
            else getattr(config, "duration_dim", 8)
        )
        if history_lengths is None:
            history_lengths = getattr(
                config,
                "history_lengths",
                DEFAULT_GID_SEQUENCE_LENGTHS,
            )
        history_lengths = {
            name: int(history_lengths[name])
            for name in DEFAULT_GID_SEQUENCE_LENGTHS
        }
        validate_gid_sequence_lengths(history_lengths)
        if qformer_query_counts is None:
            qformer_query_counts = getattr(
                config,
                "qformer_query_counts",
                DEFAULT_QFORMER_QUERY_COUNTS,
            )
        qformer_query_counts = {
            name: int(qformer_query_counts[name])
            for name in CONTEXT_SEQUENCE_NAMES
        }
        validate_qformer_query_counts(qformer_query_counts)
        qformer_layers = int(
            qformer_layers
            if qformer_layers is not None
            else getattr(config, "qformer_layers", 1)
        )
        if user_categorical_cardinalities is None:
            user_categorical_cardinalities = getattr(
                config,
                "user_categorical_cardinalities",
                None,
            )
        if user_categorical_cardinalities is None:
            raise ValueError("user categorical cardinalities are required")
        user_categorical_cardinalities = tuple(
            int(value) for value in user_categorical_cardinalities
        )
        config.num_gid_embeddings = num_gid_embeddings
        config.gid_dim = gid_dim
        config.user_categorical_cardinalities = list(
            user_categorical_cardinalities
        )
        config.user_id_dim = user_id_dim
        config.categorical_dim = categorical_dim
        config.continuous_dim = continuous_dim
        config.duration_dim = duration_dim
        config.history_lengths = history_lengths
        config.qformer_query_counts = qformer_query_counts
        config.qformer_layers = qformer_layers
        super().__init__(config)
        self.context_feature_embedding = KuaiRandContextEmbedding(
            num_gid_embeddings=num_gid_embeddings,
            user_categorical_cardinalities=user_categorical_cardinalities,
            d_model=config.d_model,
            gid_dim=gid_dim,
            user_id_dim=user_id_dim,
            categorical_dim=categorical_dim,
            continuous_dim=continuous_dim,
            duration_dim=duration_dim,
            history_lengths=history_lengths,
            qformer_query_counts=qformer_query_counts,
            qformer_layers=qformer_layers,
            qformer_heads=config.n_heads,
            qformer_d_ff=config.d_ff,
            rms_norm_eps=config.rms_norm_eps,
        )

    def _embed_context(
        self,
        context_inputs: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        missing = [
            name for name in self.context_input_names if name not in context_inputs
        ]
        if missing:
            raise ValueError(f"missing context inputs: {missing}")

        batch_size = context_inputs["click_gid_ids"].size(0)
        if context_inputs["user_categorical_features"].shape != (
            batch_size,
            len(CATEGORICAL_USER_FIELDS),
        ):
            raise ValueError("expected user_categorical_features with shape [B,26]")
        if context_inputs["user_continuous_features"].shape != (
            batch_size,
            len(CONTINUOUS_USER_FIELDS),
        ):
            raise ValueError("expected user_continuous_features with shape [B,4]")
        if context_inputs["request_categorical_features"].shape != (
            batch_size,
            len(REQUEST_CATEGORICAL_FIELDS),
        ):
            raise ValueError(
                "expected request_categorical_features with shape [B,4]"
            )
        return self.context_feature_embedding(
            **{
                name: context_inputs[name]
                for name in self.context_input_names
            }
        )

    def encode_context(
        self,
        context_inputs: Mapping[str, torch.Tensor],
    ):
        """Encode KuaiRand features once into reusable cross-attention KV."""
        context_inputs_embeds, context_attention_mask = self._embed_context(
            context_inputs
        )
        return self.context_processor(
            context_inputs_embeds=context_inputs_embeds,
            context_attention_mask=context_attention_mask,
        )

    def decode_with_context(
        self,
        target_input_ids: torch.LongTensor,
        context_kv_blocks,
        context_attention_mask: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
    ):
        """Decode SID tokens without recomputing KuaiRand context features."""
        return super().forward(
            target_input_ids=target_input_ids,
            context_kv_blocks=context_kv_blocks,
            context_attention_mask=context_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    def forward(
        self,
        click_gid_ids: torch.LongTensor,
        click_attention_mask: torch.Tensor,
        long_view_gid_ids: torch.LongTensor,
        long_view_attention_mask: torch.Tensor,
        long_view_duration_bucket: torch.LongTensor,
        long_view_duration_attention_mask: torch.Tensor,
        like_gid_ids: torch.LongTensor,
        like_attention_mask: torch.Tensor,
        deep_interact_gid_ids: torch.LongTensor,
        deep_interact_attention_mask: torch.Tensor,
        hate_gid_ids: torch.LongTensor,
        hate_attention_mask: torch.Tensor,
        user_categorical_features: torch.LongTensor,
        user_continuous_features: torch.Tensor,
        request_categorical_features: torch.LongTensor,
        target_input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        context_kv_blocks=None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ):
        context_inputs_embeds, context_attention_mask = self._embed_context(
            {
                "click_gid_ids": click_gid_ids,
                "click_attention_mask": click_attention_mask,
                "long_view_gid_ids": long_view_gid_ids,
                "long_view_attention_mask": long_view_attention_mask,
                "long_view_duration_bucket": long_view_duration_bucket,
                "long_view_duration_attention_mask": (
                    long_view_duration_attention_mask
                ),
                "like_gid_ids": like_gid_ids,
                "like_attention_mask": like_attention_mask,
                "deep_interact_gid_ids": deep_interact_gid_ids,
                "deep_interact_attention_mask": deep_interact_attention_mask,
                "hate_gid_ids": hate_gid_ids,
                "hate_attention_mask": hate_attention_mask,
                "user_categorical_features": user_categorical_features,
                "user_continuous_features": user_continuous_features,
                "request_categorical_features": request_categorical_features,
            }
        )
        return super().forward(
            context_inputs_embeds=context_inputs_embeds,
            context_attention_mask=context_attention_mask,
            target_input_ids=target_input_ids,
            labels=labels,
            context_kv_blocks=context_kv_blocks,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
