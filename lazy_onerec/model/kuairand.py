"""KuaiRand context-feature adapter around the generic LazyOneRec model."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .configuration import LazyOneRecConfig
from .modeling import LazyOneRecForCausalLM


BINARY_CONTEXT_FIELDS = (
    "long_view",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
)


class KuaiRandContextEmbedding(nn.Module):
    """Build one event token per clicked video from independent sequence slots."""

    def __init__(
        self,
        num_gid_embeddings: int,
        d_model: int,
        gid_dim: int = 128,
        n_tabs: int = 15,
        n_play_ratio_buckets: int = 8,
        n_time_gap_buckets: int = 8,
    ):
        super().__init__()
        self.gid_embedding = nn.Embedding(
            num_gid_embeddings, gid_dim, padding_idx=0
        )
        self.gid_projection = nn.Linear(gid_dim, d_model, bias=False)
        self.binary_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(2, d_model)
                for field in BINARY_CONTEXT_FIELDS
            }
        )
        self.play_ratio_embedding = nn.Embedding(
            n_play_ratio_buckets, d_model
        )
        self.time_gap_embedding = nn.Embedding(n_time_gap_buckets, d_model)
        self.tab_embedding = nn.Embedding(n_tabs, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.gid_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.gid_embedding.weight[0].zero_()
        nn.init.xavier_uniform_(self.gid_projection.weight)
        for embedding in self.binary_embeddings.values():
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.play_ratio_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.time_gap_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.tab_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        context_gid_ids: torch.LongTensor,
        context_play_ratio_bucket: torch.LongTensor,
        context_time_gap_bucket: torch.LongTensor,
        context_tab: torch.LongTensor,
        **binary_context: torch.LongTensor,
    ) -> torch.Tensor:
        event = self.gid_projection(self.gid_embedding(context_gid_ids))
        event = event + self.play_ratio_embedding(context_play_ratio_bucket)
        event = event + self.time_gap_embedding(context_time_gap_bucket)
        event = event + self.tab_embedding(context_tab)
        for field, embedding in self.binary_embeddings.items():
            key = f"context_{field}"
            if key not in binary_context:
                raise ValueError(f"missing aligned KuaiRand sequence slot: {key}")
            event = event + embedding(binary_context[key])
        return self.norm(event)


class KuaiRandLazyOneRecForCausalLM(LazyOneRecForCausalLM):
    """LazyOneRec with an independent KuaiRand GID/action context encoder."""

    def __init__(
        self,
        config: LazyOneRecConfig,
        num_gid_embeddings: Optional[int] = None,
        gid_dim: Optional[int] = None,
    ):
        num_gid_embeddings = int(
            num_gid_embeddings
            if num_gid_embeddings is not None
            else getattr(config, "num_gid_embeddings")
        )
        gid_dim = int(
            gid_dim if gid_dim is not None else getattr(config, "gid_dim", 128)
        )
        config.num_gid_embeddings = num_gid_embeddings
        config.gid_dim = gid_dim
        super().__init__(config)
        self.context_feature_embedding = KuaiRandContextEmbedding(
            num_gid_embeddings=num_gid_embeddings,
            d_model=config.d_model,
            gid_dim=gid_dim,
        )

    def forward(
        self,
        context_gid_ids: torch.LongTensor,
        context_attention_mask: torch.Tensor,
        context_play_ratio_bucket: torch.LongTensor,
        context_time_gap_bucket: torch.LongTensor,
        context_tab: torch.LongTensor,
        target_input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        context_kv_blocks=None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ):
        binary_context = {
            key: kwargs.pop(key)
            for key in tuple(kwargs)
            if key.startswith("context_")
        }
        context_inputs_embeds = self.context_feature_embedding(
            context_gid_ids=context_gid_ids,
            context_play_ratio_bucket=context_play_ratio_bucket,
            context_time_gap_bucket=context_time_gap_bucket,
            context_tab=context_tab,
            **binary_context,
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
