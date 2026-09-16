"""Cached constrained beam search for SID generation."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from ..sid.evaluation import EVALUATION_TOP_K, SidPrefixIndex
from ..sid.layout import BOS_ID, sid_level_offsets


def _repeat_context_cache(context_kv_blocks, repeats: int):
    if repeats == 1:
        return context_kv_blocks
    return tuple(
        (
            key.repeat_interleave(repeats, dim=0),
            value.repeat_interleave(repeats, dim=0),
        )
        for key, value in context_kv_blocks
    )


def _prefix_row_ids(
    beam_codes: torch.LongTensor,
    codebook_sizes: tuple[int, ...],
) -> torch.LongTensor:
    prefix_ids = torch.zeros(
        beam_codes.shape[:2],
        dtype=torch.long,
        device=beam_codes.device,
    )
    for level in range(beam_codes.size(-1)):
        prefix_ids = (
            prefix_ids * codebook_sizes[level] + beam_codes[..., level]
        )
    return prefix_ids


def _reorder_past_key_values(
    past_key_values,
    parent_indices: torch.LongTensor,
    current_beams: int,
):
    batch_offsets = (
        torch.arange(parent_indices.size(0), device=parent_indices.device)
        * current_beams
    )
    flat_indices = (
        parent_indices + batch_offsets.unsqueeze(1)
    ).reshape(-1)
    return tuple(
        (
            key.index_select(0, flat_indices),
            value.index_select(0, flat_indices),
        )
        for key, value in past_key_values
    )


@torch.inference_mode()
def constrained_sid_beam_search(
    model,
    context_inputs: Mapping[str, torch.Tensor],
    prefix_index: SidPrefixIndex,
    beam_size: int = EVALUATION_TOP_K,
    use_kv_cache: bool = True,
) -> tuple[torch.LongTensor, torch.Tensor]:
    """Generate valid raw SID codes with vectorized prefix constraints."""
    if beam_size < EVALUATION_TOP_K:
        raise ValueError(
            f"beam_size must be at least {EVALUATION_TOP_K}"
        )
    codebook_sizes = tuple(int(size) for size in model.config.codebook_sizes)
    if codebook_sizes != prefix_index.codebook_sizes:
        raise ValueError("model and SID prefix index codebook sizes differ")
    if not context_inputs:
        raise ValueError("context_inputs must not be empty")

    first = next(iter(context_inputs.values()))
    batch_size = first.size(0)
    if any(value.size(0) != batch_size for value in context_inputs.values()):
        raise ValueError("all context tensors must have the same batch size")

    device = first.device
    offsets = sid_level_offsets(codebook_sizes)
    allowed_masks = prefix_index.allowed_mask_tensors(device)
    cache_enabled = (
        use_kv_cache
        and callable(getattr(model, "encode_context", None))
        and callable(getattr(model, "decode_with_context", None))
    )
    context_cache_by_beams = {}
    past_key_values = None
    if cache_enabled:
        context_kv_blocks, context_attention_mask = model.encode_context(
            context_inputs
        )
        context_cache_by_beams[1] = (
            context_kv_blocks,
            context_attention_mask,
        )

    beam_codes = torch.empty(
        batch_size,
        1,
        0,
        dtype=torch.long,
        device=device,
    )
    beam_tokens = torch.full(
        (batch_size, 1, 1),
        BOS_ID,
        dtype=torch.long,
        device=device,
    )
    beam_scores = torch.zeros(batch_size, 1, device=device)

    for level, codebook_size in enumerate(codebook_sizes):
        current_beams = beam_codes.size(1)
        if cache_enabled:
            if current_beams not in context_cache_by_beams:
                context_cache_by_beams[current_beams] = (
                    _repeat_context_cache(
                        context_kv_blocks,
                        current_beams,
                    ),
                    (
                        None
                        if context_attention_mask is None
                        else context_attention_mask.repeat_interleave(
                            current_beams,
                            dim=0,
                        )
                    ),
                )
            expanded_kv, expanded_mask = context_cache_by_beams[current_beams]
            outputs = model.decode_with_context(
                target_input_ids=beam_tokens[:, :, -1:].reshape(
                    batch_size * current_beams,
                    1,
                ),
                context_kv_blocks=expanded_kv,
                context_attention_mask=expanded_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        else:
            expanded_context = {
                name: values.repeat_interleave(current_beams, dim=0)
                for name, values in context_inputs.items()
            }
            outputs = model(
                target_input_ids=beam_tokens.view(
                    batch_size * current_beams,
                    -1,
                ),
                **expanded_context,
            )
        start = offsets[level]
        code_logits = outputs.logits[:, -1, start : start + codebook_size]
        code_logits = code_logits.view(
            batch_size,
            current_beams,
            codebook_size,
        )

        prefix_ids = _prefix_row_ids(beam_codes, codebook_sizes)
        allowed_mask = allowed_masks[level].index_select(
            0,
            prefix_ids.reshape(-1),
        ).view(batch_size, current_beams, codebook_size)
        if torch.any(~allowed_mask.any(dim=-1)):
            raise ValueError("a generated SID prefix has no valid continuation")

        log_probabilities = F.log_softmax(
            code_logits.masked_fill(~allowed_mask, float("-inf")),
            dim=-1,
        )
        candidate_scores = (
            beam_scores.unsqueeze(-1) + log_probabilities
        ).view(batch_size, -1)
        finite_candidates = torch.isfinite(candidate_scores).sum(dim=1)
        if torch.any(finite_candidates < beam_size):
            raise ValueError(
                "fewer valid SID paths than the requested beam size"
            )

        beam_scores, candidate_indices = torch.topk(
            candidate_scores,
            k=beam_size,
            dim=-1,
        )
        parent_indices = torch.div(
            candidate_indices,
            codebook_size,
            rounding_mode="floor",
        )
        next_codes = candidate_indices.remainder(codebook_size)
        if cache_enabled and level + 1 < len(codebook_sizes):
            if outputs.past_key_values is None:
                raise ValueError("model did not return decoder KV cache")
            past_key_values = _reorder_past_key_values(
                outputs.past_key_values,
                parent_indices,
                current_beams,
            )

        selected_codes = torch.gather(
            beam_codes,
            dim=1,
            index=parent_indices.unsqueeze(-1).expand(
                -1,
                -1,
                beam_codes.size(-1),
            ),
        )
        beam_codes = torch.cat(
            [selected_codes, next_codes.unsqueeze(-1)],
            dim=-1,
        )
        selected_tokens = torch.gather(
            beam_tokens,
            dim=1,
            index=parent_indices.unsqueeze(-1).expand(
                -1,
                -1,
                beam_tokens.size(-1),
            ),
        )
        next_tokens = next_codes + offsets[level]
        beam_tokens = torch.cat(
            [selected_tokens, next_tokens.unsqueeze(-1)],
            dim=-1,
        )

    return beam_codes, beam_scores
