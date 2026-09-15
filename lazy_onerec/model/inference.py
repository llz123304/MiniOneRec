"""Non-cached constrained beam search for SID generation."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from ..sid.evaluation import EVALUATION_TOP_K, SidPrefixIndex
from ..sid.layout import BOS_ID, sid_level_offsets


@torch.inference_mode()
def constrained_sid_beam_search(
    model,
    context_inputs: Mapping[str, torch.Tensor],
    prefix_index: SidPrefixIndex,
    beam_size: int = EVALUATION_TOP_K,
) -> tuple[torch.LongTensor, torch.Tensor]:
    """Generate valid raw SID codes without reusing context or decoder caches."""
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

        allowed_mask = torch.zeros_like(code_logits, dtype=torch.bool)
        for batch_index in range(batch_size):
            for beam_index in range(current_beams):
                prefix = beam_codes[batch_index, beam_index].tolist()
                allowed = prefix_index.allowed(prefix)
                if not allowed:
                    raise ValueError(
                        f"no valid continuation for SID prefix {prefix}"
                    )
                allowed_mask[
                    batch_index,
                    beam_index,
                    torch.tensor(allowed, device=device),
                ] = True

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
