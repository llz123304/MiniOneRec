"""Evaluate KuaiRand SID generation with fixed Top-10 metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data.kuairand import (
    KuaiRandCollator,
    KuaiRandExposureCorpus,
    KuaiRandNextExposureSidDataset,
    build_exposure_sample_indices,
)
from ..data.schema import DEFAULT_GID_SEQUENCE_LENGTHS
from ..model import KuaiRandLazyOneRecForCausalLM
from ..model.inference import constrained_sid_beam_search
from ..sid.artifact import SemanticIDArtifact
from ..sid.evaluation import (
    EVALUATION_TOP_K,
    SidPrefixIndex,
    SidRankingMetrics,
)
from ..sid.token_codec import SidTokenCodec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="lazy_onerec/KuaiRand-1K",
    )
    parser.add_argument("--sid-artifact", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-days", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=3)
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=EVALUATION_TOP_K)
    parser.add_argument(
        "--device",
        choices=("cuda", "mps", "cpu"),
        default="cuda",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(name)


def decode_target_codes(
    labels: torch.Tensor,
    codec: SidTokenCodec,
) -> torch.LongTensor:
    target_tokens = labels[:, 1:]
    if target_tokens.size(1) != codec.n_levels:
        raise ValueError("target labels do not match SID levels")
    target_codes = target_tokens.clone()
    for level, offset in enumerate(codec.level_offsets):
        target_codes[:, level] -= offset
        if torch.any(target_codes[:, level] < 0) or torch.any(
            target_codes[:, level] >= codec.codebook_sizes[level]
        ):
            raise ValueError(f"invalid target token at SID level {level}")
    return target_codes


def main() -> None:
    args = parse_args()
    if args.beam_size < EVALUATION_TOP_K:
        raise ValueError(
            f"beam_size must be at least {EVALUATION_TOP_K}"
        )
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    artifact = SemanticIDArtifact.load(args.sid_artifact)
    model = KuaiRandLazyOneRecForCausalLM.from_pretrained(args.checkpoint)
    if tuple(model.config.codebook_sizes) != artifact.codebook_sizes:
        raise ValueError("checkpoint and SID artifact codebook sizes differ")
    model.to(device)
    model.eval()

    corpus = KuaiRandExposureCorpus(args.data_root)
    expected_cardinalities = tuple(
        model.config.user_categorical_cardinalities
    )
    if corpus.user_features.categorical_cardinalities != expected_cardinalities:
        raise ValueError(
            "checkpoint and dataset user feature cardinalities differ"
        )
    history_lengths = dict(
        getattr(
            model.config,
            "history_lengths",
            DEFAULT_GID_SEQUENCE_LENGTHS,
        )
    )
    sample_indices = build_exposure_sample_indices(
        corpus=corpus,
        sid_artifact=artifact,
        min_history=args.min_history,
        warmup_days=args.warmup_days,
        test_days=args.test_days,
    )
    dataset = KuaiRandNextExposureSidDataset(
        corpus=corpus,
        sid_artifact=artifact,
        sample_index=sample_indices["test"],
        history_lengths=history_lengths,
        sample=args.sample,
        seed=args.seed,
    )
    if not dataset:
        raise ValueError("test dataset is empty")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=KuaiRandCollator(history_lengths=history_lengths),
        pin_memory=device.type == "cuda",
    )

    codec = SidTokenCodec(artifact.codebook_sizes)
    prefix_index = SidPrefixIndex(
        artifact.item_codes.values(),
        artifact.codebook_sizes,
    )
    metrics = SidRankingMetrics(prefix_index)
    for batch in tqdm(
        dataloader,
        desc="Evaluating SID HR@10",
        unit="batch",
        dynamic_ncols=True,
    ):
        targets = decode_target_codes(batch.pop("labels"), codec)
        batch.pop("target_input_ids")
        context_inputs = {
            name: tensor.to(device, non_blocking=device.type == "cuda")
            for name, tensor in batch.items()
        }
        predictions, _ = constrained_sid_beam_search(
            model=model,
            context_inputs=context_inputs,
            prefix_index=prefix_index,
            beam_size=args.beam_size,
        )
        metrics.update(
            predictions[:, :EVALUATION_TOP_K],
            targets,
        )

    result = metrics.compute()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    print(f"[done] metrics saved to {output}")


if __name__ == "__main__":
    main()
