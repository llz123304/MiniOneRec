"""Train the custom LazyOneRec model on local KuaiRand standard logs."""

from __future__ import annotations

import argparse

from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, set_seed

from ..data.kuairand import (
    KuaiRandCollator,
    KuaiRandExposureCorpus,
    KuaiRandNextExposureSidDataset,
    build_exposure_sample_indices,
)
from ..data.schema import (
    DEFAULT_GID_SEQUENCE_LENGTHS,
    DEFAULT_QFORMER_QUERY_COUNTS,
    raw_context_length,
    total_context_length,
)
from ..data.day_batch_sampler import DayBatchSampler
from ..model import LazyOneRecConfig, KuaiRandLazyOneRecForCausalLM
from ..sid.artifact import SemanticIDArtifact


class DayOrderedTrainer(Trainer):
    """Trainer whose training batches never mix target dates."""

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer requires a train_dataset")
        sample_dates = getattr(self.train_dataset, "sample_dates", None)
        if sample_dates is None:
            raise ValueError(
                "DayOrderedTrainer requires train_dataset.sample_dates"
            )
        if int(self.args.world_size) != 1:
            raise ValueError(
                "partial per-day batches currently require single-process training"
            )

        per_device_batch_size = int(
            getattr(self, "_train_batch_size", self.args.train_batch_size)
        )
        batch_sampler = DayBatchSampler(
            sample_dates=sample_dates,
            batch_size=per_device_batch_size,
            seed=self.args.seed,
        )

        print(
            f"[batching] dates={len(batch_sampler.dates)} "
            f"range={batch_sampler.dates[0]}..{batch_sampler.dates[-1]} "
            f"micro_batches={len(batch_sampler)} "
            "keep_partial_day_batches=True"
        )

        dataloader_kwargs = {
            "batch_sampler": batch_sampler,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if self.args.dataloader_num_workers > 0:
            dataloader_kwargs["persistent_workers"] = getattr(
                self.args, "dataloader_persistent_workers", False
            )
            prefetch_factor = getattr(
                self.args, "dataloader_prefetch_factor", None
            )
            if prefetch_factor is not None:
                dataloader_kwargs["prefetch_factor"] = prefetch_factor

        return self.accelerator.prepare(
            DataLoader(self.train_dataset, **dataloader_kwargs)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="lazy_onerec/KuaiRand-1K",
        help="KuaiRand variant root containing the data/ directory",
    )
    parser.add_argument("--sid-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument(
        "--click-history-length",
        type=int,
        default=DEFAULT_GID_SEQUENCE_LENGTHS["click"],
    )
    parser.add_argument(
        "--long-view-history-length",
        type=int,
        default=DEFAULT_GID_SEQUENCE_LENGTHS["long_view"],
    )
    parser.add_argument(
        "--like-history-length",
        type=int,
        default=DEFAULT_GID_SEQUENCE_LENGTHS["like"],
    )
    parser.add_argument(
        "--deep-interact-history-length",
        type=int,
        default=DEFAULT_GID_SEQUENCE_LENGTHS["deep_interact"],
    )
    parser.add_argument(
        "--hate-history-length",
        type=int,
        default=DEFAULT_GID_SEQUENCE_LENGTHS["hate"],
    )
    parser.add_argument("--warmup-days", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--gid-dim", type=int, default=64)
    parser.add_argument("--user-id-dim", type=int, default=128)
    parser.add_argument("--categorical-dim", type=int, default=8)
    parser.add_argument("--continuous-dim", type=int, default=16)
    parser.add_argument("--duration-dim", type=int, default=8)
    parser.add_argument("--qformer-layers", type=int, default=1)
    parser.add_argument(
        "--click-query-tokens",
        type=int,
        default=DEFAULT_QFORMER_QUERY_COUNTS["click"],
    )
    parser.add_argument(
        "--long-view-query-tokens",
        type=int,
        default=DEFAULT_QFORMER_QUERY_COUNTS["long_view"],
    )
    parser.add_argument(
        "--long-view-duration-query-tokens",
        type=int,
        default=DEFAULT_QFORMER_QUERY_COUNTS["long_view_duration"],
    )
    parser.add_argument(
        "--like-query-tokens",
        type=int,
        default=DEFAULT_QFORMER_QUERY_COUNTS["like"],
    )
    parser.add_argument(
        "--deep-interact-query-tokens",
        type=int,
        default=DEFAULT_QFORMER_QUERY_COUNTS["deep_interact"],
    )
    parser.add_argument(
        "--hate-query-tokens",
        type=int,
        default=DEFAULT_QFORMER_QUERY_COUNTS["hate"],
    )
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-context-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument(
        "--per-token-qkv",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--per-token-ffn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--kv-sharing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--kv-share-every", type=int, default=2)
    parser.add_argument(
        "--position-encoding", choices=["rope", "learned"], default="rope"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    artifact = SemanticIDArtifact.load(args.sid_artifact)
    corpus = KuaiRandExposureCorpus(args.data_root)
    history_lengths = {
        "click": args.click_history_length,
        "long_view": args.long_view_history_length,
        "like": args.like_history_length,
        "deep_interact": args.deep_interact_history_length,
        "hate": args.hate_history_length,
    }
    qformer_query_counts = {
        "click": args.click_query_tokens,
        "long_view": args.long_view_query_tokens,
        "long_view_duration": args.long_view_duration_query_tokens,
        "like": args.like_query_tokens,
        "deep_interact": args.deep_interact_query_tokens,
        "hate": args.hate_query_tokens,
    }
    sample_indices = build_exposure_sample_indices(
        corpus=corpus,
        sid_artifact=artifact,
        min_history=args.min_history,
        warmup_days=args.warmup_days,
        test_days=args.test_days,
    )
    datasets = {
        split: KuaiRandNextExposureSidDataset(
            corpus=corpus,
            sid_artifact=artifact,
            sample_index=sample_indices[split],
            history_lengths=history_lengths,
            sample=args.sample if split == "train" else -1,
            seed=args.seed,
        )
        for split in ("train", "test")
    }
    if not datasets["train"]:
        raise ValueError(
            "no training samples: SID artifact does not cover KuaiRand targets"
        )

    print(
        f"[data] users={len(corpus.sequences)} "
        f"gid_vocab={corpus.num_gid_embeddings} "
        f"warmup_days={args.warmup_days} test_days={args.test_days} "
        f"train={len(datasets['train'])} test={len(datasets['test'])}"
    )
    print(
        f"[user] categorical={corpus.user_features.categorical.shape[1]} "
        f"continuous={corpus.user_features.continuous.shape[1]}"
    )
    print(
        f"[context] raw={raw_context_length(history_lengths)} "
        f"compressed={total_context_length(history_lengths, qformer_query_counts)} "
        f"qformer_queries={qformer_query_counts}"
    )
    print(
        f"[sid] items={artifact.n_items} "
        f"sizes={list(artifact.codebook_sizes)} "
        f"collision_rate={artifact.collision_rate:.6f}"
    )

    config = LazyOneRecConfig.from_codebook_sizes(
        artifact.codebook_sizes,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        n_context_layers=args.n_context_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        use_per_token_qkv=args.per_token_qkv,
        use_per_token_ffn=args.per_token_ffn,
        kv_sharing=args.kv_sharing,
        kv_share_every=args.kv_share_every,
        max_context_len=total_context_length(
            history_lengths,
            qformer_query_counts,
        ),
        position_encoding=args.position_encoding,
    )
    model = KuaiRandLazyOneRecForCausalLM(
        config,
        num_gid_embeddings=corpus.num_gid_embeddings,
        user_categorical_cardinalities=(
            corpus.user_features.categorical_cardinalities
        ),
        gid_dim=args.gid_dim,
        user_id_dim=args.user_id_dim,
        categorical_dim=args.categorical_dim,
        continuous_dim=args.continuous_dim,
        duration_dim=args.duration_dim,
        history_lengths=history_lengths,
        qformer_query_counts=qformer_query_counts,
        qformer_layers=args.qformer_layers,
    )
    n_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"[model] params={n_params:,}")

    if (
        args.micro_batch_size <= 0
        or args.batch_size < args.micro_batch_size
        or args.batch_size % args.micro_batch_size
    ):
        raise ValueError(
            "batch_size must be a positive multiple of micro_batch_size"
        )
    accumulation = args.batch_size // args.micro_batch_size
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=accumulation,
        num_train_epochs=1,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        save_strategy="epoch",
        eval_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        accelerator_config={"even_batches": False},
    )
    trainer = DayOrderedTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        data_collator=KuaiRandCollator(history_lengths=history_lengths),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    if datasets["test"]:
        test_metrics = trainer.evaluate(
            eval_dataset=datasets["test"],
            metric_key_prefix="test",
        )
        trainer.save_metrics("test", test_metrics)
    print(f"[done] model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
