"""Train the custom LazyOneRec model on local KuaiRand standard logs."""

from __future__ import annotations

import argparse
import math
import time

import torch
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

    def set_initial_training_values(
        self,
        args,
        dataloader,
        total_train_batch_size,
    ):
        values = list(
            super().set_initial_training_values(
                args,
                dataloader,
                total_train_batch_size,
            )
        )
        len_dataloader = values[5]
        if len_dataloader is not None and args.max_steps < 0:
            updates_per_epoch = max(
                math.ceil(
                    len_dataloader / args.gradient_accumulation_steps
                ),
                1,
            )
            values[1] = updates_per_epoch
            values[6] = math.ceil(args.num_train_epochs * updates_per_epoch)
        return tuple(values)

    def get_batch_samples(self, epoch_iterator, _num_batches, device):
        # Trainer 4.51 derives the final request from example count rather than
        # micro-batch count. Always request a full accumulation window and let
        # StopIteration return the smaller final window.
        return super().get_batch_samples(
            epoch_iterator,
            self.args.gradient_accumulation_steps,
            device,
        )

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
        if not getattr(self, "_logged_day_batches", False):
            for date in batch_sampler.dates:
                n_samples = len(batch_sampler.indices_by_date[date])
                tail = n_samples % per_device_batch_size
                print(
                    f"[day] date={date} samples={n_samples:,} "
                    f"micro_batches={batch_sampler.batch_counts[date]:,} "
                    f"tail_batch={tail or per_device_batch_size}"
                )
            self._logged_day_batches = True

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
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_started = time.perf_counter()
    print(
        f"[config] data_root={args.data_root} "
        f"sid_artifact={args.sid_artifact} output_dir={args.output_dir}"
    )
    print(
        f"[config] sample={args.sample} min_history={args.min_history} "
        f"seed={args.seed} position_encoding={args.position_encoding}"
    )
    print(
        f"[runtime] torch={torch.__version__} "
        f"cuda_available={torch.cuda.is_available()} "
        f"mps_available={torch.backends.mps.is_available()}"
    )
    if torch.cuda.is_available():
        print(
            f"[runtime] cuda_device={torch.cuda.current_device()} "
            f"name={torch.cuda.get_device_name(torch.cuda.current_device())} "
            f"bf16_supported={torch.cuda.is_bf16_supported()}"
        )

    stage_started = time.perf_counter()
    artifact = SemanticIDArtifact.load(args.sid_artifact)
    print(
        f"[stage] load_sid_artifact elapsed="
        f"{time.perf_counter() - stage_started:.2f}s"
    )
    stage_started = time.perf_counter()
    corpus = KuaiRandExposureCorpus(args.data_root)
    print(
        f"[stage] build_exposure_corpus elapsed="
        f"{time.perf_counter() - stage_started:.2f}s"
    )
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
    stage_started = time.perf_counter()
    sample_indices = build_exposure_sample_indices(
        corpus=corpus,
        sid_artifact=artifact,
        min_history=args.min_history,
        warmup_days=args.warmup_days,
        test_days=args.test_days,
    )
    print(
        f"[stage] build_sample_indices elapsed="
        f"{time.perf_counter() - stage_started:.2f}s"
    )
    stage_started = time.perf_counter()
    train_dataset = KuaiRandNextExposureSidDataset(
        corpus=corpus,
        sid_artifact=artifact,
        sample_index=sample_indices["train"],
        history_lengths=history_lengths,
        sample=args.sample,
        seed=args.seed,
    )
    print(
        f"[stage] build_train_dataset elapsed="
        f"{time.perf_counter() - stage_started:.2f}s"
    )
    if not train_dataset:
        raise ValueError(
            "no training samples: SID artifact does not cover KuaiRand targets"
        )

    total_exposures = sum(len(sequence.gid_ids) for sequence in corpus.sequences)
    positive_exposures = sum(
        int(sequence.positive_target.sum()) for sequence in corpus.sequences
    )
    train_dates = sorted(set(int(date) for date in sample_indices["train"].dates))
    test_dates = sorted(set(int(date) for date in sample_indices["test"].dates))
    train_date_range = (
        f"{train_dates[0]}..{train_dates[-1]}" if train_dates else "empty"
    )
    test_date_range = (
        f"{test_dates[0]}..{test_dates[-1]}" if test_dates else "empty"
    )
    print(
        f"[data] users={len(corpus.sequences):,} "
        f"exposures={total_exposures:,} positives={positive_exposures:,} "
        f"positive_rate={positive_exposures / total_exposures:.4%}"
    )
    print(
        f"[data] gid_vocab={corpus.num_gid_embeddings:,} "
        f"warmup_days={args.warmup_days} test_days={args.test_days} "
        f"train_candidates={len(sample_indices['train']):,} "
        f"train_used={len(train_dataset):,} "
        f"test={len(sample_indices['test']):,}"
    )
    print(
        f"[split] train_dates={train_date_range} "
        f"({len(train_dates)}) test_dates={test_date_range} "
        f"({len(test_dates)})"
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

    stage_started = time.perf_counter()
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
    print(
        f"[stage] initialize_model elapsed="
        f"{time.perf_counter() - stage_started:.2f}s"
    )
    named_parameters = list(model.named_parameters())
    n_params = sum(parameter.numel() for _, parameter in named_parameters)
    trainable_params = sum(
        parameter.numel()
        for _, parameter in named_parameters
        if parameter.requires_grad
    )
    gid_params = model.context_feature_embedding.gid_embedding.weight.numel()
    qformer_params = sum(
        parameter.numel()
        for name, parameter in named_parameters
        if "sequence_qformers" in name
    )
    decoder_params = sum(
        parameter.numel()
        for name, parameter in named_parameters
        if name.startswith("layers.")
    )
    print(
        f"[model] params={n_params:,} trainable={trainable_params:,} "
        f"gid={gid_params:,} qformer={qformer_params:,} "
        f"decoder={decoder_params:,} "
        f"fp32_weights={n_params * 4 / 2**30:.2f}GiB"
    )

    if (
        args.micro_batch_size <= 0
        or args.batch_size < args.micro_batch_size
        or args.batch_size % args.micro_batch_size
    ):
        raise ValueError(
            "batch_size must be a positive multiple of micro_batch_size"
        )
    accumulation = args.batch_size // args.micro_batch_size
    batch_preview = DayBatchSampler(
        sample_dates=train_dataset.sample_dates,
        batch_size=args.micro_batch_size,
        seed=args.seed,
    )
    optimizer_steps = math.ceil(len(batch_preview) / accumulation)
    print(
        f"[optimization] micro_batch={args.micro_batch_size} "
        f"gradient_accumulation={accumulation} "
        f"effective_batch={args.batch_size} "
        f"micro_batches={len(batch_preview):,} "
        f"optimizer_steps={optimizer_steps:,}"
    )
    print(
        f"[optimization] lr={args.learning_rate} "
        f"weight_decay={args.weight_decay} warmup_steps={args.warmup_steps} "
        f"logging_steps={args.logging_steps} bf16={args.bf16}"
    )
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
        dataloader_num_workers=args.num_workers,
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_prefetch_factor=2 if args.num_workers > 0 else None,
        accelerator_config={"even_batches": False},
    )
    print(
        f"[runtime] trainer_device={training_args.device} "
        f"world_size={training_args.world_size} "
        f"dataloader_workers={training_args.dataloader_num_workers}"
    )
    trainer = DayOrderedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=KuaiRandCollator(history_lengths=history_lengths),
    )
    training_started = time.perf_counter()
    train_output = trainer.train()
    print(
        f"[stage] train elapsed={time.perf_counter() - training_started:.2f}s "
        f"global_steps={trainer.state.global_step:,} "
        f"train_loss={train_output.training_loss:.6f}"
    )
    save_started = time.perf_counter()
    trainer.save_model(args.output_dir)
    print(
        f"[stage] save_model elapsed={time.perf_counter() - save_started:.2f}s"
    )
    print(
        f"[done] model saved to {args.output_dir} "
        f"total_elapsed={time.perf_counter() - run_started:.2f}s"
    )


if __name__ == "__main__":
    main()
