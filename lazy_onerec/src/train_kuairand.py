"""Train the custom LazyOneRec model on local KuaiRand standard logs."""

from __future__ import annotations

import argparse

from transformers import Trainer, TrainingArguments, set_seed

from ..data import KuaiRandClickCorpus, KuaiRandCollator, KuaiRandNextSidDataset
from ..model import LazyOneRecConfig, KuaiRandLazyOneRecForCausalLM
from ..sid import SemanticIDArtifact


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
    parser.add_argument("--max-history", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--gid-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-context-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--kv-share-every", type=int, default=2)
    parser.add_argument(
        "--position-encoding", choices=["rope", "learned"], default="rope"
    )
    parser.add_argument("--num-epochs", type=float, default=10)
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
    corpus = KuaiRandClickCorpus(args.data_root)
    datasets = {
        split: KuaiRandNextSidDataset(
            corpus=corpus,
            sid_artifact=artifact,
            split=split,
            min_history=args.min_history,
            max_history=args.max_history,
            sample=args.sample if split == "train" else -1,
            seed=args.seed,
        )
        for split in ("train", "valid", "test")
    }
    if not datasets["train"]:
        raise ValueError(
            "no training samples: SID artifact does not cover KuaiRand targets"
        )

    print(
        f"[data] users={len(corpus.sequences)} "
        f"gid_vocab={corpus.num_gid_embeddings} "
        f"train={len(datasets['train'])} "
        f"valid={len(datasets['valid'])} test={len(datasets['test'])}"
    )
    print(
        f"[sid] items={artifact.n_items} "
        f"sizes={list(artifact.codebook_sizes)} "
        f"collision_rate={artifact.collision_rate:.6f}"
    )

    config = LazyOneRecConfig.from_codebook_sizes(
        artifact.codebook_sizes,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_context_layers=args.n_context_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        kv_share_every=args.kv_share_every,
        max_context_len=args.max_history,
        position_encoding=args.position_encoding,
    )
    model = KuaiRandLazyOneRecForCausalLM(
        config,
        num_gid_embeddings=corpus.num_gid_embeddings,
        gid_dim=args.gid_dim,
    )
    n_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"[model] params={n_params:,}")

    accumulation = max(1, args.batch_size // args.micro_batch_size)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=accumulation,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid"],
        data_collator=KuaiRandCollator(),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[done] model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
