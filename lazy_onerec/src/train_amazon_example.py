"""Legacy Amazon training example, not the project data pipeline.

It consumes a generic ``sid_index.json`` artifact and demonstrates one possible
dataset adapter. Custom KuaiRand/GID sequence data should use a separate script.

Example:
    python -m lazy_onerec.src.train_amazon_example \
        --sid-artifact output/sid_index.json \
        --train_file data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
        --valid_file data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
        --output_dir output/lazy_industrial \
        --n_layers 6 --n_context_layers 2 --d_model 768 \
        --position_encoding rope

Evaluation/generation is a separate step (needs the context-KV cache in
model.modeling.prepare_inputs_for_generation, still TODO).
"""

import argparse

from transformers import Trainer, TrainingArguments, set_seed

from ..data.amazon_example import AmazonSidSequenceDataset, LazyTwoStreamCollator
from ..model import LazyOneRecConfig, LazyOneRecForCausalLM
from ..sid import SemanticIDArtifact


def parse_args():
    p = argparse.ArgumentParser(description="Train the Lazy Decoder-Only recommender from scratch")
    # data
    p.add_argument("--sid-artifact", required=True)
    p.add_argument("--train_file", required=True, help="training CSV")
    p.add_argument("--valid_file", default=None, help="validation CSV (optional)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--sample", type=int, default=-1, help="subsample N rows (-1 = all)")
    p.add_argument("--max_context_len", type=int, default=3000)
    # model
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_context_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--n_kv_heads", type=int, default=2)
    p.add_argument("--kv_share_every", type=int, default=2)
    p.add_argument("--position_encoding", choices=["rope", "learned"], default="rope")
    # training
    p.add_argument("--num_epochs", type=float, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--micro_batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    artifact = SemanticIDArtifact.load(args.sid_artifact)
    print(
        f"[sid] levels={artifact.n_levels} "
        f"sizes={list(artifact.codebook_sizes)} items={artifact.n_items}"
    )

    train_ds = AmazonSidSequenceDataset(
        args.train_file,
        artifact,
        max_context_len=args.max_context_len,
        sample=args.sample,
        seed=args.seed,
    )
    eval_ds = None
    if args.valid_file:
        eval_ds = AmazonSidSequenceDataset(
            args.valid_file,
            artifact,
            max_context_len=args.max_context_len,
            seed=args.seed,
        )
    print(f"[data] train={len(train_ds)}" + (f" valid={len(eval_ds)}" if eval_ds else ""))

    config = LazyOneRecConfig.from_codebook_sizes(
        artifact.codebook_sizes,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_context_layers=args.n_context_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        kv_share_every=args.kv_share_every,
        max_context_len=args.max_context_len,
        position_encoding=args.position_encoding,
    )
    model = LazyOneRecForCausalLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params:,} tied_codebooks={model.n_sid_levels} "
          f"pe={args.position_encoding}")

    # 4) Trainer
    grad_accum = max(1, args.batch_size // args.micro_batch_size)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        report_to=[],
        remove_unused_columns=False,  # our collator consumes the raw dict fields
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=LazyTwoStreamCollator(pad_id=config.pad_token_id),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[done] model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
