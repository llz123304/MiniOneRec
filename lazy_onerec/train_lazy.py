"""Training entry point for the Lazy Decoder-Only recommender (from scratch).

Wires the pieces together:
    index.json --> SidCodec + item_codes        (sid_codec)
    train/valid CSV --> LazySidSeqDataset        (data_lazy)  two-stream samples
    LazyOneRecConfig.from_codec(codec) --> model (from scratch, no pretrained)
    HF Trainer with LazyTwoStreamCollator        --> train

Example:
    python -m lazy_onerec.train_lazy \
        --index_path data/Amazon/index/Industrial_and_Scientific.index.json \
        --train_file data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
        --valid_file data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
        --output_dir output/lazy_industrial \
        --codebook_sizes 256 256 256 \
        --n_layers 6 --n_context_layers 2 --d_model 768 \
        --position_encoding rope

Evaluation/generation is a separate step (needs the context-KV cache in
modeling_lazy_onerec.prepare_inputs_for_generation, still TODO).
"""

import argparse

import transformers
from transformers import Trainer, TrainingArguments, set_seed

from .sid_codec import build_codec_from_index
from .data_lazy import LazySidSeqDataset, LazyTwoStreamCollator
from .configuration_lazy_onerec import LazyOneRecConfig
from .modeling_lazy_onerec import LazyOneRecForCausalLM


def parse_args():
    p = argparse.ArgumentParser(description="Train the Lazy Decoder-Only recommender from scratch")
    # data
    p.add_argument("--index_path", required=True, help="path to <dataset>.index.json")
    p.add_argument("--train_file", required=True, help="training CSV")
    p.add_argument("--valid_file", default=None, help="validation CSV (optional)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--sample", type=int, default=-1, help="subsample N rows (-1 = all)")
    p.add_argument("--max_context_len", type=int, default=3000)
    # SID / vocab
    p.add_argument("--codebook_sizes", type=int, nargs="+", default=[256, 256, 256],
                   help="per-level codebook size K (must match SID generation)")
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

    # 1) SID codec + item -> codes mapping (K is an explicit hyperparameter)
    codec, item_codes = build_codec_from_index(args.index_path, codebook_sizes=args.codebook_sizes)
    print(f"[codec] levels={codec.n_levels} sizes={codec.codebook_sizes} "
          f"vocab_size={codec.vocab_size} items={len(item_codes)}")

    # 2) datasets (two-stream: context / target / labels)
    train_ds = LazySidSeqDataset(args.train_file, codec, item_codes,
                                 max_context_len=args.max_context_len,
                                 sample=args.sample, seed=args.seed)
    eval_ds = None
    if args.valid_file:
        eval_ds = LazySidSeqDataset(args.valid_file, codec, item_codes,
                                    max_context_len=args.max_context_len, seed=args.seed)
    print(f"[data] train={len(train_ds)}" + (f" valid={len(eval_ds)}" if eval_ds else ""))

    # 3) model from scratch, vocabulary aligned to the codec
    config = LazyOneRecConfig.from_codec(
        codec,
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
    print(f"[model] params={n_params:,} heads={len(model.level_heads)} "
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
