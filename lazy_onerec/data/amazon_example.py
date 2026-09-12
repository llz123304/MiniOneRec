"""Amazon/MiniOneRec-format dataset example.

This is not part of the core model contract. It demonstrates how one legacy
CSV format can consume a framework-neutral SemanticIDArtifact. Replace it with
your own KuaiRand/GID sequence-slot pipeline.

Expected legacy CSV columns:
    history_item_id : list-as-string of integer item ids
    item_id         : integer item id (ground-truth next item)
"""

from typing import List, Dict
import ast

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..sid.artifact import SemanticIDArtifact
from ..sid.token_codec import PAD_ID, SidTokenCodec


class AmazonSidSequenceDataset(Dataset):
    """Example adapter from MiniOneRec Amazon CSVs to model-ready fields."""

    def __init__(self, train_file: str, sid_artifact: SemanticIDArtifact,
                 max_context_len: int = 3000, sample: int = -1, seed: int = 0):
        self.codec = SidTokenCodec(sid_artifact.codebook_sizes)
        self.item_codes = sid_artifact.item_codes
        self.max_context_len = max_context_len

        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed).reset_index(drop=True)

        self.samples = [s for i in range(len(self.data))
                        if (s := self._build(self.data.iloc[i])) is not None]

    def _codes_for(self, item_id) -> List[int]:
        return self.item_codes.get(str(item_id))

    def _build(self, row):
        history_ids = ast.literal_eval(str(row["history_item_id"]))
        context_ids: List[int] = []
        for iid in history_ids:
            codes = self._codes_for(iid)
            if codes is None:
                continue  # skip items missing from the index
            context_ids.extend(self.codec.codes_to_ids(codes))
        context_ids = context_ids[-self.max_context_len:]

        target_codes = self._codes_for(row["item_id"])
        if target_codes is None or not context_ids:
            return None
        target_ids = self.codec.target_ids(target_codes)          # [BOS] c0 c1 c2
        labels = [-100] + self.codec.codes_to_ids(target_codes)   # no loss on BOS

        return {
            "context_input_ids": context_ids,
            "target_input_ids": target_ids,
            "labels": labels,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class LazyTwoStreamCollator:
    """Pads the two streams independently.

    - context: right-padded; mask marks valid tokens for cross-attention
    - target : right-padded; labels padded with -100 so pad steps are ignored
    Target length is fixed (BOS + n_levels codes) so padding is usually a no-op.
    """

    def __init__(self, pad_id: int = PAD_ID):
        self.pad_id = pad_id

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        ctx_len = max(len(f["context_input_ids"]) for f in features)
        tgt_len = max(len(f["target_input_ids"]) for f in features)

        ctx_ids, ctx_mask, tgt_ids, labels = [], [], [], []
        for f in features:
            c, t, l = f["context_input_ids"], f["target_input_ids"], f["labels"]
            pc, pt = ctx_len - len(c), tgt_len - len(t)
            ctx_ids.append(c + [self.pad_id] * pc)
            ctx_mask.append([1] * len(c) + [0] * pc)
            tgt_ids.append(t + [self.pad_id] * pt)
            labels.append(l + [-100] * pt)

        return {
            "context_input_ids": torch.tensor(ctx_ids, dtype=torch.long),
            "context_attention_mask": torch.tensor(ctx_mask, dtype=torch.long),
            "target_input_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
