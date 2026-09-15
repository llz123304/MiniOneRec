"""Trainer regressions for partial gradient-accumulation windows."""

import tempfile
import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import TrainingArguments

from lazy_onerec.src.train_kuairand import DayOrderedTrainer


class _ToyDataset(Dataset):
    def __init__(self):
        self.sample_dates = [20220411] * 6

    def __len__(self):
        return len(self.sample_dates)

    def __getitem__(self, index):
        return {
            "feature": torch.tensor([float(index + 1)]),
            "labels": torch.tensor([index]),
        }


class _CountingModel(nn.Module):
    accepts_loss_kwargs = True

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.seen = []
        self.denominators = []

    def forward(self, feature, labels, num_items_in_batch=None):
        self.seen.extend(int(value) for value in labels.detach().view(-1))
        self.denominators.append(int(num_items_in_batch))
        prediction = feature.view(-1) * self.weight
        target = labels.view(-1).to(prediction.dtype)
        loss = (prediction - target).square().sum()
        loss = loss / num_items_in_batch.to(loss.dtype)
        return {"loss": loss}


def _collate(features):
    return {
        "feature": torch.stack([item["feature"] for item in features]),
        "labels": torch.stack([item["labels"] for item in features]),
    }


class DayOrderedTrainerTest(unittest.TestCase):
    def test_final_partial_accumulation_window_is_trained(self):
        with tempfile.TemporaryDirectory() as output_dir:
            arguments = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                num_train_epochs=1,
                learning_rate=1e-3,
                save_strategy="no",
                report_to=[],
                disable_tqdm=True,
                remove_unused_columns=False,
            )
            model = _CountingModel()
            trainer = DayOrderedTrainer(
                model=model,
                args=arguments,
                train_dataset=_ToyDataset(),
                data_collator=_collate,
            )

            trainer.train()

        self.assertEqual(sorted(model.seen), list(range(6)))
        self.assertEqual(trainer.state.global_step, 2)
        self.assertEqual(model.denominators, [4, 4, 4, 4, 2, 2])


if __name__ == "__main__":
    unittest.main()
