import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
import numpy as np
import math
from argparse import ArgumentParser

from gpt2 import GPT2
from utils import quantize


def _to_sequence(x):
    """shape batch of images for input into GPT2 model"""
    x = x.view(x.shape[0], -1)  # flatten images into sequences
    x = x.transpose(0, 1).contiguous()  # to shape [seq len, batch]
    return x


class ImageGPT(pl.LightningModule):
    def __init__(
        self,
        centroids,
        embed_dim=16,
        num_heads=2,
        num_layers=8,
        num_pixels=28,
        num_vocab=16,
        num_classes=10,
        classify=False,
        learning_rate=3e-3,
        steps=10_000,
        warmup_steps=500,
        **kwargs,
    ):
        super(ImageGPT, self).__init__()
        self.save_hyperparameters()
        self.gpt = GPT2(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_positions=num_pixels * num_pixels,
            num_vocab=num_vocab,
            num_classes=num_classes,
        )

        self.centroids = nn.Parameter(
            torch.from_numpy(np.load(centroids)), requires_grad=False
        )
        self.criterion = nn.CrossEntropyLoss()
        self.classify = classify
        self.learning_rate = learning_rate
        self.steps = steps
        self.warmup_steps = warmup_steps

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--embed_dim", type=int, default=16)
        parser.add_argument("--num_heads", type=int, default=2)
        parser.add_argument("--num_layers", type=int, default=8)
        parser.add_argument("--num_pixels", type=int, default=28)
        parser.add_argument("--num_vocab", type=int, default=16)
        parser.add_argument("--num_classes", type=int, default=10)
        parser.add_argument("--classify", action="store_true", default=False)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--learning_rate", type=float, default=3e-3)
        parser.add_argument("--steps", type=int, default=25_000)
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.gpt.parameters(), lr=self.learning_rate)

        # no learning rate schedule for fine-tuning
        if self.classify:
            return optimizer

        scheduler = {
            "scheduler": LambdaLR(
                optimizer, learning_rate_schedule(self.warmup_steps, self.steps)
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.gpt(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        x = quantize(x, self.centroids)
        x = _to_sequence(x)

        if self.classify:
            # TODO: joint loss
            clf_logits = self.gpt(x, classify=True)
            loss = self.criterion(clf_logits, y)
        else:
            logits = self.gpt(x)
            loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1))

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x = quantize(x, self.centroids)
        x = _to_sequence(x)

        if self.classify:
            clf_logits = self.gpt(x, classify=True)
            loss = self.criterion(clf_logits, y)
            _, preds = torch.max(clf_logits, 1)
            correct = preds == y
            return {"val_loss": loss, "correct": correct}
        else:
            logits = self.gpt(x)
            loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1))
            return {"val_loss": loss}

    def validation_epoch_end(self, outs):
        avg_loss = torch.stack([x["val_loss"] for x in outs]).mean()
        logs = {"val_loss": avg_loss}
        if self.classify:
            correct = torch.cat([x["correct"] for x in outs])
            logs["val_acc"] = correct.sum().float() / correct.shape[0]
        return {"val_loss": avg_loss, "log": logs}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outs):
        result = self.validation_epoch_end(outs)

        # replace valid stats with test stats becuase we are reusing function
        result["log"]["test_loss"] = result["log"].pop("val_loss")
        result["test_loss"] = result.pop("val_loss")
        if self.hparams.classify:
            result["log"]["test_acc"] = result["log"].pop("val_acc")
        return result


def learning_rate_schedule(warmup_steps, total_steps):
    """Linear warmup for warmup_steps, with cosine annealing to 0 at total_steps"""

    def learning_rate_fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return learning_rate_fn
