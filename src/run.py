import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np
import argparse
import yaml


from image_gpt import ImageGPT
from data import dataloaders


def train(args):

    with open(args.config, "rb") as f:
        config = yaml.safe_load(f)

    # experiment name
    name = f"{config['name']}_{args.dataset}"

    if args.pretrained is not None:
        model = ImageGPT.load_from_checkpoint(args.pretrained)
        # potentially modify model for finetuning
        model.learning_rate = config["learning_rate"]
        model.classify = config["classify"]
    else:
        model = ImageGPT(centroids=args.centroids, **config)

    train_dl, valid_dl, test_dl = dataloaders(args.dataset, config["batch_size"])
    logger = pl_loggers.TensorBoardLogger("logs", name=name)

    if config["classify"]:
        # classification
        # stop early for best validation accuracy for finetuning
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_acc", patience=3, mode="max"
        )
        checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_acc")
        trainer = pl.Trainer(
            max_steps=config["steps"],
            gpus=config["gpus"],
            accumulate_grad_batches=config["accumulate_grad_batches"],
            precision=config["precision"],
            early_stop_callback=early_stopping,
            checkpoint_callback=checkpoint,
            logger=logger,
        )

    else:
        # pretraining
        checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss")
        trainer = pl.Trainer(
            max_steps=config["steps"],
            gpus=config["gpus"],
            precision=config["precision"],
            accumulate_grad_batches=config["accumulate_grad_batches"],
            checkpoint_callback=checkpoint,
            logger=logger,
        )

    trainer.fit(model, train_dl, valid_dl)


def test(args):
    with open(args.config, "rb") as f:
        config = yaml.safe_load(f)
    trainer = pl.Trainer(gpus=config["gpus"])
    _, _, test_dl = dataloaders(args.dataset, config["batch_size"])
    model = ImageGPT.load_from_checkpoint(args.checkpoint)
    trainer.test(model, test_dataloaders=test_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="mnist")

    subparsers = parser.add_subparsers()
    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--pretrained", type=str, default=None)
    parser_train.add_argument("config", type=str)
    parser_train.set_defaults(func=train)

    parser_test = subparsers.add_parser("test")
    parser_test.add_argument("checkpoint", type=str)
    parser_test.add_argument("config", type=str)
    parser_test.set_defaults(func=test)

    args = parser.parse_args()
    args.centroids = f"data/{args.dataset}_centroids.npy"

    args.func(args)
