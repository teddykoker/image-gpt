import pytorch_lightning as pl
import pytorch_lightning.logging
import argparse

from module import ImageGPT


def train(args):
    if args.pretrained is not None:
        model = ImageGPT.load_from_checkpoint(args.pretrained)
        # potentially modify model for classification
        model.hparams = args
    else:
        model = ImageGPT(args)

    logger = pl.logging.TensorBoardLogger("logs", name=args.name)

    if args.classify:
        # classification
        # stop early for best validation accuracy for finetuning
        early_stopping = pl.callbacks.EarlyStopping(monitor="val_acc", patience=3)
        checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_acc")
        trainer = pl.Trainer(
            max_steps=args.steps,
            gpus=args.gpus,
            early_stopping_callback=early_stopping,
            checkpoint_callback=checkpoint,
            logger=logger,
        )

    else:
        # pretraining
        checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss")
        trainer = pl.Trainer(
            max_steps=args.steps,
            gpus=args.gpus,
            checkpoint_callback=checkpoint,
            logger=logger,
        )

    trainer.fit(model)


def test(args):
    trainer = pl.Trainer(gpus=args.gpus)
    model = ImageGPT.load_from_checkpoint(args.checkpoint)
    model.prepare_data()
    trainer.test(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = ImageGPT.add_model_specific_args(parser)

    parser.add_argument("--dataset", default="mnist")
    #     parser.add_argument("--train_y", default="data/train_y.npy")
    #     parser.add_argument("--test_x", default="data/test_x.npy")
    #     parser.add_argument("--test_y", default="data/test_y.npy")
    #
    parser.add_argument("--gpus", default="0")
    subparsers = parser.add_subparsers()
    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--pretrained", type=str, default=None)
    parser_train.add_argument("-n", "--name", type=str, required=True)
    parser_train.set_defaults(func=train)

    parser_test = subparsers.add_parser("test")
    parser_test.add_argument("checkpoint", type=str)
    parser_test.set_defaults(func=test)

    args = parser.parse_args()
    #     args.train_x = f"data/{args.dataset}_train_x.npy"
    #     args.train_y = f"data/{args.dataset}_train_y.npy"
    #     args.test_x = f"data/{args.dataset}_test_x.npy"
    #     args.test_y = f"data/{args.dataset}_test_y.npy"

    args.func(args)
