import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image

from image_gpt import ImageGPT
from utils import quantize, unquantize
from data import dataloaders

import matplotlib.pyplot as plt


def sample(model, context, length, num_samples=1, temperature=1.0):

    output = context.unsqueeze(-1).repeat_interleave(
        num_samples, dim=-1
    )  # add batch so shape [seq len, batch]

    pad = torch.zeros(1, num_samples, dtype=torch.long).cuda()  # to pad prev output
    with torch.no_grad():
        for _ in tqdm(range(length), leave=False):
            logits = model(torch.cat((output, pad), dim=0))
            logits = logits[-1, :, :] / temperature
            probs = F.softmax(logits, dim=-1)
            pred = torch.multinomial(probs, num_samples=1).transpose(1, 0)
            output = torch.cat((output, pred), dim=0)
    return output


def make_figure(rows, centroids):
    figure = np.stack(rows, axis=0)
    print(figure.shape)
    rows, cols, h, w = figure.shape
    figure = unquantize(figure.swapaxes(1, 2).reshape(h * rows, w * cols), centroids)
    figure = (figure * 256).astype(np.uint8)
    return Image.fromarray(np.squeeze(figure))


def main(args):
    model = ImageGPT.load_from_checkpoint(args.checkpoint).gpt.cuda()

    centroids = np.load(args.centroids)
    train_dl, valid_dl, test_dl = dataloaders(args.dataset, 1)

    dl = iter(valid_dl)

    # rows for figure
    rows = []

    for example in tqdm(range(args.num_examples)):
        img, _ = next(dl)
        h, w = img.shape[-2:]

        img = quantize(img, torch.from_numpy(centroids)).numpy()[0]
        seq = img.reshape(-1)

        # first half of image is context
        context = seq[: int(len(seq) / 2)]
        context_img = np.pad(context, (0, int(len(seq) / 2))).reshape(h, w)
        context = torch.from_numpy(context).cuda()

        # predict second half of image
        preds = (
            sample(model, context, int(len(seq) / 2), num_samples=args.num_samples)
            .cpu()
            .numpy()
            .transpose()
        )

        preds = preds.reshape(-1, h, w)

        # combine context, preds, and truth for figure
        rows.append(
            np.concatenate([context_img[None, ...], preds, img[None, ...]], axis=0)
        )

    figure = make_figure(rows, centroids)
    figure.save("figure.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--centroids", default="data/mnist_centroids.npy")
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--num_examples", default=5)
    parser.add_argument("--num_samples", default=5)
    args = parser.parse_args()
    main(args)
