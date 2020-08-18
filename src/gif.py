import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image

from module import ImageGPT
from prepare_data import unquantize

from moviepy.editor import ImageSequenceClip


def generate(model, context, length, num_samples=1, temperature=1.0):

    output = context.unsqueeze(-1).repeat_interleave(
        num_samples, dim=-1
    )  # add batch so shape [seq len, batch]

    frames = []
    pad = torch.zeros(1, num_samples, dtype=torch.long).cuda()  # to pad prev output
    with torch.no_grad():
        for _ in tqdm(range(length), leave=False):
            logits = model(torch.cat((output, pad), dim=0))
            logits = logits[-1, :, :] / temperature
            probs = F.softmax(logits, dim=-1)
            pred = torch.multinomial(probs, num_samples=1).transpose(1, 0)
            output = torch.cat((output, pred), dim=0)
            frames.append(output.cpu().numpy().transpose())
    return frames


def main(args):
    model = ImageGPT.load_from_checkpoint(args.checkpoint).gpt.cuda()
    model.eval()

    centroids = np.load(args.centroids)

    context = torch.zeros(0, dtype=torch.long).cuda()

    frames = generate(model, context, 28 * 28, num_samples=args.rows * args.cols)

    pad_frames = []
    for frame in frames:
        pad = ((0, 0), (0, 28 * 28 - frame.shape[1]))
        pad_frames.append(np.pad(frame, pad_width=pad))

    pad_frames = np.stack(pad_frames)
    f, n, _ = pad_frames.shape
    pad_frames = pad_frames.reshape(f, args.rows, args.cols, 28, 28)
    pad_frames = pad_frames.swapaxes(2, 3).reshape(f, 28 * args.rows, 28 * args.cols)
    pad_frames = np.squeeze(unquantize(pad_frames, centroids))
    pad_frames = pad_frames[..., np.newaxis] * np.ones(3) * 256
    pad_frames = pad_frames.astype(np.uint8)

    clip = ImageSequenceClip(list(pad_frames)[:: args.downsample], fps=args.fps).resize(
        args.scale
    )
    clip.write_gif("out.gif", fps=args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--centroids", default="data/mnist_centroids.npy")
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--downsample", type=int, default=5)
    args = parser.parse_args()
    main(args)
