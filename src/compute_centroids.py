import numpy as np
from pathlib import Path
import argparse

from torchvision.transforms import ToTensor
from sklearn.cluster import KMeans, MiniBatchKMeans

from data import DATASETS


def download(dataset, datapath):
    train_ds = DATASETS[dataset](
        datapath, train=True, download=True, transform=ToTensor()
    )
    train_x = np.stack([x.numpy() for x, _ in train_ds])
    train_x = train_x.transpose(0, 2, 3, 1)  # put channel dimension last
    return train_x


def find_centroids(train_x, num_clusters=16, batch_size=1024):
    pixels = train_x.reshape(-1, train_x.shape[-1])
    if batch_size:
        kmeans = MiniBatchKMeans(
            n_clusters=num_clusters, random_state=0, batch_size=batch_size
        ).fit(pixels)
    else:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
    return kmeans.cluster_centers_


def main(args):
    datapath = Path("data")
    datapath.mkdir(exist_ok=True)

    train_x = download(args.dataset, datapath)
    centroids = find_centroids(train_x, args.num_clusters, args.batch_size)
    np.save(datapath / f"{args.dataset}_centroids.npy", centroids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=DATASETS.keys(), default="mnist")
    parser.add_argument("--num_clusters", default=16, type=int)
    parser.add_argument(
        "--batch_size",
        default=1024,
        type=int,
        help="batch size for mini batch kmeans to quantize images",
    )
    args = parser.parse_args()
    main(args)
