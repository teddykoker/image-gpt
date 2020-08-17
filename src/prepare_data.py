import numpy as np
from pathlib import Path
import argparse

from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.cluster import KMeans, MiniBatchKMeans

DATASETS = {
    "mnist": datasets.MNIST,
    "fmnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
}


def download(dataset, datapath):
    train_ds = DATASETS[dataset](
        datapath, train=True, download=True, transform=ToTensor()
    )
    test_ds = DATASETS[dataset](
        datapath, train=False, download=True, transform=ToTensor()
    )
    train_x = np.stack([x.numpy() for x, _ in train_ds])
    train_y = np.stack([y for _, y in train_ds])
    test_x = np.stack([x.numpy() for x, _ in test_ds])
    test_y = np.stack([y for _, y in test_ds])

    train_x = train_x.transpose(0, 2, 3, 1)  # put channel dimension last
    test_x = test_x.transpose(0, 2, 3, 1)  # put channel dimension last

    return train_x, train_y, test_x, test_y


def find_centroids(train_x, num_clusters=8, batch_size=1024):
    pixels = train_x.reshape(-1, train_x.shape[-1])
    if batch_size:
        kmeans = MiniBatchKMeans(
            n_clusters=num_clusters, random_state=0, batch_size=batch_size
        ).fit(pixels)
    else:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
    return kmeans.cluster_centers_


def squared_euclidean_distance(a, b):
    b = np.transpose(b)
    a2 = np.sum(np.square(a), axis=1, keepdims=True)
    b2 = np.sum(np.square(b), axis=0, keepdims=True)
    ab = np.matmul(a, b)
    d = a2 - 2 * ab + b2
    return d


def quantize(x, centroids):
    """
    Quantize image. In original OpenAI paper, they use KNN to cluster rgb images
    512 clusters (essentially 9-bit).
    """
    x_shape = x.shape
    x = x.reshape(-1, x_shape[-1])
    d = squared_euclidean_distance(x, centroids)
    x = np.argmin(d, 1)
    x = x.reshape(x_shape[:-1])
    return x


def unquantize(x, clusters):
    """
    Unquantize image
    """
    x = clusters[x]
    return x


def main(args):
    datapath = Path("data")
    datapath.mkdir(exist_ok=True)

    train_x, train_y, test_x, test_y = download(args.dataset, datapath)

    centroids = find_centroids(train_x, args.num_clusters, args.batch_size)

    train_x = quantize(train_x, centroids)
    test_x = quantize(test_x, centroids)

    np.save(datapath / f"{args.dataset}_centroids.npy", centroids)
    np.save(datapath / f"{args.dataset}_train_x.npy", train_x.astype(int))
    np.save(datapath / f"{args.dataset}_train_y.npy", train_y.astype(int))
    np.save(datapath / f"{args.dataset}_test_x.npy", test_x.astype(int))
    np.save(datapath / f"{args.dataset}_test_y.npy", test_y.astype(int))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=DATASETS.keys(), default="mnist")
    parser.add_argument("--num_clusters", default=8, type=int)
    parser.add_argument(
        "--batch_size",
        default=1024,
        type=int,
        help="batch size for mini batch KNN to quantize images",
    )
    args = parser.parse_args()
    main(args)
