import numpy as np
from pathlib import Path
import argparse
import urllib.request
import shutil
import gzip
import struct

MNIST_URLS = [
    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
]

FASHION_URLS = [
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
]


def read_label_file(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_image_file(path):
    with open(path, "rb") as f:
        magic, length, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        return np.fromfile(f, dtype=np.uint8).reshape(length, rows, cols)


def download(url, filepath):
    with urllib.request.urlopen(url) as response, open(filepath, "wb") as f:
        shutil.copyfileobj(response, f)


def gunzip(inpath, outpath):
    with gzip.open(inpath, "rb") as f_in:
        with open(outpath, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    inpath.unlink()  # remove original file


def quantize(x):
    """
    Quantize image. In original OpenAI paper, they use KNN to cluster rgb images
    512 clusters (essentially 9-bit). Since we're using grey scale images, we can
    simply divide to quantize
    """

    # divide by 17 = (255 / 15) to convert 8 bit ints to 4 bit
    x = x // 17
    return x


def unquantize(x):
    """
    See quantize. This will need to be modified for three channel images
    """
    return x * 17


def main(args):
    datapath = Path("data")
    datapath.mkdir(exist_ok=True)

    urls = FASHION_URLS if args.fashion else MNIST_URLS

    # Download MNIST if not downloaded
    for url in urls:
        filename = url.rsplit("/")[-1]
        if not (datapath / filename[:-3]).exists():
            print(f"Downloading {url} ...")
            download(url, datapath / filename)
            gunzip(datapath / filename, datapath / filename[:-3])

    train_x = quantize(read_image_file("data/train-images-idx3-ubyte"))
    train_y = read_label_file("data/train-labels-idx1-ubyte")
    test_x = quantize(read_image_file("data/t10k-images-idx3-ubyte"))
    test_y = read_label_file("data/t10k-labels-idx1-ubyte")

    np.save(datapath / "train_x.npy", train_x.astype(int))
    np.save(datapath / "train_y.npy", train_y.astype(int))
    np.save(datapath / "test_x.npy", test_x.astype(int))
    np.save(datapath / "test_y.npy", test_y.astype(int))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fashion", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
