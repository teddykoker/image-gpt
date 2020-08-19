import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader, random_split

DATASETS = {
    "mnist": datasets.MNIST,
    "fmnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
}


def dataloaders(dataset, batch_size, datapath="data"):
    train_ds = DATASETS[dataset](
        datapath, train=True, download=True, transform=ToTensor()
    )

    test_ds = DATASETS[dataset](
        datapath, train=False, download=True, transform=ToTensor()
    )

    # TODO paper uses 90/10 split for every dataset besides ImageNet (96/4)
    train_size = int(0.9 * len(train_ds))

    # reproducable split
    train_ds, valid_ds = random_split(
        train_ds,
        [train_size, len(train_ds) - train_size],
        generator=torch.Generator().manual_seed(0),
    )

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=8)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=8)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=8)
    return train_dl, valid_dl, test_dl
