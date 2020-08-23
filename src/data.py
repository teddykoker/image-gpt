import torch
from torchvision import datasets
import torchvision.transforms as T

from torch.utils.data import DataLoader, random_split

DATASETS = {
    "mnist": datasets.MNIST,
    "fmnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
}


def train_transforms(dataset):
    if dataset == "cifar10":
        # "When full-network fine-tuning on CIFAR-10 and CIFAR100, we use the augmentation popularized by Wide Residual
        # Networks: 4 pixels are reflection padded on each side, and
        # a 32 × 32 crop is randomly sampled from the padded image or its horizontal flip"
        return T.Compose(
            [
                T.RandomCrop(32, padding=4, padding_mode="reflect"),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )

    elif dataset == "mnist" or dataset == "fmnist":
        return T.ToTensor()


def test_transforms(dataset):
    return T.ToTensor()


def dataloaders(dataset, batch_size, datapath="data"):
    train_ds = DATASETS[dataset](
        datapath, train=True, download=True, transform=train_transforms(dataset)
    )
    valid_ds = DATASETS[dataset](
        datapath, train=True, download=True, transform=test_transforms(dataset)
    )
    test_ds = DATASETS[dataset](
        datapath, train=False, download=True, transform=test_transforms(dataset)
    )

    # TODO paper uses 90/10 split for every dataset besides ImageNet (96/4)
    train_size = int(0.9 * len(train_ds))

    # reproducable split
    # NOTE: splitting is done twice as datasets have different transforms attributes
    train_ds, _ = random_split(
        train_ds,
        [train_size, len(train_ds) - train_size],
        generator=torch.Generator().manual_seed(0),
    )
    _, valid_ds = random_split(
        valid_ds,
        [train_size, len(valid_ds) - train_size],
        generator=torch.Generator().manual_seed(0),
    )

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=8)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=8)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=8)
    return train_dl, valid_dl, test_dl
