import torch


def squared_euclidean_distance(a, b):
    b = torch.transpose(b, 0, 1)
    a2 = torch.sum(torch.square(a), dim=1, keepdims=True)
    b2 = torch.sum(torch.square(b), dim=0, keepdims=True)
    ab = torch.matmul(a, b)
    d = a2 - 2 * ab + b2
    return d


def quantize(x, centroids):
    b, c, h, w = x.shape
    # [B, C, H, W] => [B, H, W, C]
    x = x.permute(0, 2, 3, 1).contiguous()
    x = x.view(-1, c)  # flatten to pixels
    d = squared_euclidean_distance(x, centroids)
    x = torch.argmin(d, 1)
    x = x.view(b, h, w)
    return x


def unquantize(x, centroids):
    return centroids[x]
