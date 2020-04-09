import torch
from torch.distributions.poisson import Poisson
from torchvision.transforms import (
    Compose,
    ToPILImage,
    ToTensor,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomResizedCrop,
)


# Gaussian Noise
def apply_gaussian(device, data, std=1, mean=0):
    return data + torch.randn(data.size()).to(device) * std + mean


# Poisson Noise
def apply_poisson(device, data, rate=0.5):
    m = Poisson(rate)
    noise = m.sample(data.size()).to(device)
    noise = torch.clamp(noise, 0, 1)
    return data + noise


def apply_hflip(data):
    images = []
    transform = Compose([RandomHorizontalFlip(p=1)])
    for d in data:
        image = ToPILImage()(d)
        image = transform(image)
        image = ToTensor()(image)
        images.append(image)
    return torch.stack(images)


def apply_vflip(data):
    images = []
    transform = Compose([RandomVerticalFlip(p=1)])
    for d in data:
        image = ToPILImage()(d)
        image = transform(image)
        image = ToTensor()(image)
        images.append(image)
    return torch.stack(images)


def apply_random_crop(data, size):
    images = []
    transform = Compose([RandomResizedCrop(size=size)])
    for d in data:
        image = ToPILImage()(d)
        image = transform(image)
        image = ToTensor()(image)
        images.append(image)
    return torch.stack(images)
