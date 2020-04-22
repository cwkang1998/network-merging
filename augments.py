import numpy as np
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
import imgaug.augmenters as iaa

# Gaussian Noise
def apply_gaussian(device, data, std=1, mean=0):
    return data + torch.randn(data.size()).to(device) * std + mean


# Poisson Noise
def apply_poisson(device, data, rate=0.5):
    m = Poisson(rate)
    noise = m.sample(data.size()).to(device)
    noise = torch.clamp(noise, 0, 1)
    return data + noise


def apply_hflip(device, data):
    images = []
    transform = Compose([RandomHorizontalFlip(p=1)])
    for d in data:
        image = ToPILImage()(d)
        image = transform(image)
        image = ToTensor()(image)
        images.append(image)
    return torch.stack(images)


def apply_vflip(device, data):
    images = []
    transform = Compose([RandomVerticalFlip(p=1)])
    for d in data:
        image = ToPILImage()(d)
        image = transform(image)
        image = ToTensor()(image)
        images.append(image)
    return torch.stack(images)


def apply_random_crop(device, data, size=32):
    images = []
    transform = Compose([RandomResizedCrop(size=size)])
    for d in data:
        image = ToPILImage()(d)
        image = transform(image)
        image = ToTensor()(image)
        images.append(image)
    return torch.stack(images)


def apply_sharpen(device, data, alpha=0.5):
    images = []
    sharpen = iaa.Sharpen(alpha=0.5)
    for d in data:
        image = ToPILImage()(d)
        image = sharpen.augment_image(np.array(image))
        image = ToTensor()(image)
        images.append(image)
    return torch.stack(images)
