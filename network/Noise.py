# Author: Aspertaine
# Date: 2022/6/11 22:29
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from random import random, randint
import random as ra
import math
import config

def random_float(min_, max_):
    return ((np.random.rand() * (max_ - min_) + min_) * 100 // 4 * 4) / 100

class Scale_noise(nn.Module):

    def __init__(self, min_pct=0.6, max_pct=1.4, interpolation_method='bilinear'):
        super(Scale_noise, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct
        self.interpolation_method = interpolation_method

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        video_num, _, _, height, width = x.size()
        resize_ratio_H = random_float(self.min_pct, self.max_pct)
        resize_ratio_W = random_float(self.min_pct, self.max_pct)
        height, width = (resize_ratio_H * height // 4) * 4, (resize_ratio_W * width // 4) * 4

        list_video = []
        for v in range(video_num):
            list_video.append(F.interpolate(
                x[v, :, :, :, :].permute(1, 0, 2, 3),
                size=(int(height), int(width)),
                mode=self.interpolation_method,
                align_corners=True
            ).permute(1, 0, 2, 3))
        scaled_videos = torch.stack(list_video, dim=0)
        return scaled_videos.permute(0, 2, 3, 4, 1)

class Scale_decode_train(nn.Module):

    def __init__(self, min_pct=0.6, max_pct=1.4, interpolation_method='bilinear'):
        super(Scale_decode_train, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct
        self.interpolation_method = interpolation_method

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        video_num, _, _, height, width = x.size()
        list_video = []
        for v in range(video_num):
            list_video.append(F.interpolate(
                x[v, :, :, :, :].permute(1, 0, 2, 3),
                size=(512, 512),
                mode=self.interpolation_method,
                align_corners=True
            ).permute(1, 0, 2, 3))
        scaled_videos = torch.stack(list_video, dim=0)
        return scaled_videos.permute(0, 2, 3, 4, 1)
class Scale_decode_eval(nn.Module):

    def __init__(self, min_pct=0.6, max_pct=1.4, interpolation_method='bilinear'):
        super(Scale_decode_eval, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct
        self.interpolation_method = interpolation_method

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        video_num, _, _, height, width = x.size()
        list_video = []
        for v in range(video_num):
            list_video.append(F.interpolate(
                x[v, :, :, :, :].permute(1, 0, 2, 3),
                size=(768, 768),
                mode=self.interpolation_method,
                align_corners=True
            ).permute(1, 0, 2, 3))
        scaled_videos = torch.stack(list_video, dim=0)
        return scaled_videos.permute(0, 2, 3, 4, 1)
class Scale_decode_test(nn.Module):

    def __init__(self, min_pct=0.6, max_pct=1.4, interpolation_method='bilinear'):
        super(Scale_decode_test, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct
        self.interpolation_method = interpolation_method

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        video_num, _, _, height, width = x.size()
        list_video = []
        for v in range(video_num):
            list_video.append(F.interpolate(
                x[v, :, :, :, :].permute(1, 0, 2, 3),
                size=(384, 384),
                mode=self.interpolation_method,
                align_corners=True
            ).permute(1, 0, 2, 3))
        scaled_videos = torch.stack(list_video, dim=0)
        return scaled_videos.permute(0, 2, 3, 4, 1)
class Scale_decode_big(nn.Module):

    def __init__(self, min_pct=0.6, max_pct=1.4, interpolation_method='bilinear'):
        super(Scale_decode_big, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct
        self.interpolation_method = interpolation_method

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        video_num, _, _, height, width = x.size()
        list_video = []
        for v in range(video_num):
            list_video.append(F.interpolate(
                x[v, :, :, :, :].permute(1, 0, 2, 3),
                size=(1024, 1024),
                mode=self.interpolation_method,
                align_corners=True
            ).permute(1, 0, 2, 3))
        scaled_videos = torch.stack(list_video, dim=0)
        return scaled_videos.permute(0, 2, 3, 4, 1)

class Crop(nn.Module):

    def __init__(self, min_pct=0.4, max_pct=1.0):
        super(Crop, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct

    def _pct(self):
        return self.min_pct + random() * (self.max_pct - self.min_pct)

    def forward(self, frames):
        frames = frames.permute(0, 4, 1, 2, 3)
        _, _, _, height, width = frames.size()
        r = self._pct()

        dx = int(r * width)
        dy = int(r * height)

        dx, dy = (dx // 4) * 4, (dy // 4) * 4
        x = randint(0, width - dx - 1)
        y = randint(0, height - dy - 1)

        crop_mask = frames.clone()
        crop_mask[:, :, :, :, :] = 0.0
        crop_mask[:, :, :, y:y + dy, x:x + dx] = 1.0
        return (frames * crop_mask).permute(0, 2, 3, 4, 1)

def Del_frames(x):
    list_ = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    num = randint(4, 9)
    r = sorted(ra.sample(list_, num))
    # r = sorted(ra.sample(list_, 8))
    # r = sorted(ra.sample(list_, 6))
    # r = sorted(ra.sample(list_, 4))
    # r = sorted(ra.sample(list_, 2))
    # r = sorted(ra.sample(list_, 1))
    return x[:, r, :, :, :]

def Swap_frame(x):
    list_ = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    r = sorted(ra.sample(list_, 2))
    first = r[0]
    second = r[1]
    temp = x[:, [first], :, :, :]
    x[:, [first], :, :, :] = x[:, [second], :, :, :]
    x[:, [second], :, :, :] = temp
    return x

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

def gaussian_blur(x):
    kernel = get_gaussian_kernel().to(config.device)
    x = x.permute(0, 1, 4, 2, 3) 
    a = []
    for i in range(x.size(0)):
        a.append(kernel(x[i, :, :, :, :]))
    return torch.stack(a, dim=0).permute(0, 1, 3, 4, 2)

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.04, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0.).to(config.device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x




