# Author: Aspertaine
# Date: 2022/6/26 12:04

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class spatial_temporal_features(nn.Module):
    def __init__(self):
        super(spatial_temporal_features, self).__init__()
        self.spatial_features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.temporal_features = nn.Sequential(
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3, 7, 7), padding=(1, 3, 3)),
        )

    def forward(self, x):
        spatial = self.spatial_features(x)
        spatial_temporal = self.temporal_features(spatial)
        out = F.relu(spatial_temporal, inplace=True)
        return out

class Down(nn.Module):
    def __init__(self):
        super(Down, self).__init__()
        self.max_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((9, 256, 256)),
            nn.Conv3d(64, 96, kernel_size=1),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((9, 128, 128)),
            nn.Conv3d(96, 96, kernel_size=1),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.max_pool(x)
        return out

class Up(nn.Module):
    def __init__(self):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(96, 64, kernel_size=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((9, 256, 256)),

            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(64, 3, kernel_size=1, padding=1),
            nn.BatchNorm3d(3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((9, 512, 512))
        )

    def forward(self, x):
        out = self.up(x)
        return out

class h264_(nn.Module):
    def __init__(self):
        super(h264_, self).__init__()
        self.features = spatial_temporal_features()
        self.down = Down()
        self.up = Up()

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3) 
        H = x.size(3)
        W = x.size(4)
        out = self.features(x)
        print(out.shape)
        out = self.down(out)
        print(out.shape)
        out = self.up(out)
        print(out.shape)
        recover_scale = F.interpolate(
            out,
            size=(9, H, W),
            mode='trilinear',
            align_corners=True
        )
        return (x + recover_scale).permute(0, 2, 3, 4, 1)


