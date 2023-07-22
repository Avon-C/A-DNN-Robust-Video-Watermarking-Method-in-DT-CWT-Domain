# Author: Aspertaine
# Date: 2022/7/6 20:56

import torch
import torch.nn as nn
import torch.nn.functional as F
import config

def pool(wm, batch_size, host_size, wm_size):
    time = int(host_size / wm_size)
    line = 0
    ex_wm = torch.zeros(batch_size, wm_size, wm_size).to(config.device)
    for _ in range(time):
        row = 0
        for _ in range(time):
            ex_wm += wm[:, row:row+wm_size, line:line+wm_size]
            row += wm_size
        line += wm_size
    return ex_wm / (time * time)


class spatial_temporal_features(nn.Module):
    def __init__(self):
        super(spatial_temporal_features, self).__init__()

        self.spatial_features = nn.Sequential(
            nn.Conv3d(4, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
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

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, 4 * out_channels, kernel_size=1)

        self.bn2 = nn.BatchNorm3d(4 * out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(4 * out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out

class Dense_Decoder(nn.Module):
    def __init__(self):
        super(Dense_Decoder, self).__init__()
        self.feature = spatial_temporal_features()

        self.dense_1 = Bottleneck(64, 64)
        self.dense_2 = Bottleneck(128, 64)
        self.dense_3 = Bottleneck(192, 64)

        self.r_1 = nn.Sequential(
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.AdaptiveAvgPool3d((9, None, None)),

            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.AdaptiveAvgPool3d((6, None, None)),

            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.AdaptiveAvgPool3d((3, None, None)),

            nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.AdaptiveAvgPool3d((1, None, None))
        )

    def forward(self, x):

        H = x.size(3)
        W = x.size(4)

        out = self.feature(x)

        out = self.dense_1(out)
        out = self.dense_2(out)
        out = self.dense_3(out)

        out = self.r_1(out)
        out = out.view(-1, H, W)
        out = pool(out, x.size(0), H, 32)
        return out


if __name__ == '__main__':
    test = torch.randn(2, 4, 9, 192, 192).to(config.device)
    decoder = Dense_Decoder().to(config.device)
    ans = decoder(test)
    print(ans.shape)
