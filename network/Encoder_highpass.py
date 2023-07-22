
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.WatermarkProcess_highpass import watermark_expend
from network import Attention as Att
import config
from ptflops import get_model_complexity_info
from thop import profile

class spatial_temporal_features(nn.Module):
    def __init__(self):
        super(spatial_temporal_features, self).__init__()

        self.spatial_features = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.temporal_features = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 7, 7), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        spatial = self.spatial_features(x)
        spatial_temporal = self.temporal_features(spatial)
        return spatial_temporal


def Up(H, W, wm):
    time_H = int(H / 32.0)
    time_W = int(W / 32.0)
    ori = wm
    for _ in range(time_H - 1):
        wm = torch.cat([wm, ori], 3)
    ori = wm
    for _ in range(time_W - 1):
        wm = torch.cat([wm, ori], 4)
    return wm


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

class Dense_Encoder(nn.Module):
    def __init__(self, wm_expand, wm_factor):
        super(Dense_Encoder, self).__init__()
        self.wm_expand = wm_expand
        self.wm_factor = wm_factor

        self.feature = spatial_temporal_features()
        self.attention_mask = Att.ResBlock_CBAM(64, 16)

        self.dense_1 = Bottleneck(64 + wm_expand[0], 64)
        self.dense_2 = Bottleneck(2 * 64 + wm_expand[0] + wm_expand[1], 64)
        self.dense_3 = Bottleneck(3 * 64 + wm_expand[0] + wm_expand[1] + wm_expand[2], 64)

        self.r = nn.Sequential(
            nn.BatchNorm3d(259),
            nn.ReLU(inplace=True),
            nn.Conv3d(259, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 2, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        )

    def forward(self, x, watermark, batch_size, frame_to_process):
        H = x.size(3)
        W = x.size(4)
        
        out = self.feature(x)
        mask = self.attention_mask(out)

        watermark = watermark.view(1, 1, 1, 32, 32)
        watermark_128 = Up(H, W, watermark)

        w_1 = watermark_expend(watermark_128, batch_size, self.wm_expand[0], frame_to_process, H, W)
        out = self.dense_1(torch.cat([out, w_1], dim=1))

        w_2 = watermark_expend(watermark_128, batch_size, self.wm_expand[1], frame_to_process, H, W)
        out = self.dense_2(torch.cat([out, w_2], dim=1))

        w_3 = watermark_expend(watermark_128, batch_size, self.wm_expand[2], frame_to_process, H, W)
        out = self.dense_3(torch.cat([out, w_3], dim=1))

        out = self.r(out)
        out = x + self.wm_factor * mask * out

        return out


