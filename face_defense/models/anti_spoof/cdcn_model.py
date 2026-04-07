import torch
import torch.nn as nn
import torch.nn.functional as F


class CDCLayer(nn.Module):
    # Central Difference Convolution Layer

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, theta=0.7):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.theta = theta

    def forward(self, x):
        # Standard convolution
        out_normal = self.conv(x)

        if abs(self.theta) < 1e-8:
            return out_normal

        # Central difference term
        kernel_diff = self.conv.weight.sum(dim=[2, 3], keepdim=True)
        out_diff = F.conv2d(x, kernel_diff, stride=self.conv.stride, padding=0)

        return out_normal - self.theta * out_diff


class CDCBlock(nn.Module):
    # CDC + BatchNorm + ReLU block

    def __init__(self, in_ch, out_ch, theta=0.7):
        super().__init__()
        self.cdc = CDCLayer(in_ch, out_ch, kernel_size=3, stride=1, padding=1, theta=theta)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.cdc(x)))


class CDCN(nn.Module):
    # CDCN: Central Difference Convolutional Network for Face Anti-Spoofing

    def __init__(self, in_channels=3, theta=0.7):
        super().__init__()

        # Low-level features
        self.low1 = CDCBlock(in_channels, 64, theta)
        self.low2 = CDCBlock(64, 128, theta)
        self.low3 = CDCBlock(128, 196, theta)

        # Mid-level features
        self.mid1 = CDCBlock(196, 128, theta)
        self.mid2 = CDCBlock(128, 64, theta)
        self.mid3 = CDCBlock(64, 128, theta)

        # High-level features
        self.high1 = CDCBlock(128, 128, theta)
        self.high2 = CDCBlock(128, 64, theta)
        self.high3 = CDCBlock(64, 128, theta)

        # Depth map output (1 channel)
        self.depth_final = nn.Sequential(
            CDCBlock(128, 64, theta),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Low-level
        x = self.low1(x)
        x = self.low2(x)
        x = self.low3(x)

        # Mid-level
        x = self.downsample(x)
        x = self.mid1(x)
        x = self.mid2(x)
        x = self.mid3(x)

        # High-level
        x = self.downsample(x)
        x = self.high1(x)
        x = self.high2(x)
        x = self.high3(x)

        # Upsample back
        x = self.upsample(x)
        x = self.upsample(x)

        # Depth map prediction
        depth_map = self.depth_final(x)

        return depth_map
