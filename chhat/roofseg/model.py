"""Tiny U-Net for rooftop segmentation. CPU-friendly by design.

4-level, ~1.9M params, 256x256 input. Written from scratch (standard U-Net
blocks), no pretrained backbone — this is OUR model, trained on OUR data.

v2 (2026-10-11): deepened from 3 to 4 levels for the final accuracy push
(corrected labels + stronger augmentation). Same public API (TinyUNet(),
count_params) so train/inference code is unchanged.
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        x4 = self.d4(self.pool(x3))
        u = self.up3(x4)
        u = self.c3(torch.cat([u, x3], dim=1))
        u = self.up2(u)
        u = self.c2(torch.cat([u, x2], dim=1))
        u = self.up1(u)
        u = self.c1(torch.cat([u, x1], dim=1))
        return self.out(u)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
