import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.nn import Parameter
from splitatt import *

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.PReLU(out_channel))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)


class CEAM(nn.Module):
    def __init__(self, channel1=16, channel2=24):
        super(CEAM, self).__init__()

        self.smooth1 = DSConv3x3(channel1, channel2, stride=1, dilation=1)  # 16channel-> 24channel

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.smooth2 = DSConv3x3(channel2, channel2, stride=1, dilation=1)  # 24channel-> 24channel

        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)

        self.conv_1 = nn.Conv2d(channel2, channel2, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(channel2)
        self.conv_2 = nn.Conv2d(channel2, channel2, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(channel2)

        self.sigmoid = nn.Sigmoid()

        self.ChannelCorrelation = SplitAttn(in_channels=48)

    def forward(self, x1, x2):  # x1 16*144*14; x2 24*72*72

        x1_1 = self.smooth1(x1)
        x2_1 = self.smooth2(self.upsample2(x2))

        fuse = x1_1 * x2_1
        fuse_edge=self.avg_pool(fuse)

        edge1 = x1_1-fuse_edge
        edge2 = x2_1-fuse_edge

        weight1 = self.sigmoid(self.bn1(self.conv_1(edge1)))
        weight2 = self.sigmoid(self.bn2(self.conv_2(edge2)))

        out1 = weight1 * x1_1 + x1_1
        out2 = weight2 * x2_1 + x2_1


        # Channel-wise Correlation
        out = self.ChannelCorrelation(torch.cat([out1, out2], 1))

        return out # (24*2)*144*144

if __name__ == '__main__':
    net=CEAM()
    x=torch.randn(8,16,112,112)
    y=torch.randn(8,24,56,56)
    out=net(x,y)




