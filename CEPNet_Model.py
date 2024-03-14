import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from MobileNetV2 import mobilenet_v2
from torch.nn import Parameter

from ceam import *


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



class PCorrM(nn.Module):
    def __init__(self, all_channel=24, all_dim=128):
        super(PCorrM, self).__init__()
        self.channel = all_channel
        self.dim = all_dim * all_dim
        self.conv1 = DSConv3x3(all_channel, all_channel, stride=1)
        self.conv2 = DSConv3x3(all_channel, all_channel, stride=1)

    def forward(self, exemplar, query):  # exemplar: f4, query: f3
        fea_size = query.size()[2:]
        exemplar = F.interpolate(exemplar, size=fea_size, mode="bilinear", align_corners=True)
        all_dim = fea_size[0] * fea_size[1]
        exemplar_flat = exemplar.view(-1, self.channel, all_dim)  # N,C1,H,W -> N,C1,H*W
        query_flat = query.view(-1, self.channel, all_dim)  # N,C2,H,W -> N,C2,H*W


        query_corr =query_flat  # batchsize x dim x num, N,C2,H*W

        query_corr_t = torch.transpose(query_corr, 1, 2).contiguous()  #B ,H*W,C2

        A = torch.bmm(query_corr_t, exemplar_flat)  # : N,H*W,C2 x N,C1,H*W = N,H*W,H*W

        A1 = F.softmax(A.clone(), dim=2)  # N,H*W,H*W dim=2 is row-wise norm. Sr
        B = F.softmax(torch.transpose(A, 1, 2), dim=2)  # N,C1,C2 column-wise norm. Sc
        query_att = torch.bmm(query_flat,A1).contiguous()
        exemplar_att = torch.bmm(exemplar_flat,B).contiguous()

        exemplar_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])  # N,C1,H*W -> N,C1,H,W
        exemplar_out = self.conv1(exemplar_att + exemplar)

        query_att = query_att.view(-1, self.channel, fea_size[0], fea_size[1])  # N,C2,H*W -> N,C2,H,W
        query_out = self.conv1(query_att + query)

        return exemplar_out, query_out



class DSMM(nn.Module):
    def __init__(self, channel4=96, channel3=32):
        super(DSMM, self).__init__()

        self.fuse4 = convbnrelu(channel4, channel4, k=1, s=1, p=0, relu=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.smooth4 = DSConv3x3(channel4, channel4, stride=1, dilation=1)  # 96channel-> 96channel

        self.fuse3 = convbnrelu(channel3, channel3, k=1, s=1, p=0, relu=True)
        self.smooth3 = DSConv3x3(channel3, channel4, stride=1, dilation=1)  # 32channel-> 96channel
        self.PositionCorrelation = PCorrM(channel4, 32)

    def forward(self, x4,x3):  # x4:96*18*18 k4:96*5*5; x3:32*36*36 k3:32*5*5

        # Pconv
        x4_all = self.fuse4(x4)
        x4_smooth = self.smooth4(self.upsample2(x4_all))
        # Pconv
        x3_all = self.fuse3(x3)
        x3_smooth = self.smooth3(x3_all)

        # Channel-wise Correlation
        x3_out, x4_out = self.PositionCorrelation(x3_smooth, x4_smooth)

        return torch.cat([x3_out, x4_out], 1)  # (96*2)*32*32


class SalHead(nn.Module):
    def __init__(self, in_channel):
        super(SalHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 1, 1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.conv(x)


class prediction_decoder(nn.Module):
    def __init__(self, channel5=320, channel34=192, channel12=48):
        super(prediction_decoder, self).__init__()
        # 9*9
        self.decoder5 = nn.Sequential(
            DSConv3x3(channel5, channel5, stride=1),
            DSConv3x3(channel5, channel5, stride=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),  # 36*36
            DSConv3x3(channel5, channel34, stride=1)
        )
        self.s5 = SalHead(channel34)  # 36*36

        # 36*36
        self.decoder34 = nn.Sequential(
            DSConv3x3(channel34 * 2, channel34, stride=1),
            DSConv3x3(channel34, channel34, stride=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),  # 144*144
            DSConv3x3(channel34, channel12, stride=1)
        )
        self.s34 = SalHead(channel12)  # 144*144

        # 144*144
        self.decoder12 = nn.Sequential(
            DSConv3x3(channel12 * 2, channel12, stride=1),
            DSConv3x3(channel12, channel12, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 288*288
            DSConv3x3(channel12, channel12, stride=1)
        )
        self.s12 = SalHead(channel12)

    def forward(self, x5, x34, x12):
        x5_decoder = self.decoder5(x5)
        s5 = self.s5(x5_decoder)

        x34_decoder = self.decoder34(torch.cat([x5_decoder, x34], 1))
        s34 = self.s34(x34_decoder)

        x12_decoder = self.decoder12(torch.cat([x34_decoder, x12], 1))
        s12 = self.s12(x12_decoder)

        return s12, s34, s5


class CEPNet(nn.Module):
    def __init__(self, pretrained=True, channel=128):
        super(CEPNet, self).__init__()
        # Backbone model
        self.backbone = mobilenet_v2(pretrained)
        # input 256*256*3
        # conv1 128*128*16
        # conv2 64*64*24
        # conv3 32*32*32
        # conv4 16*16*96
        # conv5 8*8*320


        self.dsmm = DSMM(96, 32)
        self.ecam = CEAM(16, 24)

        self.prediction_decoder = prediction_decoder(320, 192, 48)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # generate backbone features
        conv1, conv2, conv3, conv4, conv5 = self.backbone(input)

        # conv34 is f_dsmm
        conv34 = self.dsmm(conv4,conv3)
        # conv12 is f_ecam
        conv12 = self.ecam(conv1, conv2)

        s12, s34, s5 = self.prediction_decoder(conv5, conv34, conv12)

        s5_up = self.upsample8(s5)
        s34_up = self.upsample2(s34)

        return s12, s34_up, s5_up, self.sigmoid(s12), self.sigmoid(s34_up), self.sigmoid(s5_up)


if __name__ == "__main__":
    from thop import profile

    input = torch.randn(1, 3, 224, 224)

    model = CEPNet()
    # model.load_state_dict(torch.load(
    #     r'D:\Paper\object detection paper\salient object detection\F3Net-master\F3Net-master\src\out\model-32'))
    flops, params = profile(model, inputs=(input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
