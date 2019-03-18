import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channelss, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean) * rgb_range

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=True, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Block(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=True, act=nn.ReLU(True), res_scale=1):

        super(Block, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        if bn: modules_body.append(nn.BatchNorm2d(n_feat))
        modules_body.append(act)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        modules = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                modules.append(conv(n_feat, 4 * n_feat, 3, bias))
                modules.append(nn.PixelShuffle(2))
                if bn: modules.append(nn.BatchNorm2d(n_feat))
                if act: modules.append(act())
        elif scale == 3:
            modules.append(conv(n_feat, 9 * n_feat, 3, bias))
            modules.append(nn.PixelShuffle(3))
            if bn: modules.append(nn.BatchNorm2d(n_feat))
            if act: modules.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*modules)

class MemBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=True, act=nn.ReLU(True), res_scale=1):

        super(MemBlock, self).__init__()

        modules_conv1 = [ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act)]
        modules_conv2 = [ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act)]
        modules_conv3 = [ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act)]
        modules_conv4 = [ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act)]
        modules_conv5 = [ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act)]
        modules_conv6 = [ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act)]
        modules_conv = [conv(6*n_feat, n_feat, kernel_size, bias=bias)]

        self.conv1 = nn.Sequential(*modules_conv1)
        self.conv2 = nn.Sequential(*modules_conv2)
        self.conv3 = nn.Sequential(*modules_conv3)
        self.conv4 = nn.Sequential(*modules_conv4)
        self.conv5 = nn.Sequential(*modules_conv5)
        self.conv6 = nn.Sequential(*modules_conv6)
        self.conv = nn.Sequential(*modules_conv)
        self.res_scale = res_scale

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        c_out = torch.cat([c1,c2,c3,c4,c5,c6],1)
        res = self.conv(c_out).mul(self.res_scale)
        res += x

        return res

class IRBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=True, act=nn.ReLU(True)):

        super(IRBlock, self).__init__()
        modules_irbody = []
        modules_irbody.append(nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=1, bias=True, dilation=1))
        modules_irbody.append(act)

        modules_irbody.append(nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=2, bias=True, dilation=2))
        if bn: modules_irbody.append(nn.BatchNorm2d(n_feat))
        modules_irbody.append(act)

        modules_irbody.append(nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=3, bias=True, dilation=3))
        if bn: modules_irbody.append(nn.BatchNorm2d(n_feat))
        modules_irbody.append(act)

        modules_irbody.append(nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=4, bias=True, dilation=4))
        if bn: modules_irbody.append(nn.BatchNorm2d(n_feat))
        modules_irbody.append(act)

        modules_irbody.append(nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=3, bias=True, dilation=3))
        if bn: modules_irbody.append(nn.BatchNorm2d(n_feat))
        modules_irbody.append(act)

        modules_irbody.append(nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=2, bias=True, dilation=2))
        if bn: modules_irbody.append(nn.BatchNorm2d(n_feat))
        modules_irbody.append(act)

        modules_irbody.append(nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=1, bias=True, dilation=1))

        self.body = nn.Sequential(*modules_irbody)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class IRMemBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=True, act=nn.ReLU(True)):

        super(IRMemBlock, self).__init__()
        conv1 = [nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=1, bias=True, dilation=1)]
        conv1.append(act)

        conv2 = [nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=2, bias=True, dilation=2)]
        if bn: conv2.append(nn.BatchNorm2d(n_feat))
        conv2.append(act)

        conv3 = [nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=3, bias=True, dilation=3)]
        if bn: conv3.append(nn.BatchNorm2d(n_feat))
        conv3.append(act)

        conv4 = [nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=4, bias=True, dilation=4)]
        if bn: conv4.append(nn.BatchNorm2d(n_feat))
        conv4.append(act)

        conv5 = [nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=3, bias=True, dilation=3)]
        if bn: conv5.append(nn.BatchNorm2d(n_feat))
        conv5.append(act)

        conv6 = [nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=2, bias=True, dilation=2)]
        if bn: conv6.append(nn.BatchNorm2d(n_feat))
        conv6.append(act)

        conv7 = [nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=1, bias=True, dilation=1)]
        modules_conv = [conv(7 * n_feat, n_feat, kernel_size, bias=bias)]

        self.conv1 = nn.Sequential(*conv1)
        self.conv2 = nn.Sequential(*conv2)
        self.conv3 = nn.Sequential(*conv3)
        self.conv4 = nn.Sequential(*conv4)
        self.conv5 = nn.Sequential(*conv5)
        self.conv6 = nn.Sequential(*conv6)
        self.conv7 = nn.Sequential(*conv7)
        self.conv = nn.Sequential(*modules_conv)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        c7 = self.conv7(c6)
        c_out = torch.cat([c1, c2, c3, c4, c5, c6, c7], 1)
        res = self.conv(c_out)
        res += x
        return res