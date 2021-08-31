import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch
import torch.nn as nn
import torch.nn.functional as F

import net.resnet as resnet_models


def save_as_img(x, path):
    x = x.cpu().numpy()
    plt.imsave(path, x)


class up_conv(nn.Module):
    """Up Convolution Block"""
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch,
                      out_ch,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x


class conv_block(nn.Module):
    """Convolution Block """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,
                      out_ch,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,
                      out_ch,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class SegNet(nn.Module):
    """ U-Net """
    def __init__(self,
                 arch,
                 out_ch,
                 layer=0,
                 output_dim=0,
                 encoder_w=None,
                 input_channel=3):

        base_filters = 64 if arch == 'resnet18' else 256
        filters = [
            base_filters, base_filters * 2, base_filters * 4, base_filters * 8
        ]
        super(SegNet, self).__init__()
        self.encoder = resnet_models.__dict__[arch](
            eval_mode=True, input_channel=input_channel)

        if encoder_w is not None:
            print('load weight from:' + encoder_w)
            if "PixPro" not in encoder_w:
                stat_dict = torch.load(encoder_w).net.state_dict()
            else:
                stat_dict = torch.load(
                    encoder_w).online_encoder.net.state_dict()
            self.encoder.load_state_dict(stat_dict)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0],
                              out_ch,
                              kernel_size=1,
                              stride=1,
                              padding=0)

    def forward(self, x, layer=-1):
        _, _, h, w = x.size()
        e2, e3, e4, e5 = self.encoder(x)
        d4 = self.Up4(e5)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.Up_conv4(d4)
        #         print(d4.size())

        d3 = self.Up3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.Up_conv3(d3)
        #         print(d3.size())

        d2 = self.Up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.Up_conv2(d2)
        #         print(d2.size())

        out = self.Conv(d2)
        # print(out.shape)
        out = F.upsample(
            out, size=(h, w),
            mode="bilinear")  # (shape: (batch_size, num_classes, h, w))

        if layer == -1:
            # for segmentation
            return out
        else:
            decoder_features = [e5, d4, d3, d2]
            return decoder_features[layer]


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    x = torch.rand((4, 3, 192, 192)).cuda()
    net = SegNet('resnet50', 3).cuda()
    y = net(x)
    from net.utils import load_model