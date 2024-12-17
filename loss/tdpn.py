from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class tdpn_loss(nn.Module):
    def __init__(self, rgb_range=3):
        super(tdpn_loss, self).__init__()
        # vgg_features = models.vgg19(pretrained=True).features
        # modules = [m for m in vgg_features]
        #
        # self.vgg = nn.Sequential(*modules[:8])
        # self.vgg = nn.Sequential(*modules[:35])
        self.vgg = models.vgg19(pretrained=True)[:8]
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x

        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        loss = F.l1_loss(sr, hr) + 0.05 * F.mse_loss(vgg_sr, vgg_hr)

        return loss

class TDPN_LOSS(nn.Module):
    def __init__(self, w1, w2, w3, scale):
        super(TDPN_LOSS, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.l1 = tdpn_loss()
        self.l2 = tdpn_loss()
        self.l3 = tdpn_loss()
        self.scale = scale

    def forward(self, sr, or_sr, tx_sr, hr):
        B, C, H, W = hr.shape
        x_down = F.interpolate(hr, scale_factor=float(1) / float(self.scale), mode='bicubic')
        x_down_up = F.interpolate(x_down, size=[H, W], mode='bicubic')
        tx_hr = hr - x_down_up


        l1 = self.l1(sr, hr)
        l2 = self.l2(or_sr, hr)
        l3 = self.l3(tx_sr, tx_hr)
        loss = self.w1 * l1 + self.w2 * l2 + self.w3 * l3
        return loss
