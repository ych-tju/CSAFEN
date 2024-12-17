# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import torch

from model import common
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import transforms
from einops import rearrange


def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return TDPN_SSFIB_HESSIAN(args, dilated.dilated_conv)
    else:
        return TDPN_SSFIB_HESSIAN(args)


class TDPN_SSFIB_HESSIAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(TDPN_SSFIB_HESSIAN, self).__init__()

        n_resblock = args.n_resblocks
        n_MFMblock = args.n_MFMblocks
        n_feats = args.n_feats
        n_class = args.n_class
        d_kernel_size = args.d_kernel_size
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        m_head1 = [conv(args.n_colors, n_feats, kernel_size)]
        m_head2 = [conv(args.n_colors, n_feats, kernel_size)]
        self.decompose = Decompose(scale, args.n_colors)
        self.body = MRFG(n_MFMblock, n_resblock, n_feats, act, n_class, d_kernel_size, scale)

        m_tail1 = [common.Upsampler(conv, scale, n_feats, act=False)]
        m_tail2 = [common.Upsampler(conv, scale, n_feats, act=False)]

        self.head1 = nn.Sequential(*m_head1)
        self.head2 = nn.Sequential(*m_head2)
        self.tail1 = nn.Sequential(*m_tail1)
        self.tail2 = nn.Sequential(*m_tail2)

        self.reconstruct = Reconstruct(n_feats, args.n_colors, act)
        self.fu = Fusion(n_feats, args.n_colors, act)

    def forward(self, x):
        or_x = self.sub_mean(x)
        tx_x = self.decompose(or_x)
        or_x = self.head1(or_x)
        tx_x = self.head2(tx_x)

        x = torch.cat([or_x, tx_x], dim=1)

        or_x_res, tx_x_res = self.body(x)

        or_x_res += or_x
        tx_x_res += tx_x

        or_x_res = self.tail1(or_x_res)
        tx_x_res = self.tail2(tx_x_res)
        or_x = self.reconstruct(or_x_res)
        tx_x = self.reconstruct(tx_x_res)

        x = torch.cat([or_x, tx_x], dim=1)
        x = self.fu(x)
        sr = self.add_mean(x)

        return sr


class Decompose(nn.Module):
    def __init__(self, scale, n_feats, conv=common.default_conv):
        super(Decompose, self).__init__()
        self.scale = scale
        self.transform = transforms.Compose([transforms.GaussianBlur(kernel_size=5)])

    def forward(self, x):
        B, C, H, W = x.shape
        x_down = F.interpolate(x, scale_factor=float(1) / float(self.scale), mode='bicubic')
        x_down_up = F.interpolate(x_down, size=[H, W], mode='bicubic')
        cross_scale_f = torch.abs(x - x_down_up)

        x_gaussain_blur = transforms.GaussianBlur(5, 1.5)(x)
        gaussain_f = torch.abs(x - x_gaussain_blur)

        zero = torch.zeros_like(gaussain_f)
        cross_scale_f = cross_scale_f - gaussain_f
        cross_scale_f = torch.where(cross_scale_f < 0, zero, cross_scale_f)

        return gaussain_f + cross_scale_f


class MRFG(nn.Module):
    def __init__(self, n_MFMblock, n_resblock, n_feats, act, n_class, d_kernel_size, scale, conv=common.default_conv):
        super(MRFG, self).__init__()

        kernel_size = 3
        m_body = []
        for i in range(n_MFMblock):
            m_body.append(MRFB(n_resblock, n_feats, act=act, gamma=16, n_class=n_class, d_kernel_size=d_kernel_size[0], scale=scale))
        self.or_conv = (conv(n_feats, n_feats, kernel_size))
        self.tx_conv = (conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*m_body)

    def forward(self, x):
        result = self.body(x)
        or_x, tx_x = torch.chunk(result, 2, dim=1)
        or_x = self.or_conv(or_x)
        tx_x = self.tx_conv(tx_x)

        return or_x, tx_x


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU()):
        super(ResBlock, self).__init__()

        m = [
            conv(n_feats, n_feats, kernel_size),
            act,
            conv(n_feats, n_feats, kernel_size),
        ]
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res


class MSRB(nn.Module):
    def __init__(
            self, conv, n_feats,
            bias=True, bn=False, act=nn.ReLU()):
        super(MSRB, self).__init__()

        self.conv3x3_1 = conv(n_feats, n_feats, kernel_size=3)
        self.conv3x3_2 = conv(2 * n_feats, n_feats, kernel_size=3)
        self.conv5x5_1 = conv(n_feats, n_feats, kernel_size=5)
        self.conv5x5_2 = conv(2 * n_feats, n_feats, kernel_size=5)
        self.conv1x1 = conv(2 * n_feats, n_feats, kernel_size=1)
        self.act = act

    def forward(self, x):
        s1 = self.act(self.conv3x3_1(x))
        p1 = self.act(self.conv5x5_1(x))

        x1 = torch.cat([s1, p1], dim=1)
        s2 = self.act(self.conv3x3_2(x1))
        p2 = self.act(self.conv5x5_2(x1))

        res = torch.cat([s2, p2], dim=1)
        res = self.conv1x1(res)
        res += x

        return res


class MRFB(nn.Module):
    def __init__(self, n_resblock, n_feats, act, gamma, n_class, d_kernel_size, scale, conv=common.default_conv):
        super(MRFB, self).__init__()

        kernel_size = 3
        self.or_branch = Dictionary(n_feats, d_kernel_size, n_class, scale, act)
        self.tx_branch = Dictionary(n_feats, d_kernel_size, n_class, scale, act)

        fusion = [
            conv(2 * n_feats, 2 * n_feats, kernel_size),
            act,
            conv(2 * n_feats, 2 * n_feats, kernel_size),
            act
        ]
        self.fusion = nn.Sequential(*fusion)
        self.or_conv = conv(n_feats, n_feats, kernel_size)
        self.tx_conv = conv(n_feats, n_feats, kernel_size)

    def forward(self, x):
        or_x, tx_x = torch.chunk(x, 2, dim=1)

        or_x = self.or_branch(or_x)
        tx_x = self.tx_branch(tx_x)

        fu_x = torch.cat([or_x, tx_x], dim=1)
        fu_x = self.fusion(fu_x)
        or_x_res, tx_x_res = torch.chunk(fu_x, 2, dim=1)

        or_x = or_x + or_x_res
        tx_x = tx_x + tx_x_res
        result = torch.cat([or_x, tx_x], dim=1)
        return result


class Reconstruct(nn.Module):
    def __init__(self, n_feats, n_colors, act, conv=common.default_conv):
        super(Reconstruct, self).__init__()

        kernel_size = 3
        m_body = [
            conv(n_feats, n_feats, kernel_size),
            ResBlock(conv, n_feats, kernel_size=kernel_size, act=act),
            conv(n_feats, n_colors, kernel_size)
        ]
        self.body = nn.Sequential(*m_body)

    def forward(self, x):
        result = self.body(x)
        return result


class Dictionary(nn.Module):
    def __init__(self, n_feats, kernel_size, n_class, scale, act, conv=common.default_conv):
        super(Dictionary, self).__init__()
        self.kernel_size = kernel_size
        m_body = [
            MSRB(conv, n_feats, act=act),
            MSRB(conv, n_feats, act=act),
            MSRB(conv, n_feats, act=act),
        ]

        self.body = nn.Sequential(*m_body)
        self.n_class = n_class
        self.scale = scale
        self.dictionary = Parameter(torch.zeros(1, n_class, scale*kernel_size*scale*kernel_size), requires_grad=True).cuda()
        mapping = [
            nn.Linear(kernel_size * kernel_size, 256),
            act,
            nn.Linear(256, kernel_size * kernel_size),
        ]

        fuse = [
            nn.Linear(2 * kernel_size * kernel_size, 256),
            act,
            nn.Linear(256, kernel_size * kernel_size),
        ]
        self.mapping = nn.Sequential(*mapping)
        self.fuse = nn.Sequential(*fuse)
        self.sigmoid = nn.Sigmoid()

        threshold = torch.rand(1)
        self.threshold = nn.Parameter(threshold)

        self.unfold = torch.nn.Unfold(kernel_size=(kernel_size, kernel_size), stride=kernel_size)
        self.conv1x1 = common.default_conv(in_channels=2*n_feats, out_channels=n_feats, kernel_size=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, dilation=1, bias=True),
            act,
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=2, dilation=2, bias=True),
            act,
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=4, dilation=4, bias=True),
            act,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_feats, n_feats, kernel_size=3, padding=4, dilation=4, bias=True),
            act,
            nn.ConvTranspose2d(n_feats, n_feats, kernel_size=3, padding=2, dilation=2, bias=True),
            act,
            nn.ConvTranspose2d(n_feats, n_feats, kernel_size=3, padding=1, dilation=1, bias=True),
            act,
        )
        self.gate = nn.Conv2d(in_channels=n_feats, out_channels=1, kernel_size=1)

        self.conv3x3 = common.default_conv(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=scale)

    def forward(self, x):
        B, N_feats, H, W = x.shape
        x1 = self.body(x)
        x = x
        x1 = x1
        x1 = F.unfold(x1, self.kernel_size, stride=self.kernel_size).transpose(1, 2).contiguous()
        B, L, lar = x1.shape  # lar=N_feats*kernel^2
        x1 = x1.view(B, L, N_feats, self.kernel_size * self.kernel_size)
        x1 = self.mapping(x1)
        dictionary_resahpe = rearrange(self.dictionary, 'B n (h w) -> B n h w', h=self.scale * self.kernel_size, w=self.scale * self.kernel_size)
        dictionary_LR = F.interpolate(dictionary_resahpe, size=[self.kernel_size, self.kernel_size], mode='bicubic')
        dictionary_LR = dictionary_LR.squeeze()
        dictionary_LR = rearrange(dictionary_LR, 'n h w -> n (h w)', h=self.kernel_size, w=self.kernel_size)
        dictionary1 = dictionary_LR.expand(B, L, N_feats, self.n_class,
                                                  self.kernel_size * self.kernel_size).transpose(2, 3).transpose(1,
                                                                                                                 2).transpose(
            0, 1)
        x1_expand = x1.expand(self.n_class, B, L, N_feats, self.kernel_size * self.kernel_size).to(self.device)
        similarity = torch.cosine_similarity(x1_expand, dictionary1, dim=4).transpose(0, 1).transpose(1,
                                                                                                           2).transpose(
            2,
            3)
        # similarity = self.sigmoid(similarity)

        max_val, idx = torch.max(similarity, 3)
        max_val = torch.unsqueeze(max_val, 3)
        max_val = max_val.expand(B, L, N_feats, self.n_class).to(self.device)
        # print(max_val.shape)
        zero = torch.zeros_like(max_val)
        similarity = torch.where(similarity < max_val, zero, similarity)
        # print(similarity.shape)

        x2_calibrate = torch.matmul(similarity, self.dictionary)

        fold1 = torch.nn.Fold(output_size=(H, W), kernel_size=(self.kernel_size, self.kernel_size),
                             stride=self.kernel_size)
        x1 = fold1(x1.view(B, L, lar).transpose(1, 2).contiguous())

        fold2 = torch.nn.Fold(output_size=(self.scale*H, self.scale*W), kernel_size=(self.scale*self.kernel_size, self.scale*self.kernel_size),
                             stride=self.scale*self.kernel_size)
        x2_calibrate = fold2(x2_calibrate.view(B, L, lar*self.scale*self.scale).transpose(1, 2).contiguous())

        x1_upsmaple = F.interpolate(x1, size=[self.scale*H, self.scale*W], mode='bicubic').to(self.device)
        x3 = torch.cat([x1_upsmaple, x2_calibrate], dim=1).to(self.device)
        x3 = self.conv1x1(x3)

        attn = self.gate(self.decoder(self.encoder(x3)))
        x3 = attn * x3
        x3 = self.conv3x3(x3)
        return x3


class Fusion(nn.Module):
    def __init__(self, n_feats, n_colors, act, conv=common.default_conv):
        super(Fusion, self).__init__()

        kernel_size = 3
        m_body = [
            conv(2 * n_colors, n_feats, kernel_size),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=2, dilation=2, bias=True),
            act,
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=2, dilation=2, bias=True),
            conv(n_feats, n_colors, kernel_size)
        ]
        self.body = nn.Sequential(*m_body)

    def forward(self, x):
        result = self.body(x)
        return result




#x1 = torch.ones(2, 32, 127, 129)
#x2 = torch.ones(2, 32, 128, 128)

#model = Dictionary(n_feats=32, kernel_size=3, n_class=32, scale=4, act=nn.ReLU())
#y = model(x1)
#print(y.shape)


