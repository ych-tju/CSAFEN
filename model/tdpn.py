import torch

from model import common
import math
import torch.nn as nn


def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return TDPN(args, dilated.dilated_conv)
    else:
        return TDPN(args)


class TDPN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(TDPN, self).__init__()

        n_resblock = args.n_resblocks
        n_MFMblock = args.n_MFMblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.decompose = Decompose()
        m_head1 = [conv(args.n_colors, n_feats, kernel_size)]
        m_head2 = [conv(args.n_colors, n_feats, kernel_size)]

        self.body1 = MRFG(n_MFMblock, n_resblock, n_feats, act)
        self.body2 = MRFG(n_MFMblock, n_resblock, n_feats, act)

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

        or_x_res = self.body1(or_x)
        tx_x_res = self.body2(tx_x)
        or_x_res += or_x
        tx_x_res += tx_x

        or_x_res = self.tail1(or_x_res)
        tx_x_res = self.tail2(tx_x_res)
        or_x = self.reconstruct(or_x_res)
        tx_x = self.reconstruct(tx_x_res)

        x = torch.cat([or_x, tx_x], 1)
        x = self.fu(x)
        x = self.add_mean(x)

        return x


def get_gaussian_kernel(kernel_size=5, sigma=1.5, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


class Decompose(nn.Module):
    def __init__(self):
        super(Decompose, self).__init__()
        self.blur_layer = get_gaussian_kernel().cuda()

    def forward(self, x):
        blurred_img = self.blur_layer(x)
        return x - blurred_img


class MRFG(nn.Module):
    def __init__(self, n_MFMblock, n_resblock, n_feats, act, conv=common.default_conv):
        super(MRFG, self).__init__()


        kernel_size = 3
        m_body = []
        for i in range(n_MFMblock):
            m_body.append(MFM(conv, n_resblock, n_feats, act=act, gamma=16))
        m_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*m_body)

    def forward(self, x):
        result = self.body(x)
        return result


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

class MFM(nn.Module):
    def __init__(self, conv, n_resblock, n_feats, act, gamma):
        super(MFM, self).__init__()

        resblock = []
        for i in range(n_resblock):
            resblock.append(ResBlock(
                conv, n_feats, kernel_size=3, act=act
            ))
        self.resblock = nn.Sequential(*resblock)

        self.conv1 = conv(n_feats, n_feats, kernel_size=1)
        self.conv3 = conv(n_feats, n_feats, kernel_size=3)
        self.conv5 = conv(n_feats, n_feats, kernel_size=5)

        branch1 = [
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(n_feats, n_feats // gamma, kernel_size=3, padding=1, dilation=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(n_feats // gamma, n_feats, kernel_size=3, padding=1, dilation=1, bias=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        ]
        branch2 = [
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(n_feats, n_feats // gamma, kernel_size=3, padding=1, dilation=3, bias=True),
            nn.ReLU(),
            nn.Conv2d(n_feats // gamma, n_feats, kernel_size=3, padding=1, dilation=3, bias=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        ]
        branch3 = [
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(n_feats, n_feats // gamma, kernel_size=3, padding=1, dilation=5, bias=True),
            nn.ReLU(),
            nn.Conv2d(n_feats // gamma, n_feats, kernel_size=3, padding=1, dilation=5, bias=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        ]

        fusion = [
            conv(n_feats * 3, n_feats, kernel_size=3),
            conv(n_feats, n_feats, kernel_size=3),
            nn.ReLU(),
            conv(n_feats, n_feats, kernel_size=3),
        ]

        self.branch1 = nn.Sequential(*branch1)
        self.branch2 = nn.Sequential(*branch2)
        self.branch3 = nn.Sequential(*branch3)
        self.fu = nn.Sequential(*fusion)

    def forward(self, x):
        x = self.resblock(x)
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x3 = self.conv5(x)

        attn1 = self.branch1(x1)
        attn2 = self.branch2(x2)
        attn3 = self.branch3(x3)

        x1 = attn1 * x1
        x2 = attn2 * x2
        x3 = attn3 * x3

        res = torch.cat([x1, x2], 1)
        res = torch.cat([res, x3], 1)
        res = self.fu(res)
        res += x

        return res


class Reconstruct(nn.Module):
    def __init__(self, n_feats, n_colors, act, conv=common.default_conv):
        super(Reconstruct, self).__init__()


        kernel_size = 3
        m_body = [
            conv(n_feats, n_feats, kernel_size),
            ResBlock(conv, n_feats, kernel_size, act=act),
            conv(n_feats, n_colors, kernel_size)
        ]
        self.body = nn.Sequential(*m_body)

    def forward(self, x):
        result = self.body(x)
        return result


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

# x = torch.ones(1,64,128,128)
# model = MRFG(20, 5, 64, nn.ReLU(), conv=common.default_conv)
# y = model(x)
# print(y.shape)