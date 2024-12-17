import os
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import rotate
import torch.nn.functional as F

def normalize(x):
    return x.mul_(2).add_(-1)

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()
    
    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks
def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


class RandCrop(object):
    def __init__(self, crop_size, scale):
        # if output size is tuple -> (height, width)
        assert isinstance(crop_size, (int, tuple))
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            assert len(crop_size) == 2
            self.crop_size = crop_size

        self.scale = scale

    def __call__(self, sample):
        # img_LR: H x W x C (numpy array)
        img_LR, img_HR = sample['img_LR'], sample['img_HR']

        h, w, c = img_LR.shape
        new_h, new_w = self.crop_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        img_LR_crop = img_LR[top: top + new_h, left: left + new_w, :]

        h, w, c = img_HR.shape
        # top = np.random.randint(0, h - self.scale * new_h)
        # left = np.random.randint(0, w - self.scale * new_w)
        top = self.scale[0] * top
        left = self.scale[0] * left
        img_HR_crop = img_HR[top: top + self.scale[0] * new_h, left: left + self.scale[0] * new_w, :]

        sample = {'img_LR': img_LR_crop, 'img_HR': img_HR_crop}
        return sample


class RandRotate(object):
    def __call__(self, sample):
        # img_LR: H x W x C (numpy array)
        img_LR, img_HR = sample['img_LR'], sample['img_HR']

        prob_rotate = np.random.random()
        if prob_rotate < 0.25:
            img_LR = rotate(img_LR, 90).copy()
            img_HR = rotate(img_HR, 90).copy()
        elif prob_rotate < 0.5:
            img_LR = rotate(img_LR, 90).copy()
            img_HR = rotate(img_HR, 90).copy()
        elif prob_rotate < 0.75:
            img_LR = rotate(img_LR, 90).copy()
            img_HR = rotate(img_HR, 90).copy()

        sample = {'img_LR': img_LR, 'img_HR': img_HR}
        return sample


class RandHorizontalFlip(object):
    def __call__(self, sample):
        # img_LR: H x W x C (numpy array)
        img_LR, img_HR = sample['img_LR'], sample['img_HR']

        prob_lr = np.random.random()
        if prob_lr < 0.5:
            img_LR = np.fliplr(img_LR).copy()
            img_HR = np.fliplr(img_HR).copy()

        sample = {'img_LR': img_LR, 'img_HR': img_HR}
        return sample


class ToTensor(object):
    def __call__(self, sample):
        # img_LR : H x W x C (numpy array) -> C x H x W (torch tensor)
        img_LR, img_HR = sample['img_LR'], sample['img_HR']

        img_LR = img_LR.transpose((2, 0, 1))
        img_HR = img_HR.transpose((2, 0, 1))

        img_LR = torch.from_numpy(img_LR)
        img_HR = torch.from_numpy(img_HR)

        sample = {'img_LR': img_LR, 'img_HR': img_HR}
        return sample


def cutting_pad(x, block_size, pad):
    N, C, H, W = x.shape  # [N,C,H,W]
    patch_size = block_size - pad
    num_patch_h = H // patch_size + 1 if H % patch_size != 0 else H // patch_size
    num_patch_w = W // patch_size + 1 if W % patch_size != 0 else W // patch_size
    num_patch = num_patch_h * num_patch_w

    pad_2 = pad // 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block_x = torch.zeros([N, num_patch_h, num_patch_w, C, block_size, block_size]).to(device)
    # block_x = torch.zeros([N, num_patch_h, num_patch_w, C, block_size, block_size])
    for i in range(1,num_patch_h - 1):
        for j in range(1,num_patch_w - 1):
            block_x[:, i, j, :, :, :] = x[:, :, i * patch_size - pad_2:(i + 1) * patch_size + pad_2,
                                        j * patch_size - pad_2:(j + 1) * patch_size + pad_2]
    for i in range(1,num_patch_h - 1):
        block_x[:, i, 0, :, :, :] = x[:, :, i * patch_size - pad_2:(i + 1) * patch_size + pad_2,
                                                  0:block_size]
        block_x[:, i, num_patch_w - 1, :, :, :] = x[:, :, i * patch_size - pad_2:(i + 1) * patch_size + pad_2,
                                                  -block_size:]
    for j in range(1,num_patch_w - 1):
        block_x[:, 0, j, :, :, :] = x[:, :, 0:block_size,
                                                  j * patch_size - pad_2:(j + 1) * patch_size + pad_2]
        block_x[:, num_patch_h - 1, j, :, :, :] = x[:, :, -block_size:,
                                                  j * patch_size - pad_2:(j + 1) * patch_size + pad_2]
    block_x[:, 0, 0, :, :, :] = x[:, :, 0:block_size, 0:block_size]
    block_x[:, 0, num_patch_w - 1, :, :, :] = x[:, :, 0:block_size, -block_size:]
    block_x[:, num_patch_h - 1, 0, :, :, :] = x[:, :, -block_size:, 0:block_size]
    block_x[:, num_patch_h - 1, num_patch_w - 1, :, :, :] = x[:, :, -block_size:, -block_size:]
    x = block_x.reshape(N * num_patch, C, block_size, block_size)
    return x, num_patch_h, num_patch_w, H, W


def recutting_pad(x, block_size, num_patch_h, num_patch_w, H, W, pad ,scale):
    scale_block_size = x.shape[-1]
    scale = scale_block_size // block_size

    scale_pad = pad * scale
    scale_pad_2 = scale_pad // 2
    scale_patch_size = scale_block_size - scale_pad
    N_fold, C, _, _ = x.shape  # [N*num_h*num_w,C,H,W]
    N = int(N_fold / (num_patch_h * num_patch_w))
    x = x.reshape(N, num_patch_h, num_patch_w, C, scale_block_size, scale_block_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.zeros([N, C, scale * H, scale * W]).to(device)

    #####################
    for i in range(1, num_patch_h - 1):
        for j in range(1, num_patch_w - 1):
            result[:, :, i * scale_patch_size:(i + 1) * scale_patch_size,
            j * scale_patch_size:(j + 1) * scale_patch_size] = x[:, i, j, :, scale_pad_2: scale_pad_2 + scale_patch_size, scale_pad_2: scale_pad_2 + scale_patch_size]
    for i in range(1, num_patch_h - 1):
        result[:, :, i * scale_patch_size:(i + 1) * scale_patch_size, 0:scale_patch_size] = x[:, i, 0,
                                                                                            :,
                                                                                            scale_pad_2: scale_pad_2 + scale_patch_size,
                                                                                            0: scale_patch_size]
        result[:, :, i * scale_patch_size:(i + 1) * scale_patch_size, -scale_patch_size:] = x[:, i, num_patch_w - 1,
                                                                                            :,
                                                                                            scale_pad_2: scale_pad_2 + scale_patch_size,
                                                                                            -scale_patch_size:]
    for j in range(1, num_patch_w - 1):
        result[:, :, 0:scale_patch_size, j * scale_patch_size:(j + 1) * scale_patch_size] = x[:, 0, j,
                                                                                            :,
                                                                                            0: scale_patch_size,
                                                                                            scale_pad_2: scale_pad_2 + scale_patch_size]
        result[:, :, -scale_patch_size:, j * scale_patch_size:(j + 1) * scale_patch_size] = x[:, num_patch_h - 1, j,
                                                                                            :,
                                                                                            -scale_patch_size:,
                                                                                            scale_pad_2: scale_pad_2 + scale_patch_size]
    result[:, :, 0:scale_patch_size, 0:scale_patch_size] = x[:, 0, 0, :, 0:scale_patch_size,
                                                           0:scale_patch_size]
    result[:, :, 0:scale_patch_size, -scale_patch_size:] = x[:, 0, num_patch_w - 1, :, 0:scale_patch_size,
                                                           -scale_patch_size:]
    result[:, :, -scale_patch_size:, 0:scale_patch_size] = x[:, num_patch_h - 1, 0, :, -scale_patch_size:,
                                                           0:scale_patch_size]
    result[:, :, -scale_patch_size:, -scale_patch_size:] = x[:, num_patch_h - 1, num_patch_w - 1, :, -scale_patch_size:,
                                                           -scale_patch_size:]
    return result

def cutting(x, block_size):
    N, C, H, W = x.shape  # [N,C,H,W]
    patch_size = block_size
    num_patch_h = H // patch_size + 1 if H % patch_size != 0 else H // patch_size
    num_patch_w = W // patch_size + 1 if W % patch_size != 0 else W // patch_size
    num_patch = num_patch_h * num_patch_w



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block_x = torch.zeros([N, num_patch_h, num_patch_w, C, block_size, block_size]).to(device)
    # block_x = torch.zeros([N, num_patch_h, num_patch_w, C, block_size, block_size])
    for i in range(0,num_patch_h - 1):
        for j in range(0,num_patch_w - 1):
            block_x[:, i, j, :, :, :] = x[:, :, i * patch_size :(i + 1) * patch_size ,
                                        j * patch_size :(j + 1) * patch_size ]
    for i in range(0,num_patch_h - 1):
        block_x[:, i, num_patch_w - 1, :, :, :] = x[:, :, i * patch_size :(i + 1) * patch_size ,
                                                  -block_size:]
    for j in range(0,num_patch_w - 1):
        block_x[:, num_patch_h - 1, j, :, :, :] = x[:, :, -block_size:,
                                                  j * patch_size:(j + 1) * patch_size]
    block_x[:, num_patch_h - 1, num_patch_w - 1, :, :, :] = x[:, :, -block_size:, -block_size:]
    x = block_x.reshape(N * num_patch, C, block_size, block_size)
    return x, num_patch_h, num_patch_w, H, W


def recutting(x, block_size, num_patch_h, num_patch_w, H, W):
    scale_block_size = x.shape[-1]
    scale = scale_block_size // block_size

    scale_patch_size = scale_block_size
    N_fold, C, _, _ = x.shape  # [N*num_h*num_w,C,H,W]
    N = int(N_fold / (num_patch_h * num_patch_w))
    x = x.reshape(N, num_patch_h, num_patch_w, C, scale_block_size, scale_block_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.zeros([N, C, scale * H, scale * W]).to(device)

    #####################
    for i in range(0, num_patch_h - 1):
        for j in range(0, num_patch_w - 1):
            result[:, :, i * scale_patch_size:(i + 1) * scale_patch_size,
            j * scale_patch_size:(j + 1) * scale_patch_size] = x[:, i, j, :, :, :]
    for i in range(0, num_patch_h - 1):
        result[:, :, i * scale_patch_size:(i + 1) * scale_patch_size, -scale_patch_size:] = x[:, i, num_patch_w - 1, :, :, :]
    for j in range(0, num_patch_w - 1):
        result[:, :, -scale_patch_size:, j * scale_patch_size:(j + 1) * scale_patch_size] = x[:, num_patch_h - 1, j, :, :, :]
    result[:, :, -scale_patch_size:, -scale_patch_size:] = x[:, num_patch_h - 1, num_patch_w - 1, :, :, :]
    return result

# import torch
# from option import args
# x = torch.ones(1,3,140,140)
# y, num_patch_h, num_patch_w, H, W = cutting(x,128)
#
# print(y.shape)