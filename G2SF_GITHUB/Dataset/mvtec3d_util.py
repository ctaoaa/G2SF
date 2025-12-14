import tifffile as tiff
import torch
import numpy as np
import math


def organized_to_unorganized(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])


def read_tiff(path):
    tiff_img = tiff.imread(path)
    return tiff_img


def resize_organized_pcd(organized_pcd, height=224, width=224, tensor_out=True):
    torch_organized_pcd = torch.tensor(organized_pcd).permute(2, 0, 1).unsqueeze(dim=0).contiguous()
    resized_organized_pcd = torch.nn.functional.interpolate(torch_organized_pcd, size=(height, width), mode='nearest')
    if tensor_out:
        return resized_organized_pcd.squeeze(dim=0).contiguous()
    else:
        return resized_organized_pcd.squeeze().permute(1, 2, 0).contiguous().numpy()


def depth_map(organized_pc):
    return organized_pc[:, :, 2]


def roundup_next_100(x):
    return int(math.ceil(x / 100.0)) * 100


def pad_crop(cropped_pc, single_channel=False):
    orig_h, orig_w = cropped_pc.shape[0], cropped_pc.shape[1]
    round_orig_h = roundup_next_100(orig_h)
    round_orig_w = roundup_next_100(orig_w)
    large_side = max(round_orig_h, round_orig_w)
    a = (large_side - orig_h) // 2
    aa = large_side - a - orig_h
    b = (large_side - orig_w) // 2
    bb = large_side - b - orig_w
    if single_channel:
        return np.pad(cropped_pc, pad_width=((a, aa), (b, bb)), mode='constant')
    else:
        return np.pad(cropped_pc, pad_width=((a, aa), (b, bb), (0, 0)), mode='constant')