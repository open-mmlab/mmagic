# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def warp(img, flow):
    """Warp inputs image with correspoind flow."""
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([
        flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] /
        ((H - 1.0) / 2.0)
    ], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(
        input=img,
        grid=grid_,
        mode='bilinear',
        padding_mode='border',
        align_corners=True)
    return output


def resize(x, scale_factor):
    """Resize inputs by scale_factor."""
    return F.interpolate(
        x, scale_factor=scale_factor, mode='bilinear', align_corners=False)


class Ternary(nn.Module):
    """Task-oriented Flow Distillatoin Ternary  Loss."""

    def __init__(self, patch_size=7):
        super(Ternary, self).__init__()
        self.patch_size = patch_size
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape(
            (patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float()

    def transform(self, tensor):
        tensor_ = tensor.mean(dim=1, keepdim=True)
        patches = F.conv2d(
            tensor_,
            self.w.to(tensor.device),
            padding=self.patch_size // 2,
            bias=None)
        loc_diff = patches - tensor_
        loc_diff_norm = loc_diff / torch.sqrt(0.81 + loc_diff**2)
        return loc_diff_norm

    def valid_mask(self, tensor):
        padding = self.patch_size // 2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding,
                           w - 2 * padding).type_as(tensor)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, x, y):
        loc_diff_x = self.transform(x)
        loc_diff_y = self.transform(y)
        diff = loc_diff_x - loc_diff_y.detach()
        dist = (diff**2 / (0.1 + diff**2)).mean(dim=1, keepdim=True)
        mask = self.valid_mask(x).to(x.device)
        loss = (dist * mask).mean()
        return loss


class Geometry(nn.Module):
    """Feature Space Geometry Consistency Loss."""

    def __init__(self, patch_size=3):
        super(Geometry, self).__init__()
        self.patch_size = patch_size
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape(
            (patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float()

    def transform(self, tensor):
        b, c, h, w = tensor.size()
        tensor_ = tensor.reshape(b * c, 1, h, w)
        patches = F.conv2d(
            tensor_,
            self.w.to(tensor.device),
            padding=self.patch_size // 2,
            bias=None)
        loc_diff = patches - tensor_
        loc_diff_ = loc_diff.reshape(b, c * (self.patch_size**2), h, w)
        loc_diff_norm = loc_diff_ / torch.sqrt(0.81 + loc_diff_**2)
        return loc_diff_norm

    def valid_mask(self, tensor):
        padding = self.patch_size // 2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding,
                           w - 2 * padding).type_as(tensor)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, x, y):
        loc_diff_x = self.transform(x)
        loc_diff_y = self.transform(y)
        diff = loc_diff_x - loc_diff_y
        dist = (diff**2 / (0.1 + diff**2)).mean(dim=1, keepdim=True)
        mask = self.valid_mask(x)
        loss = (dist * mask).mean()
        return loss


class Charbonnier_L1(nn.Module):
    """Charbonnier L1 loss by putting epsilon into sqrt."""

    def __init__(self):
        super(Charbonnier_L1, self).__init__()

    def forward(self, diff, mask=None):
        if mask is None:
            loss = ((diff**2 + 1e-6)**0.5).mean()
        else:
            loss = (((diff**2 + 1e-6)**0.5) * mask).mean() / (
                mask.mean() + 1e-9)
        return loss


class Charbonnier_Ada(nn.Module):
    """Charbonnier Adam loss by using weights as epsilon."""

    def __init__(self):
        super(Charbonnier_Ada, self).__init__()

    def forward(self, diff, weight):
        alpha = weight / 2
        epsilon = 10**(-(10 * weight - 1) / 3)
        loss = ((diff**2 + epsilon**2)**alpha).mean()
        return loss
