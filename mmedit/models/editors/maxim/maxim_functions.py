# Copyright (c) OpenMMLab. All rights reserved.
import einops


def block_images_einops(x, patch_size):
    """Image to patches."""
    batch, height, width, channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    x = einops.rearrange(
        x,
        'n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c',
        gh=grid_height,
        gw=grid_width,
        fh=patch_size[0],
        fw=patch_size[1])
    return x


def unblock_images_einops(x, grid_size, patch_size):
    """patches to images."""
    x = einops.rearrange(
        x,
        'n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c',
        gh=grid_size[0],
        gw=grid_size[1],
        fh=patch_size[0],
        fw=patch_size[1])
    return x
