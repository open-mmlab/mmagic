import os.path
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AlignedDataset(Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.fine_height = 256
        self.fine_width = 192

        self.text = opt.test_pairs

        dir_I = '_img'
        self.dir_I = os.path.join(opt.dataroot, opt.phase + dir_I)

        dir_C = '_clothes'
        self.dir_C = os.path.join(opt.dataroot, opt.phase + dir_C)

        dir_E = '_edge'
        self.dir_E = os.path.join(opt.dataroot, opt.phase + dir_E)

        self.im_name = []
        self.c_name = []
        self.e_name = []
        self.get_file_name()
        self.dataset_size = len(self.im_name)

    def get_file_name(self):

        with open(self.text, 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                self.im_name.append(os.path.join(self.dir_I, im_name))
                self.c_name.append(os.path.join(self.dir_C, c_name))
                self.e_name.append(os.path.join(self.dir_E, c_name))

    def __getitem__(self, index):

        I_path = os.path.join(self.im_name[index])
        img = Image.open(I_path).convert('RGB')

        params = get_params(self.opt, img.size)
        transform = get_transform(self.opt, params)
        transform_E = get_transform(
            self.opt, params, method=Image.NEAREST, normalize=False)

        I_tensor = transform(img)
        C_path = os.path.join(self.c_name[index])
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform(C)

        E_path = os.path.join(self.e_name[index])
        E = Image.open(E_path).convert('L')
        E_tensor = transform_E(E)

        input_dict = {
            'image': I_tensor,
            'clothes': C_tensor,
            'edge': E_tensor,
            'p_name': self.im_name[index].split('/')[-1]
        }
        return input_dict

    def __len__(self):
        return self.dataset_size


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    flip = 0
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform_resize(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    transform_list.append(
        transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize, method)))
    osize = [256, 192]
    transform_list.append(transforms.Scale(osize, method))
    if 'crop' in opt.resize_or_crop:
        transform_list.append(
            transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2**opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2**opt.n_local_enhancers)
        transform_list.append(
            transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(
            transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    return transforms.Compose(transform_list)


def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(
            transforms.Lambda(
                lambda img: __scale_width(img, opt.loadSize, method)))
        osize = [256, 192]
        transform_list.append(transforms.Resize(osize, method))
    if 'crop' in opt.resize_or_crop:
        transform_list.append(
            transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(16)
        transform_list.append(
            transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(
            transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
