import os.path as osp
import torch
import torch.utils.data as data
import data.util as util
import lmdb
import numpy as np


class VideoTestDataset(data.Dataset):
    """
    A video test dataset. Support:
    Vid4
    REDS4
    Vimeo90K-Test

    no need to prepare LMDB files
    """

    def __init__(self, opt):
        super(VideoTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}
        #### Generate data info and cache data
        self.imgs_LQ, self.imgs_GT = {}, {}
        if opt['name'].lower() in ['vid4', 'reds4']:
            if self.data_type == 'lmdb':
                self.lmdb_paths_GT, _ = util.get_image_paths(self.data_type, self.GT_root)
                self.lmdb_paths_LQ, _ = util.get_image_paths(self.data_type, self.LQ_root)
                self.GT_env, self.LQ_env = None, None
                previous_name_a = None
                previous_name_b = None
                for lmdb_path_GT, lmdb_path_LQ in zip(self.lmdb_paths_GT, self.lmdb_paths_LQ):
                    GT_name_a, GT_name_b = lmdb_path_GT.split('_')
                    assert lmdb_path_GT == lmdb_path_LQ, 'GT path and LQ path in lmdb is not matched'
                    if previous_name_a != GT_name_a and previous_name_a is not None:
                        max_idx = int(previous_name_b) + 1
                        for i in range(max_idx):
                            self.data_info['idx'].append('{}/{}'.format(i, max_idx))
                        border_l = [0] * max_idx
                        for i in range(self.half_N_frames):
                            border_l[i] = 1
                            border_l[max_idx - i - 1] = 1
                        self.data_info['border'].extend(border_l)
                    self.data_info['folder'].append(GT_name_a)
                    previous_name_a = GT_name_a
                    previous_name_b = GT_name_b
            else:
                subfolders_LQ = util.glob_file_list(self.LQ_root)
                subfolders_GT = util.glob_file_list(self.GT_root)
                for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
                    subfolder_name = osp.basename(subfolder_GT)
                    img_paths_LQ = util.glob_file_list(subfolder_LQ)
                    img_paths_GT = util.glob_file_list(subfolder_GT)
                    max_idx = len(img_paths_LQ)
                    assert max_idx == len(
                        img_paths_GT), 'Different number of images in LQ and GT folders'
                    self.data_info['path_LQ'].extend(img_paths_LQ)
                    self.data_info['path_GT'].extend(img_paths_GT)
                    self.data_info['folder'].extend([subfolder_name] * max_idx)
                    for i in range(max_idx):
                        self.data_info['idx'].append('{}/{}'.format(i, max_idx))
                    border_l = [0] * max_idx
                    for i in range(self.half_N_frames):
                        border_l[i] = 1
                        border_l[max_idx - i - 1] = 1
                    self.data_info['border'].extend(border_l)

                    if self.cache_data:
                        self.imgs_LQ[subfolder_name] = util.read_img_seq(img_paths_LQ)
                        self.imgs_GT[subfolder_name] = util.read_img_seq(img_paths_GT)
                    else:
                        self.imgs_LQ[subfolder_name] = img_paths_LQ
                        self.imgs_GT[subfolder_name] = img_paths_GT
        elif opt['name'].lower() in ['vimeo90k-test']:
            pass  # TODO
        else:
            raise ValueError(
                'Not support video test dataset. Support Vid4, REDS4 and Vimeo90k-Test.')

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.GT_root, readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.LQ_root, readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        # path_LQ = self.data_info['path_LQ'][index]
        # path_GT = self.data_info['path_GT'][index]
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        select_idx = util.index_generation(idx, max_idx, self.opt['N_frames'],
                                           padding=self.opt['padding'])
        if self.data_type == 'lmdb':
            if self.GT_env is None or self.LQ_env is None:
                self._init_lmdb()
            key = self.lmdb_paths_GT[index]
            name_a, name_b = key.split('_')
            center_frame_idx = int(name_b)
            GT_size_tuple = self.opt['GT_shape']
            LQ_size_tuple = self.opt['LQ_shape']
            img_GT = util.read_img(self.GT_env, key, GT_size_tuple)
            img_LQ_l = []
            for v in select_idx:
                img_LQ = util.read_img(self.LQ_env, '{}_{:08d}'.format(name_a, v), LQ_size_tuple)
                img_LQ_l.append(img_LQ)
            # stack LQ images to NHWC, N is the frame number
            img_LQs = np.stack(img_LQ_l, axis=0)
            # BGR to RGB, HWC to CHW, numpy to tensor
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQs = img_LQs[:, :, :, [2, 1, 0]]
            img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
            imgs_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                         (0, 3, 1, 2)))).float()
        elif self.cache_data:
            imgs_LQ = self.imgs_LQ[folder].index_select(0, torch.LongTensor(select_idx))
            img_GT = self.imgs_GT[folder][idx]
        else:
            imgs_LQ = util.read_img_seq(self.imgs_LQ[folder]).index_select(0, torch.LongTensor(select_idx))
            img_GT = util.read_img_seq(self.imgs_GT[folder])[idx]

        return {
            'LQs': imgs_LQ,
            'GT': img_GT,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border
        }

    def __len__(self):
        if self.data_type == 'lmdb':
            return len(self.lmdb_paths_GT)
        return len(self.data_info['path_GT'])
