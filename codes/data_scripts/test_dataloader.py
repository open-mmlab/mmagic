import sys
import os.path as osp
import math
import torchvision.utils

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from data import create_dataloader, create_dataset  # noqa: E402
from utils import util  # noqa: E402


def main():
    dataset = 'DIV2K800_sub'  # REDS | Vimeo90K | DIV2K800_sub
    opt = {}
    opt['dist'] = False
    opt['gpu_ids'] = [0]
    if dataset == 'REDS':
        opt['name'] = 'test_REDS'
        opt['dataroot_GT'] = '../../datasets/REDS/train_sharp_wval.lmdb'
        opt['dataroot_LQ'] = '../../datasets/REDS/train_sharp_bicubic_wval.lmdb'
        opt['mode'] = 'REDS'
        opt['N_frames'] = 5
        opt['phase'] = 'train'
        opt['use_shuffle'] = True
        opt['n_workers'] = 8
        opt['batch_size'] = 16
        opt['GT_size'] = 256
        opt['LQ_size'] = 64
        opt['scale'] = 4
        opt['use_flip'] = True
        opt['use_rot'] = True
        opt['interval_list'] = [1]
        opt['random_reverse'] = False
        opt['border_mode'] = False
        opt['cache_keys'] = None
        opt['data_type'] = 'lmdb'  # img | lmdb | mc
    elif dataset == 'Vimeo90K':
        opt['name'] = 'test_Vimeo90K'
        opt['dataroot_GT'] = '../../datasets/vimeo90k/vimeo90k_train_GT.lmdb'
        opt['dataroot_LQ'] = '../../datasets/vimeo90k/vimeo90k_train_LR7frames.lmdb'
        opt['mode'] = 'Vimeo90K'
        opt['N_frames'] = 7
        opt['phase'] = 'train'
        opt['use_shuffle'] = True
        opt['n_workers'] = 8
        opt['batch_size'] = 16
        opt['GT_size'] = 256
        opt['LQ_size'] = 64
        opt['scale'] = 4
        opt['use_flip'] = True
        opt['use_rot'] = True
        opt['interval_list'] = [1]
        opt['random_reverse'] = False
        opt['border_mode'] = False
        opt['cache_keys'] = None
        opt['data_type'] = 'lmdb'  # img | lmdb | mc
    elif dataset == 'DIV2K800_sub':
        opt['name'] = 'DIV2K800'
        opt['dataroot_GT'] = '../../datasets/DIV2K/DIV2K800_sub.lmdb'
        opt['dataroot_LQ'] = '../../datasets/DIV2K/DIV2K800_sub_bicLRx4.lmdb'
        opt['mode'] = 'LQGT'
        opt['phase'] = 'train'
        opt['use_shuffle'] = True
        opt['n_workers'] = 8
        opt['batch_size'] = 16
        opt['GT_size'] = 128
        opt['scale'] = 4
        opt['use_flip'] = True
        opt['use_rot'] = True
        opt['color'] = 'RGB'
        opt['data_type'] = 'lmdb'  # img | lmdb
    else:
        raise ValueError('Please implement by yourself.')

    util.mkdir('tmp')
    train_set = create_dataset(opt)
    train_loader = create_dataloader(train_set, opt, opt, None)
    nrow = int(math.sqrt(opt['batch_size']))
    padding = 2 if opt['phase'] == 'train' else 0

    print('start...')
    for i, data in enumerate(train_loader):
        if i > 5:
            break
        print(i)
        if dataset == 'REDS' or dataset == 'Vimeo90K':
            LQs = data['LQs']
        else:
            LQ = data['LQ']
        GT = data['GT']

        if dataset == 'REDS' or dataset == 'Vimeo90K':
            for j in range(LQs.size(1)):
                torchvision.utils.save_image(LQs[:, j, :, :, :],
                                             'tmp/LQ_{:03d}_{}.png'.format(i, j), nrow=nrow,
                                             padding=padding, normalize=False)
        else:
            torchvision.utils.save_image(LQ, 'tmp/LQ_{:03d}.png'.format(i), nrow=nrow,
                                         padding=padding, normalize=False)
        torchvision.utils.save_image(GT, 'tmp/GT_{:03d}.png'.format(i), nrow=nrow, padding=padding,
                                     normalize=False)


if __name__ == "__main__":
    main()
