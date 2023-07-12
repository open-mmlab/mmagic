# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys

import mmcv
import torch
from mmengine import Config, print_log
from mmengine.logging import MMLogger
from mmengine.runner import load_checkpoint, set_random_seed

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmagic.engine import *  # isort:skip  # noqa: F401,F403,E402
from mmagic.datasets import *  # isort:skip  # noqa: F401,F403,E402
from mmagic.models import *  # isort:skip  # noqa: F401,F403,E402

from mmagic.registry import MODELS  # isort:skip  # noqa

# yapf: enable


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a GAN model')
    parser.add_argument('config', help='evaluation config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--samples-path',
        type=str,
        default='./',
        help='path to store images. If not given, remove it after evaluation\
             finished')
    parser.add_argument(
        '--save-prev-res',
        action='store_true',
        help='whether to store the results from previous stages')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='the number of synthesized samples')
    args = parser.parse_args()
    return args


def _tensor2img(img):
    img = img.permute(1, 2, 0)
    img = img.clamp(0, 255).to(torch.uint8)

    return img.cpu().numpy()


@torch.no_grad()
def main():
    MMLogger.get_instance('mmagic')

    args = parse_args()
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # set scope manually
    cfg.model['_scope_'] = 'mmagic'
    # build the model and load checkpoint
    model = MODELS.build(cfg.model)

    model.eval()

    # load ckpt
    print_log(f'Loading ckpt from {args.checkpoint}')
    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # add dp wrapper
    if torch.cuda.is_available():
        model = model.cuda()

    for sample_iter in range(args.num_samples):
        outputs = model.test_step(
            dict(inputs=dict(num_batches=1, get_prev_res=args.save_prev_res)))

        # store results from previous stages
        if args.save_prev_res:
            fake_img = outputs[0].fake_img.data
            prev_res_list = outputs[0].prev_res_list
            prev_res_list.append(fake_img)
            for i, img in enumerate(prev_res_list):
                img = _tensor2img(img)
                mmcv.imwrite(
                    img,
                    os.path.join(args.samples_path, f'stage{i}',
                                 f'rand_sample_{sample_iter}.png'))
        # just store the final result
        else:
            img = _tensor2img(outputs[0].fake_img.data)
            mmcv.imwrite(
                img,
                os.path.join(args.samples_path,
                             f'rand_sample_{sample_iter}.png'))


if __name__ == '__main__':
    main()
