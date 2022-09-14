# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys

import mmengine
from mmengine import DictAction
from torchvision import utils

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmedit.apis import init_model, sample_conditional_model  # isort:skip  # noqa
# yapf: enable


def parse_args():
    parser = argparse.ArgumentParser(description='Generation demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--save-path',
        type=str,
        default='./work_dirs/demos/conditional_samples.png',
        help='path to save unconditional samples')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CUDA device id')

    # args for inference/sampling
    parser.add_argument(
        '--num-batches', type=int, default=4, help='Batch size in inference')
    parser.add_argument(
        '--samples-per-classes',
        type=int,
        default=5,
        help=('This argument work together with `label`, and decide the '
              'number of samples to generate for each class in the given '
              '`label`. If `label` is not given, samples-per-classes would '
              'be regard as the total number of the images to sample.'))
    parser.add_argument(
        '--label',
        type=int,
        nargs='+',
        help=('Labels want to sample. If not defined, '
              'random sampling would be applied.'))
    parser.add_argument(
        '--sample-all-classes',
        action='store_true',
        help='Whether sample all classes of the dataset.')

    parser.add_argument(
        '--sample-model',
        type=str,
        default='ema',
        help='Which model to use for sampling')
    parser.add_argument(
        '--sample-cfg',
        nargs='+',
        action=DictAction,
        help='Other customized kwargs for sampling function')

    # args for image grid
    parser.add_argument(
        '--padding', type=int, default=0, help='Padding in the image grid.')
    parser.add_argument(
        '--nrow',
        type=int,
        default=6,
        help=('Number of images displayed in each row of the grid. '
              'This argument would work only when label is not given.'))

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = init_model(
        args.config, checkpoint=args.checkpoint, device=args.device)

    if args.sample_cfg is None:
        args.sample_cfg = dict()

    if args.label is None and not args.sample_all_classes:
        label = None
        num_samples, nrow = args.samples_per_classes, args.nrow
        mmengine.print_log('`label` is not passed, code would randomly sample '
                           f'`samples-per-classes` (={num_samples}) images.')
    else:
        if args.sample_all_classes:
            mmengine.print_log(
                '`sample_all_classes` is set as True, `num-samples`, `label`, '
                'and `nrows` would be ignored.')

            # get num_classes
            if hasattr(model, 'num_classes') and model.num_classes is not None:
                num_classes = model.num_classes
            else:
                raise AttributeError(
                    'Cannot get attribute `num_classes` from '
                    f'{type(model)}. Please check your config.')
            # build label list
            meta_labels = [idx for idx in range(num_classes)]
        else:
            # get unique label
            meta_labels = list(set(args.label))
            meta_labels.sort()

        # generate label to sample
        label = []
        for idx in meta_labels:
            label += [idx] * args.samples_per_classes
        num_samples = len(label)
        nrow = args.samples_per_classes

        mmengine.print_log('Set `nrows` as number of samples for each class '
                           f'(={args.samples_per_classes}).')

    results = sample_conditional_model(model, num_samples, args.num_batches,
                                       args.sample_model, label,
                                       **args.sample_cfg)
    results = (results[:, [2, 1, 0]] + 1.) / 2.

    # save images
    mmengine.mkdir_or_exist(os.path.dirname(args.save_path))
    utils.save_image(results, args.save_path, nrow=nrow, padding=args.padding)


if __name__ == '__main__':
    main()
