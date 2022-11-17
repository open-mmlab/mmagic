# Copyright (c) OpenMMLab. All rights reserved.
# isort: off
from argparse import ArgumentParser
from mmedit.edit import MMEdit
from mmengine import DictAction


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--img', type=str, default=None, help='Input image file.')
    parser.add_argument(
        '--video', type=str, default=None, help='Input video file.')
    parser.add_argument(
        '--label',
        type=int,
        default=None,
        help='Input label for conditional models.')
    parser.add_argument(
        '--trimap', type=str, default=None, help='Input for matting models.')
    parser.add_argument(
        '--mask',
        type=str,
        default=None,
        help='path to input mask file for inpainting models')
    parser.add_argument(
        '--result-out-dir',
        type=str,
        default=None,
        help='Output img or video path.')
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Pretrained editing algorithm')
    parser.add_argument(
        '--model-setting',
        type=int,
        default=None,
        help='Pretrained editing algorithm setting')
    parser.add_argument(
        '--model-config',
        type=str,
        default=None,
        help='Path to the custom config file of the selected editing model.')
    parser.add_argument(
        '--model-ckpt',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected det model.')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device used for inference.')
    parser.add_argument(
        '--extra-parameters',
        nargs='+',
        action=DictAction,
        help='Other customized kwargs for different model')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    editor = MMEdit(**vars(args))
    editor.infer(**vars(args))


if __name__ == '__main__':
    main()
