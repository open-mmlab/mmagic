# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import mmcv
import torch

from mmedit.apis import init_model, matting_inference


def parse_args():
    parser = argparse.ArgumentParser(description='Matting demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img_path', help='path to input image file')
    parser.add_argument('trimap_path', help='path to input trimap file')
    parser.add_argument('save_path', help='path to save alpha matte result')
    parser.add_argument(
        '--imshow', action='store_true', help='whether show image with opencv')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.device < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)

    model = init_model(args.config, args.checkpoint, device=device)

    pred_alpha = matting_inference(model, args.img_path,
                                   args.trimap_path) * 255

    mmcv.imwrite(pred_alpha, args.save_path)
    if args.imshow:
        mmcv.imshow(pred_alpha, 'predicted alpha matte')


if __name__ == '__main__':
    main()
