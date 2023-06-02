# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import mmcv
import torch

from mmengine.dataset import Compose
from mmengine.dataset.utils import default_collate as collate
from torch.nn.parallel import scatter

from mmagic.apis import MMagicInferencer

from mmagic.utils import tensor2img


def parse_args():
    parser = argparse.ArgumentParser(description='Deblurganv2 demo')
    parser.add_argument('--config', help='test config file path', default='G:/github/mmagic/configs/deblurganv2/deblurganv2_fpn_inception.py')
    #parser.add_argument('--checkpoint', help='checkpoint file', default='D:/pythonProject/DeblurGANv2/fpn_inception.pth')
    # parser.add_argument('--checkpoint', help='checkpoint file',
    #                     default='G:/github/mmagic/work_dir/best_PSNR_iter_100000.pth')
    parser.add_argument('--checkpoint', help='checkpoint file', default='G:/github/DeblurGANv2/fpn_inception.pth')
    parser.add_argument('--masked_img_path', help='path to input image file', default='G:/github/DeblurGANv2/test_img/000027.png')
    parser.add_argument('--mask_path', help='path to input mask file', default=None)
    parser.add_argument('--save_path', help='path to save deblurganv2 result', default='out.png')
    parser.add_argument(
        '--imshow', action='store_true', help='whether show image with opencv', default=True)
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args

def deblurganv2_inference(model, img, ref=None):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): File path of input image.
        ref (str | None): File path of reference image. Default: None.

    Returns:
        Tensor: The predicted restoration result.
    """
    with torch.no_grad():
        # result = model(mode='tensor', **data)
        result = model.infer(img=img)
    result = result[0]
    return result

def main():
    args = parse_args()

    if args.device < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)

    model = MMagicInferencer(model_name='deblurganv2',
                             model_setting=0,
                             model_config=args.config,
                             model_ckpt=args.checkpoint,
                             device=device)

    result = model.infer(img=args.masked_img_path, ref=args.mask_path, result_out_dir=args.save_path)
    # result = tensor2img(result)

    # mmcv.imwrite(result, args.save_path)
    if args.imshow:
        mmcv.imshow(result[1], 'predicted inpainting result')


if __name__ == '__main__':
    main()
