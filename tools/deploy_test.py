# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings
from typing import Any

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from torch import nn

from mmedit.apis import single_gpu_test
from mmedit.core.export import ONNXRuntimeEditing
from mmedit.datasets import build_dataloader, build_dataset
from mmedit.models import BasicRestorer, build_model


class TensorRTRestorerGenerator(nn.Module):
    """Inner class for tensorrt restorer model inference.

    Args:
        trt_file (str): The path to the tensorrt file.
        device_id (int): Which device to place the model.
    """

    def __init__(self, trt_file: str, device_id: int):
        super().__init__()
        from mmcv.tensorrt import TRTWrapper, load_tensorrt_plugin
        try:
            load_tensorrt_plugin()
        except (ImportError, ModuleNotFoundError):
            warnings.warn('If input model has custom op from mmcv, \
                you may have to build mmcv with TensorRT from source.')
        model = TRTWrapper(
            trt_file, input_names=['input'], output_names=['output'])
        self.device_id = device_id
        self.model = model

    def forward(self, x):
        with torch.cuda.device(self.device_id), torch.no_grad():
            seg_pred = self.model({'input': x})['output']
        seg_pred = seg_pred.detach().cpu()
        return seg_pred


class TensorRTRestorer(nn.Module):
    """A warper class for tensorrt restorer.

    Args:
        base_model (Any): The base model build from config.
        trt_file (str): The path to the tensorrt file.
        device_id (int): Which device to place the model.
    """

    def __init__(self, base_model: Any, trt_file: str, device_id: int):
        super().__init__()
        self.base_model = base_model
        restorer_generator = TensorRTRestorerGenerator(
            trt_file=trt_file, device_id=device_id)
        base_model.generator = restorer_generator

    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        return self.base_model(lq, gt=gt, test_mode=test_mode, **kwargs)


class TensorRTEditing(nn.Module):
    """A class for testing tensorrt deployment.

    Args:
        trt_file (str): The path to the tensorrt file.
        cfg (Any): The configuration of the testing, \
            decided by the config file.
        device_id (int): Which device to place the model.
    """

    def __init__(self, trt_file: str, cfg: Any, device_id: int):
        super().__init__()
        base_model = build_model(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        if isinstance(base_model, BasicRestorer):
            WrapperClass = TensorRTRestorer
        self.wrapper = WrapperClass(base_model, trt_file, device_id)

    def forward(self, **kwargs):
        return self.wrapper(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description='mmediting tester')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('model', help='input model file')
    parser.add_argument(
        'backend',
        help='backend of the model.',
        choices=['onnxruntime', 'tensorrt'])
    parser.add_argument('--out', help='output result pickle file')
    parser.add_argument(
        '--save-path',
        default=None,
        type=str,
        help='path to store images and if not given, will not save image')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # build the dataloader
    dataset = build_dataset(cfg.data.test)

    loader_cfg = {
        **dict((k, cfg.data[k]) for k in ['workers_per_gpu'] if k in cfg.data),
        **dict(
            samples_per_gpu=1,
            drop_last=False,
            shuffle=False,
            dist=distributed),
        **cfg.data.get('test_dataloader', {})
    }

    data_loader = build_dataloader(dataset, **loader_cfg)

    # build the model
    if args.backend == 'onnxruntime':
        model = ONNXRuntimeEditing(args.model, cfg=cfg, device_id=0)
    elif args.backend == 'tensorrt':
        model = TensorRTEditing(args.model, cfg=cfg, device_id=0)

    args.save_image = args.save_path is not None
    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(
        model,
        data_loader,
        save_path=args.save_path,
        save_image=args.save_image)

    print()
    # print metrics
    stats = dataset.evaluate(outputs)
    for stat in stats:
        print('Eval-{}: {}'.format(stat, stats[stat]))

    # save result pickle
    if args.out:
        print('writing results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)


if __name__ == '__main__':
    main()

    # Following strings of text style are from colorama package
    bright_style, reset_style = '\x1b[1m', '\x1b[0m'
    red_text, blue_text = '\x1b[31m', '\x1b[34m'
    white_background = '\x1b[107m'

    msg = white_background + bright_style + red_text
    msg += 'DeprecationWarning: This tool will be deprecated in future. '
    msg += blue_text + 'Welcome to use the unified model deployment toolbox '
    msg += 'MMDeploy: https://github.com/open-mmlab/mmdeploy'
    msg += reset_style
    warnings.warn(msg)
