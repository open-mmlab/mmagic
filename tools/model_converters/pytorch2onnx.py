# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import warnings

import cv2
import numpy as np
import onnx
import onnxruntime as rt
import torch
from mmcv.onnx import register_extra_symbolics
from mmengine import Config
from mmengine.dataset import Compose
from mmengine.runner import load_checkpoint

from mmedit.apis import delete_cfg
from mmedit.registry import MODELS
from mmedit.utils import register_all_modules


def pytorch2onnx(model,
                 input,
                 model_type,
                 device,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 dynamic_export=False):
    """Export Pytorch model to ONNX model and verify the outputs are same
    between Pytorch and ONNX.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input (dict): We need to use this input to execute the model.
        opset_version (int): The onnx op version. Default: 11.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether compare the outputs between Pytorch and ONNX.
            Default: False.
    """
    model.to(device)
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    elif hasattr(model, 'forward_tensor'):
        model.forward = model.forward_tensor

    if model_type == 'mattor':
        merged = input['data_samples'].trimap.data.unsqueeze(0)
        trimap = input['data_samples'].gt_merged.data.unsqueeze(0)
        data = torch.cat((merged, trimap), dim=1).float()
        data = model.resize_inputs(data)
    elif model_type == 'image_restorer':
        data = input['inputs'].unsqueeze(0)
    elif model_type == 'inpainting':
        masks = input['data_samples'].mask.data.unsqueeze(0)
        img = input['inputs'].unsqueeze(0)
        data = torch.cat((img, masks), dim=1)
    elif model_type == 'video_restorer':
        data = input['inputs'].unsqueeze(0).float()
    data = data.to(device)

    # pytorch has some bug in pytorch1.3, we have to fix it
    # by replacing these existing op
    register_extra_symbolics(opset_version)
    dynamic_axes = None
    if dynamic_export:
        dynamic_axes = {
            'input': {
                0: 'batch',
                2: 'height',
                3: 'width'
            },
            'output': {
                0: 'batch',
                2: 'height',
                3: 'width'
            }
        }
    with torch.no_grad():
        torch.onnx.export(
            model,
            data,
            output_file,
            input_names=['input'],
            output_names=['output'],
            export_params=True,
            keep_initializers_as_inputs=False,
            verbose=show,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes)
    print(f'Successfully exported ONNX model: {output_file}')
    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        if dynamic_export:
            # scale image for dynamic shape test
            data = torch.nn.functional.interpolate(data, scale_factor=1.1)

            # concate flip image for batch test
            flip_data = data.flip(-1)
            data = torch.cat((data, flip_data), 0)

        # get pytorch output, only concern pred_alpha
        with torch.no_grad():
            pytorch_result = model(data)
        if isinstance(pytorch_result, (tuple, list)):
            pytorch_result = pytorch_result[0]
        pytorch_result = pytorch_result.detach().cpu().numpy()
        # get onnx output
        sess = rt.InferenceSession(output_file)
        onnx_result = sess.run(None, {
            'input': data.detach().cpu().numpy(),
        })
        # only concern pred_alpha value
        if isinstance(onnx_result, (tuple, list)):
            onnx_result = onnx_result[0]

        if show:
            pytorch_visualize = pytorch_result[0].transpose(1, 2, 0)
            pytorch_visualize = np.clip(pytorch_visualize, 0, 1)[:, :, ::-1]
            onnx_visualize = onnx_result[0].transpose(1, 2, 0)
            onnx_visualize = np.clip(onnx_visualize, 0, 1)[:, :, ::-1]

            cv2.imshow('PyTorch', pytorch_visualize)
            cv2.imshow('ONNXRuntime', onnx_visualize)
            cv2.waitKey()

        # check the numerical value
        assert np.allclose(
            pytorch_result, onnx_result, rtol=1e-5,
            atol=1e-5), 'The outputs are different between Pytorch and ONNX'
        print('The numerical values are same between Pytorch and ONNX')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMediting to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'model_type',
        help='what kind of model the config belong to.',
        choices=['inpainting', 'mattor', 'image_restorer', 'video_restorer'])
    parser.add_argument('img_path', help='path to input image file')
    parser.add_argument(
        '--trimap-path',
        default=None,
        help='path to input trimap file, used in mattor model')
    parser.add_argument(
        '--mask-path',
        default=None,
        help='path to input mask file, used in inpainting model')
    parser.add_argument('--num-frames', type=int, default=None)
    parser.add_argument('--sequence-length', type=int, default=None)
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        help='Whether to export onnx with dynamic axis.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model_type = args.model_type

    if model_type == 'mattor' and args.trimap_path is None:
        raise ValueError('Please set `--trimap-path` to convert mattor model.')

    assert args.opset_version == 11, 'MMEditing only support opset 11 now'

    if args.device < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)

    config = Config.fromfile(args.config)
    delete_cfg(config, key='init_cfg')

    # ONNX does not support spectral norm
    if model_type == 'mattor':
        if hasattr(config.model.backbone.encoder, 'with_spectral_norm'):
            config.model.backbone.encoder.with_spectral_norm = False
            config.model.backbone.decoder.with_spectral_norm = False
        config.test_cfg.metrics = None

    register_all_modules()

    # build the model
    model = MODELS.build(config.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # select the data pipeline
    if config.get('demo_pipeline', None):
        test_pipeline = config.demo_pipeline
    elif config.get('test_pipeline', None):
        test_pipeline = config.test_pipeline
    else:
        test_pipeline = config.val_pipeline

    # remove alpha from test_pipeline
    if model_type == 'mattor':
        keys_to_remove = ['alpha', 'ori_alpha']
    elif model_type == 'image_restorer':
        keys_to_remove = ['gt', 'gt_path']
    elif model_type == 'video_restorer':
        keys_to_remove = ['gt', 'gt_path']
    else:
        keys_to_remove = []
    for key in keys_to_remove:
        for pipeline in list(test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                test_pipeline.remove(pipeline)
            if 'keys' in pipeline and key in pipeline['keys']:
                pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0:
                    test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                pipeline['meta_keys'].remove(key)

    # prepare data
    if model_type == 'mattor':
        data = dict(merged_path=args.img_path, trimap_path=args.trimap_path)
    elif model_type == 'image_restorer':
        data = dict(img_path=args.img_path)
    elif model_type == 'inpainting':
        data = dict(gt_path=args.img_path, mask_path=args.mask_path)
    elif model_type == 'video_restorer':
        # the first element in the pipeline must be 'GenerateSegmentIndices'
        if test_pipeline[0]['type'] != 'GenerateSegmentIndices':
            raise TypeError('The first element in the pipeline must be '
                            f'"GenerateSegmentIndices", but got '
                            f'"{test_pipeline[0]["type"]}".')
        # prepare data
        # sequence_length = len(glob.glob(osp.join(args.img_path, '*')))
        lq_folder = osp.dirname(args.img_path)
        key = osp.basename(args.img_path)
        data = dict(
            img_path=lq_folder,
            gt_path='',
            key=key,
            num_frames=args.num_frames,
            sequence_length=args.sequence_length)

    # build the data pipeline
    test_pipeline = Compose(test_pipeline)
    data = test_pipeline(data)

    # convert model to onnx file
    pytorch2onnx(
        model,
        data,
        model_type,
        device,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
        dynamic_export=args.dynamic_export)

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
