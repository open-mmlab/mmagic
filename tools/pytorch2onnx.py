# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings

import cv2
import mmcv
import numpy as np
import onnx
import onnxruntime as rt
import torch
from mmcv.onnx import register_extra_symbolics
from mmcv.runner import load_checkpoint

from mmedit.datasets.pipelines import Compose
from mmedit.models import build_model


def pytorch2onnx(model,
                 input,
                 model_type,
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
    model.cpu().eval()

    if model_type == 'mattor':
        merged = input['merged'].unsqueeze(0)
        trimap = input['trimap'].unsqueeze(0)
        data = torch.cat((merged, trimap), 1)
    elif model_type == 'restorer':
        data = input['lq'].unsqueeze(0)
    model.forward = model.forward_dummy
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
        pytorch_result = pytorch_result.detach().numpy()
        # get onnx output
        sess = rt.InferenceSession(output_file)
        onnx_result = sess.run(None, {
            'input': data.detach().numpy(),
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
        choices=['inpainting', 'mattor', 'restorer', 'synthesizer'])
    parser.add_argument('img_path', help='path to input image file')
    parser.add_argument(
        '--trimap-path',
        default=None,
        help='path to input trimap file, used in mattor model')
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

    config = mmcv.Config.fromfile(args.config)
    config.model.pretrained = None
    # ONNX does not support spectral norm
    if model_type == 'mattor':
        if hasattr(config.model.backbone.encoder, 'with_spectral_norm'):
            config.model.backbone.encoder.with_spectral_norm = False
            config.model.backbone.decoder.with_spectral_norm = False
        config.test_cfg.metrics = None

    # build the model
    model = build_model(config.model, test_cfg=config.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # remove alpha from test_pipeline
    if model_type == 'mattor':
        keys_to_remove = ['alpha', 'ori_alpha']
    elif model_type == 'restorer':
        keys_to_remove = ['gt', 'gt_path']
    for key in keys_to_remove:
        for pipeline in list(config.test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                config.test_pipeline.remove(pipeline)
            if 'keys' in pipeline and key in pipeline['keys']:
                pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0:
                    config.test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                pipeline['meta_keys'].remove(key)
    # build the data pipeline
    test_pipeline = Compose(config.test_pipeline)
    # prepare data
    if model_type == 'mattor':
        data = dict(merged_path=args.img_path, trimap_path=args.trimap_path)
    elif model_type == 'restorer':
        data = dict(lq_path=args.img_path)
    data = test_pipeline(data)

    # convert model to onnx file
    pytorch2onnx(
        model,
        data,
        model_type,
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
