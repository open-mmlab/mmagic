# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from typing import Iterable, Optional

import cv2
import mmcv
import numpy as np
import onnxruntime as ort
import torch
from mmcv.ops import get_onnxruntime_op_path
from mmcv.tensorrt import (TRTWrapper, is_tensorrt_plugin_loaded, onnx2trt,
                           save_trt_engine)

from mmedit.datasets.pipelines import Compose


def get_GiB(x: int):
    """return x GiB."""
    return x * (1 << 30)


def _prepare_input_img(model_type: str,
                       img_path: str,
                       config: dict,
                       rescale_shape: Optional[Iterable] = None) -> dict:
    """Prepare the input image.

    Args:
        model_type (str): which kind of model config belong to, \
            one of ['inpainting', 'mattor', 'restorer', 'synthesizer'].
        img_path (str): image path to show or verify.
        config (dict): MMCV config, determined by the inpupt config file.
        rescale_shape (Optional[Iterable]): to rescale the shape of the \
            input tensor.

    Returns:
        dict: {'imgs': imgs, 'img_metas': img_metas}
    """
    # remove alpha from test_pipeline
    model_type = model_type
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
        raise RuntimeError('Invalid model_type!', model_type)
    if model_type == 'restorer':
        data = dict(lq_path=img_path)

    data = test_pipeline(data)

    if model_type == 'restorer':
        imgs = data['lq']
    else:
        imgs = data['img']
    img_metas = [data['meta']]

    if rescale_shape is not None:
        for img_meta in img_metas:
            img_meta['ori_shape'] = tuple(rescale_shape) + (3, )

    mm_inputs = {'imgs': imgs, 'img_metas': img_metas}

    return mm_inputs


def onnx2tensorrt(onnx_file: str,
                  trt_file: str,
                  config: dict,
                  input_config: dict,
                  model_type: str,
                  img_path: str,
                  fp16: bool = False,
                  verify: bool = False,
                  show: bool = False,
                  workspace_size: int = 1,
                  verbose: bool = False):
    """Convert ONNX model to TensorRT model.

    Args:
        onnx_file (str): the path of the input ONNX file.
        trt_file (str): the path to output the TensorRT file.
        config (dict): MMCV configuration.
        input_config (dict): contains min_shape, max_shape and \
            input image path.
        fp16 (bool): whether to enable fp16 mode.
        verify (bool): whether to verify the outputs of TensorRT \
            and ONNX are same.
        show (bool): whether to show the outputs of TensorRT and ONNX.
        verbose (bool): whether to print the log when generating \
            TensorRT model.
    """
    import tensorrt as trt
    min_shape = input_config['min_shape']
    max_shape = input_config['max_shape']
    # create trt engine and wrapper
    opt_shape_dict = {'input': [min_shape, min_shape, max_shape]}
    max_workspace_size = get_GiB(workspace_size)
    trt_engine = onnx2trt(
        onnx_file,
        opt_shape_dict,
        log_level=trt.Logger.VERBOSE if verbose else trt.Logger.ERROR,
        fp16_mode=fp16,
        max_workspace_size=max_workspace_size)
    save_dir, _ = osp.split(trt_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    save_trt_engine(trt_engine, trt_file)
    print(f'Successfully created TensorRT engine: {trt_file}')

    if verify:
        inputs = _prepare_input_img(
            model_type=model_type, img_path=img_path, config=config)

        imgs = inputs['imgs']
        img_list = [imgs.unsqueeze(0)]

        if max_shape[0] > 1:
            # concate flip image for batch test
            flip_img_list = [_.flip(-1) for _ in img_list]
            img_list = [
                torch.cat((ori_img, flip_img), 0)
                for ori_img, flip_img in zip(img_list, flip_img_list)
            ]

        # Get results from ONNXRuntime
        ort_custom_op_path = get_onnxruntime_op_path()
        session_options = ort.SessionOptions()
        if osp.exists(ort_custom_op_path):
            session_options.register_custom_ops_library(ort_custom_op_path)
        sess = ort.InferenceSession(onnx_file, session_options)
        sess.set_providers(['CPUExecutionProvider'], [{}])  # use cpu mode
        onnx_output = sess.run(['output'],
                               {'input': img_list[0].detach().numpy()})[0][0]

        # Get results from TensorRT
        trt_model = TRTWrapper(trt_file, ['input'], ['output'])
        with torch.no_grad():
            trt_outputs = trt_model({'input': img_list[0].contiguous().cuda()})
        trt_output = trt_outputs['output'][0].cpu().detach().numpy()

        if show:
            onnx_visualize = onnx_output.transpose(1, 2, 0)
            onnx_visualize = np.clip(onnx_visualize, 0, 1)[:, :, ::-1]
            trt_visualize = trt_output.transpose(1, 2, 0)
            trt_visualize = np.clip(trt_visualize, 0, 1)[:, :, ::-1]

            cv2.imshow('ONNXRuntime', onnx_visualize)
            cv2.imshow('TensorRT', trt_visualize)
            cv2.waitKey()

        np.testing.assert_allclose(
            onnx_output, trt_output, rtol=1e-03, atol=1e-05)
        print('TensorRT and ONNXRuntime output all close.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMSegmentation models from ONNX to TensorRT')
    parser.add_argument('config', help='Config file of the model')
    parser.add_argument(
        'model_type',
        help='what kind of model the config belong to.',
        choices=['inpainting', 'mattor', 'restorer', 'synthesizer'])
    parser.add_argument('img_path', type=str, help='Image for test')
    parser.add_argument('onnx_file', help='Path to the input ONNX model')
    parser.add_argument(
        '--trt-file',
        type=str,
        help='Path to the output TensorRT engine',
        default='tmp.trt')
    parser.add_argument(
        '--max-shape',
        type=int,
        nargs=4,
        default=[1, 3, 512, 512],
        help='Maximum shape of model input.')
    parser.add_argument(
        '--min-shape',
        type=int,
        nargs=4,
        default=[1, 3, 32, 32],
        help='Minimum shape of model input.')
    parser.add_argument(
        '--workspace-size',
        type=int,
        default=1,
        help='Max workspace size in GiB')
    parser.add_argument('--fp16', action='store_true', help='Enable fp16 mode')
    parser.add_argument(
        '--show', action='store_true', help='Whether to show output results')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the outputs of ONNXRuntime and TensorRT')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Whether to verbose logging messages while creating \
                TensorRT engine.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    assert is_tensorrt_plugin_loaded(), 'TensorRT plugin should be compiled.'
    args = parse_args()

    # check arguments
    assert osp.exists(args.config), 'Config {} not found.'.format(args.config)
    assert osp.exists(args.onnx_file), \
        'ONNX model {} not found.'.format(args.onnx_file)
    assert args.workspace_size >= 0, 'Workspace size less than 0.'
    for max_value, min_value in zip(args.max_shape, args.min_shape):
        assert max_value >= min_value, \
            'max_shape should be larger than min shape'

    config = mmcv.Config.fromfile(args.config)
    config.model.pretrained = None

    input_config = {
        'min_shape': args.min_shape,
        'max_shape': args.max_shape,
        'input_path': args.img_path
    }

    onnx2tensorrt(
        args.onnx_file,
        args.trt_file,
        config,
        input_config,
        model_type=args.model_type,
        img_path=args.img_path,
        fp16=args.fp16,
        verify=args.verify,
        show=args.show,
        workspace_size=args.workspace_size,
        verbose=args.verbose)

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
