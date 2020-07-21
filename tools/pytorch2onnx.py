import argparse

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
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False):
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
    merged = input['merged'].unsqueeze(0)
    trimap = input['trimap'].unsqueeze(0)
    input = torch.cat((merged, trimap), 1)
    model.forward = model.forward_dummy
    # pytorch has some bug in pytorch1.3, we have to fix it
    # by replacing these existing op
    register_extra_symbolics(opset_version)
    with torch.no_grad():
        torch.onnx.export(
            model,
            input,
            output_file,
            input_names=['cat_input'],
            export_params=True,
            keep_initializers_as_inputs=True,
            verbose=show,
            opset_version=opset_version)
    print(f'Successfully exported ONNX model: {output_file}')
    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # get pytorch output, only concern pred_alpha
        pytorch_result = model(input)
        if isinstance(pytorch_result, (tuple, list)):
            pytorch_result = pytorch_result[0]
        pytorch_result = pytorch_result.detach().numpy()
        # get onnx output
        sess = rt.InferenceSession(output_file)
        onnx_result = sess.run(None, {
            'cat_input': input.detach().numpy(),
        })
        # only concern pred_alpha value
        if isinstance(onnx_result, (tuple, list)):
            onnx_result = onnx_result[0]
        # check the numerical value
        assert np.allclose(
            pytorch_result,
            onnx_result), 'The outputs are different between Pytorch and ONNX'
        print('The numerical values are same between Pytorch and ONNX')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMediting to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img_path', help='path to input image file')
    parser.add_argument('trimap_path', help='path to input trimap file')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert args.opset_version == 11, 'MMEditing only support opset 11 now'

    config = mmcv.Config.fromfile(args.config)
    config.model.pretrained = None
    # ONNX does not support spectral norm
    if hasattr(config.model.backbone.encoder, 'with_spectral_norm'):
        config.model.backbone.encoder.with_spectral_norm = False
        config.model.backbone.decoder.with_spectral_norm = False
    config.test_cfg.metrics = None

    # build the model
    model = build_model(config.model, test_cfg=config.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # remove alpha from test_pipeline
    keys_to_remove = ['alpha', 'ori_alpha']
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
    data = dict(merged_path=args.img_path, trimap_path=args.trimap_path)
    data = test_pipeline(data)

    # conver model to onnx file
    pytorch2onnx(
        model,
        data,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify)
