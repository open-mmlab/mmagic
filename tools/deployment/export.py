# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from mmedit.models import BaseMattor, BasicRestorer, build_model


def inference_with_session(sess, io_binding, output_names, input_tensor):
    device_type = input_tensor.device.type
    device_id = input_tensor.device.index
    device_id = 0 if device_id is None else device_id
    io_binding.bind_input(
        name='input',
        device_type=device_type,
        device_id=device_id,
        element_type=np.float32,
        shape=input_tensor.shape,
        buffer_ptr=input_tensor.data_ptr())
    for name in output_names:
        io_binding.bind_output(name)
    sess.run_with_iobinding(io_binding)
    pred = io_binding.copy_outputs_to_cpu()
    return pred


class ONNXRuntimeMattor(nn.Module):

    def __init__(self, sess, io_binding, output_names, base_model):
        super(ONNXRuntimeMattor, self).__init__()
        self.sess = sess
        self.io_binding = io_binding
        self.output_names = output_names
        self.base_model = base_model

    def forward(self,
                merged,
                trimap,
                meta,
                test_mode=False,
                save_image=False,
                save_path=None,
                iteration=None):
        input_tensor = torch.cat((merged, trimap), 1).contiguous()
        pred_alpha = inference_with_session(self.sess, self.io_binding,
                                            self.output_names, input_tensor)[0]

        pred_alpha = pred_alpha.squeeze()
        pred_alpha = self.base_model.restore_shape(pred_alpha, meta)
        eval_result = self.base_model.evaluate(pred_alpha, meta)

        if save_image:
            self.base_model.save_image(pred_alpha, meta, save_path, iteration)

        return {'pred_alpha': pred_alpha, 'eval_result': eval_result}


class RestorerGenerator(nn.Module):

    def __init__(self, sess, io_binding, output_names):
        super(RestorerGenerator, self).__init__()
        self.sess = sess
        self.io_binding = io_binding
        self.output_names = output_names

    def forward(self, x):
        pred = inference_with_session(self.sess, self.io_binding,
                                      self.output_names, x)[0]
        pred = torch.from_numpy(pred)
        return pred


class ONNXRuntimeRestorer(nn.Module):

    def __init__(self, sess, io_binding, output_names, base_model):
        super(ONNXRuntimeRestorer, self).__init__()
        self.sess = sess
        self.io_binding = io_binding
        self.output_names = output_names
        self.base_model = base_model
        restorer_generator = RestorerGenerator(self.sess, self.io_binding,
                                               self.output_names)
        base_model.generator = restorer_generator

    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        return self.base_model(lq, gt=gt, test_mode=test_mode, **kwargs)


class ONNXRuntimeEditing(nn.Module):

    def __init__(self, onnx_file, cfg, device_id):
        super(ONNXRuntimeEditing, self).__init__()
        ort_custom_op_path = ''
        try:
            from mmcv.ops import get_onnxruntime_op_path
            ort_custom_op_path = get_onnxruntime_op_path()
        except (ImportError, ModuleNotFoundError):
            warnings.warn('If input model has custom op from mmcv, \
                you may have to build mmcv with ONNXRuntime from source.')
        session_options = ort.SessionOptions()
        # register custom op for onnxruntime
        if osp.exists(ort_custom_op_path):
            session_options.register_custom_ops_library(ort_custom_op_path)
        sess = ort.InferenceSession(onnx_file, session_options)
        providers = ['CPUExecutionProvider']
        options = [{}]
        is_cuda_available = ort.get_device() == 'GPU'
        if is_cuda_available:
            providers.insert(0, 'CUDAExecutionProvider')
            options.insert(0, {'device_id': device_id})

        sess.set_providers(providers, options)

        self.sess = sess
        self.device_id = device_id
        self.io_binding = sess.io_binding()
        self.output_names = [_.name for _ in sess.get_outputs()]

        base_model = build_model(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

        if isinstance(base_model, BaseMattor):
            WrapperClass = ONNXRuntimeMattor
        elif isinstance(base_model, BasicRestorer):
            WrapperClass = ONNXRuntimeRestorer
        self.wrapper = WrapperClass(self.sess, self.io_binding,
                                    self.output_names, base_model)

    def forward(self, **kwargs):
        return self.wrapper(**kwargs)
