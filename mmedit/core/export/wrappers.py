import numbers
import os.path as osp

import mmcv
import numpy as np
import onnxruntime as ort
import torch

from mmedit.core import psnr, ssim, tensor2img


class ONNXRuntimeRestorer(torch.nn.Module):

    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self, onnx_file, test_cfg, device_id):
        super(ONNXRuntimeRestorer, self).__init__()
        session_options = ort.SessionOptions()
        sess = ort.InferenceSession(onnx_file, session_options)
        providers = ['CPUExecutionProvider']
        options = [{}]
        is_cuda_available = ort.get_device() == 'GPU'
        if is_cuda_available:
            providers.insert(0, 'CUDAExecutionProvider')
            options.insert(0, {'device_id': device_id})

        sess.set_providers(providers, options)

        self.test_cfg = test_cfg
        self.sess = sess
        self.device_id = device_id
        self.io_binding = sess.io_binding()
        self.output_names = [_.name for _ in sess.get_outputs()]
        self.is_cuda_available = is_cuda_available

    def evaluate(self, output, gt):
        crop_border = self.test_cfg.crop_border
        output = tensor2img(output)
        gt = tensor2img(gt)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            eval_result[metric] = self.allowed_metrics[metric](output, gt,
                                                               crop_border)
        return eval_result

    def forward(self,
                lq,
                gt=None,
                test_mode=True,
                meta=None,
                save_image=False,
                save_path=None,
                iteration=None):
        assert test_mode, 'Only support test_mode == False'
        # set io binding for inputs/outputs
        device_type = 'cuda' if self.is_cuda_available else 'cpu'
        if not self.is_cuda_available:
            lq = lq.cpu()
        self.io_binding.bind_input(
            name='input',
            device_type=device_type,
            device_id=self.device_id,
            element_type=np.float32,
            shape=lq.shape,
            buffer_ptr=lq.data_ptr())
        for name in self.output_names:
            self.io_binding.bind_output(name)

        # run session to get outputs
        self.sess.run_with_iobinding(self.io_binding)
        output = self.io_binding.copy_outputs_to_cpu()[0]
        output = torch.from_numpy(output)

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)

        return results
