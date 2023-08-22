# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.runner import auto_fp16

from mmedit.core import psnr, ssim, tensor2img
from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS


@MODELS.register_module()
class BasicInterpolator(BaseModel):
    """Basic model for video interpolation.

    It must contain a generator that takes frames as inputs and outputs an
    interpolated frame. It also has a pixel-wise loss for training.

    The subclasses should overwrite the function `forward_train`,
    `forward_test` and `train_step`.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        required_frames (int): Required frames in each process. Default: 2
        step_frames (int): Step size of video frame interpolation. Default: 1
        pretrained (str): Path for pretrained model. Default: None.
    """
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 required_frames=2,
                 step_frames=1,
                 pretrained=None):
        super().__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # support fp16
        self.fp16_enabled = False

        # generator
        self.generator = build_backbone(generator)
        self.init_weights(pretrained)

        # loss
        self.pixel_loss = build_loss(pixel_loss)

        # Required frames in each process
        self.required_frames = required_frames
        # Step size of video frame interpolation
        self.step_frames = step_frames

    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.generator.init_weights(pretrained)

    @auto_fp16(apply_to=('inputs', ))
    def forward(self, inputs, target=None, test_mode=False, **kwargs):
        """Forward function.

        Args:
            inputs (Tensor): Tensor of input frames.
            target (Tensor): Tensor of target frame. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """

        if test_mode:
            return self.forward_test(inputs, target, **kwargs)

        return self.forward_train(inputs, target)

    def forward_train(self, inputs, target):
        """Training forward function.

        This is a basic function, interpolate a frame between the given two
        frames.

        Args:
            inputs (Tensor): Tensor of input frame(s).
            target (Tensor): Tensor of target frame(s).

        Returns:
            Tensor: Output tensor.
        """
        losses = dict()
        output = self.generator(inputs)
        loss_pix = self.pixel_loss(output, target)
        losses['loss_pix'] = loss_pix
        outputs = dict(
            losses=losses,
            num_samples=len(target.data),
            results=dict(
                inputs=inputs.cpu(), target=target.cpu(), output=output.cpu()))
        return outputs

    def evaluate(self, output, target):
        """Evaluation function.

        Args:
            output (Tensor): Model output.
            target (Tensor): GT Tensor.

        Returns:
            dict: Evaluation results.
        """
        crop_border = self.test_cfg.get('crop_border', 0)
        convert_to = self.test_cfg.get('convert_to', None)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            if output.ndim == 5:  # a sequence: (n, t, c, h, w)
                avg = []
                for i in range(0, output.size(1)):
                    output_i = tensor2img(output[:, i, :, :, :])
                    target_i = tensor2img(target[:, i, :, :, :])
                    avg.append(self.allowed_metrics[metric](
                        output_i, target_i, crop_border,
                        convert_to=convert_to))
                eval_result[metric] = np.mean(avg)
            elif output.ndim == 4:  # an image: (n, c, h, w)
                output_img = tensor2img(output)
                target_img = tensor2img(target)
                value = self.allowed_metrics[metric](
                    output_img, target_img, crop_border, convert_to=convert_to)
                eval_result[metric] = value
        return eval_result

    def forward_test(self,
                     inputs,
                     target=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        This is a basic function, interpolate a frame between the given two
        frames.

        Args:
            inputs (Tensor): Tensor of input frames.
            target (Tensor): Tensor of target frame(s).
                Default: None.
            meta (list[dict]): Meta data, such as path of target file.
                Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        output = self.generator(inputs).clamp(0, 1)
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert target is not None, (
                'evaluation with metrics must have target images.')
            results = dict(eval_result=self.evaluate(output, target))
        else:
            results = dict(inputs=inputs.cpu(), output=output.cpu())
            if target is not None:
                results['target'] = target.cpu()

        # save image
        if save_image:
            self._save_image(meta, iteration, save_path, output)

        return results

    @staticmethod
    def _save_image(meta, iteration, save_path, output):
        """Save the image.

        Args:
            meta (list[dict]): Meta data, such as path of target file.
                Default: None. These dictionaries should contain
                'target_path' (str of a path) or 'inputs_path' (list of str)
            iteration (int): Iteration for the saving image name.
                Default: None.
            save_path (str): Path to save image. Default: None.
            output (Tensor): Output image.
        """

        if output.ndim == 4:  # an image
            img_name = meta[0]['key'].replace('/', '_')
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path,
                                     f'{img_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{img_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)
        elif output.ndim == 5:  # a sequence
            folder_name = meta[0]['key'].split('/')[0]
            for i in range(0, output.size(1)):
                if isinstance(iteration, numbers.Number):
                    save_path_i = osp.join(save_path, folder_name,
                                           f'{i:08d}-{iteration + 1:06d}.png')
                elif iteration is None:
                    save_path_i = osp.join(save_path, folder_name,
                                           f'{i:08d}.png')
                else:
                    raise ValueError('iteration should be number or None, '
                                     f'but got {type(iteration)}')
                mmcv.imwrite(tensor2img(output[:, i, :, :, :]), save_path_i)

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        Args:
            img (Tensor): Input frames.

        Returns:
            Tensor: Output frame(s).
        """
        out = self.generator(img)
        return out

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        outputs.update({'log_vars': log_vars})
        return outputs

    def val_step(self, data_batch, **kwargs):
        """Validation step.

        Args:
            data_batch (dict): A batch of data.
            kwargs (dict): Other arguments for ``val_step``.

        Returns:
            dict: Returned output.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output

    def split_frames(self, input_tensors):
        """split input tensors for inference.

        Args:
            input_tensors (Tensor): Tensor of input frames with shape
                [1, t, c, h, w]

        Returns:
            Tensor: Split tensor with shape [t-1, 2, c, h, w]
        """

        num_frames = input_tensors.shape[1]

        result = [
            input_tensors[:, i:i + self.required_frames]
            for i in range(0, num_frames - self.required_frames +
                           1, self.step_frames)
        ]
        result = torch.cat(result, dim=0)

        return result

    @staticmethod
    def merge_frames(input_tensors, output_tensors):
        """merge input frames and output frames.

        This is a basic function, interpolate a frame between the given two
        frames.

        Args:
            input_tensors (Tensor): The input frames with shape [n, 2, c, h, w]
            output_tensors (Tensor): The output frames with shape
                [n, 1, c, h, w].

        Returns:
            list[np.array]: The final frames.
                in_frame, out_frame, in_frame, out_frame, in_frame ...
        """

        num_frames = input_tensors.shape[0]
        result = []
        for i in range(num_frames):
            result.append(tensor2img(input_tensors[i, 0]))
            result.append(tensor2img(output_tensors[i, 0]))
        result.append(tensor2img(input_tensors[-1, 1]))

        return result
