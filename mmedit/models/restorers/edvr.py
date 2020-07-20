import numbers
import os.path as osp

import mmcv

from mmedit.core import tensor2img
from ..registry import MODELS
from .basic_restorer import BasicRestorer


@MODELS.register_module()
class EDVR(BasicRestorer):
    """EDVR model for video super-resolution.

    EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EDVR, self).__init__(generator, pixel_loss, train_cfg, test_cfg,
                                   pretrained)
        self.with_tsa = generator.get('with_tsa', False)
        self.step_counter = 0  # count training steps

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        if self.step_counter == 0 and self.with_tsa:
            if self.train_cfg is None or (self.train_cfg is not None and
                                          'tsa_iter' not in self.train_cfg):
                raise KeyError(
                    'In TSA mode, train_cfg must contain "tsa_iter".')
            # only train TSA module at the beginging if with TSA module
            for k, v in self.generator.named_parameters():
                if 'fusion' not in k:
                    v.requires_grad = False

        if self.with_tsa and (self.step_counter == self.train_cfg.tsa_iter):
            # train all the parameters
            for v in self.generator.parameters():
                v.requires_grad = True

        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        self.step_counter += 1

        outputs.update({'log_vars': log_vars})
        return outputs

    def forward_dummy(self, imgs):
        """Used for computing network FLOPs.

        Args:
            imgs (Tensor): Input images.

        Returns:
            Tensor: Restored image.
        """
        out = self.generator(imgs)
        return out

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        output = self.generator(lq)
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
            gt_path = meta[0]['gt_path'][0]
            folder_name = meta[0]['key'].split('/')[0]
            frame_name = osp.splitext(osp.basename(gt_path))[0]

            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{frame_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, folder_name,
                                     f'{frame_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)

        return results
