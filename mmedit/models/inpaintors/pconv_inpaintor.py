import os.path as osp
from pathlib import Path

import mmcv
import torch

from mmedit.core import tensor2img
from ..registry import MODELS
from .one_stage import OneStageInpaintor


@MODELS.register_module()
class PConvInpaintor(OneStageInpaintor):

    def forward_test(self,
                     masked_img,
                     mask,
                     save_image=False,
                     save_path=None,
                     iteration=None,
                     **kwargs):
        """Forward function for testing.

        Args:
            masked_img (torch.Tensor): Tensor with shape of (n, 3, h, w).
            mask (torch.Tensor): Tensor with shape of (n, 1, h, w).
            save_image (bool, optional): If True, results will be saved as
                image. Defaults to False.
            save_path (str, optional): If given a valid str, the reuslts will
                be saved in this path. Defaults to None.
            iteration (int, optional): Iteration number. Defaults to None.

        Returns:
            dict: Contain output results and eval metrics (if have).
        """
        mask_input = mask.expand_as(masked_img)
        mask_input = 1. - mask_input

        fake_res, final_mask = self.generator(masked_img, mask_input)
        fake_img = fake_res * mask + masked_img * (1. - mask)

        output = dict()
        eval_results = {}
        if self.eval_with_metrics:
            gt_img = kwargs['gt_img']
            data_dict = dict(gt_img=gt_img, fake_res=fake_res, mask=mask)
            for metric_name in self.test_cfg['metrics']:
                if metric_name in ['ssim', 'psnr']:
                    eval_results[metric_name] = self._eval_metrics[
                        metric_name](tensor2img(fake_img, min_max=(-1, 1)),
                                     tensor2img(gt_img, min_max=(-1, 1)))
                else:
                    eval_results[metric_name] = self._eval_metrics[
                        metric_name]()(data_dict).item()
            output['eval_results'] = eval_results
        else:
            output['fake_res'] = fake_res
            output['fake_img'] = fake_img
            output['final_mask'] = final_mask

        output['meta'] = None if 'meta' not in kwargs else kwargs['meta'][0]
        if save_image:
            assert save_image and save_path is not None, (
                'Save path should been given')
            assert output['meta'] is not None, (
                'Meta information should be given to save image.')

            tmp_filename = output['meta']['gt_img_path']
            filestem = Path(tmp_filename).stem
            if iteration is not None:
                filename = f'{filestem}_{iteration}.png'
            else:
                filename = f'{filestem}.png'
            mmcv.mkdir_or_exist(save_path)
            if kwargs.get('gt_img', None) is not None:
                img_list = [kwargs['gt_img']]
            else:
                img_list = []
            img_list.extend(
                [masked_img,
                 mask.expand_as(masked_img), fake_res, fake_img])
            img = torch.cat(img_list, dim=3).cpu()
            self.save_visualization(img, osp.join(save_path, filename))
            output['save_img_path'] = osp.abspath(
                osp.join(save_path, filename))
        return output

    def train_step(self, data_batch, optimizer):
        """Train step function.

        In this function, the inpaintor will finish the train step following
        the pipeline:

            1. get fake res/image
            2. optimize discriminator (if have)
            3. optimize generator

        If `self.train_cfg.disc_step > 1`, the train step will contain multiple
        iterations for optimizing discriminator with different input data and
        only one iteration for optimizing gerator after `disc_step` iterations
        for discriminator.

        Args:
            data_batch (torch.Tensor): Batch of data as input.
            optimizer (dict[torch.optim.Optimizer]): Dict with optimizers for
                generator and discriminator (if have).

        Returns:
            dict: Dict with loss, information for logger, the number of \
                samples and results for visualization.
        """
        log_vars = {}

        gt_img = data_batch['gt_img']
        mask = data_batch['mask']
        masked_img = data_batch['masked_img']

        mask_input = mask.expand_as(gt_img)
        mask_input = 1. - mask_input

        fake_res, final_mask = self.generator(masked_img, mask_input)
        fake_img = gt_img * (1. - mask) + fake_res * mask

        results, g_losses = self.generator_loss(fake_res, fake_img, data_batch)
        loss_g_, log_vars_g = self.parse_losses(g_losses)
        log_vars.update(log_vars_g)
        optimizer['generator'].zero_grad()
        loss_g_.backward()
        optimizer['generator'].step()

        results.update(dict(final_mask=final_mask))
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['gt_img'].data),
            results=results)

        return outputs

    def forward_dummy(self, x):
        mask = x[:, -3:, ...].clone()
        x = x[:, :-3, ...]
        res, _ = self.generator(x, mask)

        return res
