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
