# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Union, List, Dict

import torch
from mmengine.config import Config
from mmengine.optim import OptimWrapperDict

from mmedit.models.utils import (encode_ab_ind, generation_init_weights,
                                 get_colorization_data, lab2rgb)
from mmedit.structures import  EditDataSample, PixelData
from mmedit.registry import BACKBONES, COMPONENTS
from ..srgan import SRGAN


@BACKBONES.register_module()
class InstColorization(SRGAN):

    def __init__(self,
                 data_preprocessor: Union[dict, Config],
                 ngf,
                 output_nc,
                 avg_loss_alpha,
                 ab_norm,
                 ab_max,
                 ab_quant,
                 l_norm,
                 l_cent,
                 sample_Ps,
                 mask_cent,
                 insta_stage=None,
                 which_direction='AtoB',
                 generator=None,
                 loss=None,
                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None):

        super(InstColorization, self).__init__(
            generator=generator,
            data_preprocessor=data_preprocessor,
            pixel_loss=loss,
            init_cfg=init_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg)

        self.ngf = ngf
        self.output_nc = output_nc
        self.avg_loss_alpha = avg_loss_alpha
        self.mask_cent = mask_cent
        self.which_direction = which_direction

        self.encode_ab_opt = dict(
            ab_norm=ab_norm,
            ab_max=ab_max,
            ab_quant=ab_quant)

        self.colorization_data_opt = dict(
            ab_thresh=0,
            ab_norm=ab_norm,
            l_norm=l_norm,
            l_cent=l_cent,
            sample_PS=sample_Ps,
            mask_cent=mask_cent,
        )

        self.lab2rgb_opt = dict(
            ab_norm=ab_norm, l_norm=l_norm, l_cent=l_cent)

        self.convert_params = dict(
            ab_thresh=0,
            ab_norm=ab_norm,
            l_norm=l_norm,
            l_cent=l_cent,
            sample_PS=sample_Ps,
            mask_cent=mask_cent,
        )

        self.device = torch.device('cuda:{}'.format(0))

        self.insta_stage = insta_stage

        if self.insta_stage == 'full' or self.insta_stage == 'instance':
            self.training = False
            self.setup_to_train()

    def set_input(self, input):

        AtoB = self.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.hint_B = input['hint_B'].to(self.device)

        self.mask_B = input['mask_B'].to(self.device)
        self.mask_B_nc = self.mask_B + self.mask_cent

        self.real_B_enc = encode_ab_ind(
            self.real_B[:, :, ::4, ::4], **self.encode_ab_opt)

    def set_fusion_input(self, input, box_info):

        AtoB = self.which_direction == 'AtoB'
        self.full_real_A = input['A' if AtoB else 'B'].to(self.device)
        self.full_real_B = input['B' if AtoB else 'A'].to(self.device)

        self.full_hint_B = input['hint_B'].to(self.device)
        self.full_mask_B = input['mask_B'].to(self.device)

        self.full_mask_B_nc = self.full_mask_B + self.mask_cent
        self.full_real_B_enc = encode_ab_ind(
            self.full_real_B[:, :, ::4, ::4], **self.encode_ab_opt)
        self.box_info_list = box_info

    def set_forward_without_box(self, input):

        AtoB = self.which_direction == 'AtoB'
        self.full_real_A = input['A' if AtoB else 'B'].to(self.device)
        self.full_real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.full_hint_B = input['hint_B'].to(self.device)
        self.full_mask_B = input['mask_B'].to(self.device)
        self.full_mask_B_nc = self.full_mask_B + self.mask_cent
        self.full_real_B_enc = encode_ab_ind(self.full_real_B[:, :, ::4, ::4],
                                             **self.encode_ab_opt)

        (_, self.comp_B_reg) = self.netGComp(self.full_real_A,
                                             self.full_hint_B,
                                             self.full_mask_B)
        self.fake_B_reg = self.comp_B_reg

    def generator_loss(self):

        if self.insta_stage == 'full' or self.insta_stage == 'instance':
            self.loss_L1 = torch.mean(
                self.criterionL1(
                    self.fake_B_reg.type(torch.cuda.FloatTensor),
                    self.real_B.type(torch.cuda.FloatTensor)))
            self.loss_G = 10 * self.loss_L1

        elif self.insta_stage == 'fusion':
            self.loss_L1 = torch.mean(
                self.criterionL1(
                    self.fake_B_reg.type(torch.cuda.FloatTensor),
                    self.full_real_B.type(torch.cuda.FloatTensor)))
            self.loss_G = 10 * self.loss_L1

        else:
            print('Error! Wrong stage selection!')
            exit()

        self.error_cnt += 1
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                self.avg_losses[name] = float(getattr(
                    self, 'loss_' +
                          name)) + self.avg_loss_alpha * self.avg_losses[name]
                errors_ret[name] = (1 - self.avg_loss_alpha) / (
                        1 - self.avg_loss_alpha **  # noqa
                        self.error_cnt) * self.avg_losses[name]

        return errors_ret

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapperDict) -> Dict[str, torch.Tensor]:

        g_optim_wrapper = optim_wrapper['generator']
        data = self.data_preprocessor(data, True)
        batch_inputs, data_samples = data['inputs'], data['data_samples']

        log_vars = {}

        if self.insta_stage == 'full' or self.insta_stage == 'instance':
            rgb_img = [data_samples.rgb_img]
            gray_img = [data_samples.gray_img]

            input_data = get_colorization_data(gray_img,
                                               **self.colorization_data_opt)

            gt_data = get_colorization_data(rgb_img,
                                            **self.colorization_data_opt)

            input_data['B'] = gt_data['B']
            input_data['hint_B'] = gt_data['hint_B']
            input_data['mask_B'] = gt_data['mask_B']
            self.set_input(input_data)
            self.fake_B_reg = self.generator(self.real_A, self.hint_B, self.mask_B)

        elif self.insta_stage == 'fusion':
            box_info = data_samples.box_info
            box_info_2x = data_samples.box_info_2x
            box_info_4x = data_samples.box_info_4x
            box_info_8x = data_samples.box_info_8x

            cropped_input_data = get_colorization_data(
                data_samples.cropped_gray, **self.colorization_data_opt)
            cropped_gt_data = get_colorization_data(data_samples.cropped_rgb,
                                                    **self.colorization_data_opt)
            full_input_data = get_colorization_data(data_samples.full_gray,
                                                    **self.colorization_data_opt)
            full_gt_data = get_colorization_data(data_samples.full_rgb,
                                                 **self.colorization_data_opt)

            cropped_input_data['B'] = cropped_gt_data['B']
            full_input_data['B'] = full_gt_data['B']

            self.set_input(cropped_input_data)
            self.set_fusion_input(
                full_input_data,
                [box_info, box_info_2x, box_info_4x, box_info_8x])

            self.fake_B_reg = self.generator(
                self.real_A, self.hint_B, self.mask_B, self.full_real_A, self.full_hint_B,
                self.full_mask_B, self.box_info_list
            )

        optimizer['generator'].zero_grad()

        loss = self.generator_loss()

        loss_d, log_vars_d = self.parse_losses(loss)
        log_vars.update(log_vars_d)

        loss_d.backward()

        optimizer['generator'].step()

        results = self.get_current_visuals()

        output = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['rgb_img']),
            results=results)

        return output

    def setup_to_train(self):

        self.loss_names = ['G', 'L1']

        self.criterionL1 = self.loss

        # initialize average loss values
        self.avg_losses = OrderedDict()
        # self.avg_loss_alpha = self.avg_loss_alpha
        self.error_cnt = 0
        for loss_name in self.loss_names:
            self.avg_losses[loss_name] = 0

    def forward_tensor(self, inputs, data_samples, **kwargs):

        data = data_samples[0]
        full_img = data.full_gray

        if not data.empty_box:
            cropped_img = data.cropped_gray
            box_info = data.box_info
            box_info_2x = data.box_info_2x
            box_info_4x = data.box_info_4x
            box_info_8x = data.box_info_8x
            cropped_data = get_colorization_data(
                cropped_img,
                **self.convert_params
            )
            full_img_data = get_colorization_data(
                full_img,
                **self.convert_params
            )
            self.set_input(cropped_data)
            self.set_fusion_input(
                full_img_data,
                [box_info, box_info_2x, box_info_4x, box_info_8x])
        else:
            full_img_data = get_colorization_data(
                full_img, ab_thresh=0)
            self.set_forward_without_box(full_img_data)

        self.fake_B_reg = self.generator(
            self.real_A, self.hint_B, self.mask_B, self.full_real_A,
            self.full_hint_B, self.full_mask_B, self.box_info_list)

        out_img = torch.clamp(
            lab2rgb(
                torch.cat((self.full_real_A.type(torch.cuda.FloatTensor),
                           self.fake_B_reg.type(torch.cuda.FloatTensor)),
                          dim=1), **self.lab2rgb_opt), 0.0, 1.0)

        return out_img

    def forward_inference(self, inputs, data_samples=None, **kwargs):
        feats = self.forward_tensor(inputs, data_samples, **kwargs)
        predictions = []
        for idx in range(feats.shape[0]):
            batch_tensor = feats[idx] * 127.5 + 127.5
            pred_img = PixelData(data=batch_tensor.to('cpu'))
            predictions.append(
                EditDataSample(
                    pred_img=pred_img,
                    metainfo=data_samples[idx].metainfo))

        return predictions

    def get_current_visuals(self):

        visual_ret = OrderedDict()

        if self.insta_stage == 'full' or self.insta_stage == 'instance':

            visual_ret['gray'] = lab2rgb(
                torch.cat((self.real_A.type(
                    torch.cuda.FloatTensor), torch.zeros_like(
                    self.real_B).type(torch.cuda.FloatTensor)),
                    dim=1), **self.lab2rgb_opt)
            visual_ret['real'] = lab2rgb(
                torch.cat((self.real_A.type(torch.cuda.FloatTensor),
                           self.real_B.type(torch.cuda.FloatTensor)),
                          dim=1), **self.lab2rgb_opt)
            visual_ret['fake_reg'] = lab2rgb(
                torch.cat((self.real_A.type(torch.cuda.FloatTensor),
                           self.fake_B_reg.type(torch.cuda.FloatTensor)),
                          dim=1), **self.lab2rgb_opt)

            visual_ret['hint'] = lab2rgb(
                torch.cat((self.real_A.type(torch.cuda.FloatTensor),
                           self.hint_B.type(torch.cuda.FloatTensor)),
                          dim=1), **self.lab2rgb_opt)
            visual_ret['real_ab'] = lab2rgb(
                torch.cat((torch.zeros_like(
                    self.real_A.type(torch.cuda.FloatTensor)),
                           self.real_B.type(torch.cuda.FloatTensor)),
                    dim=1), **self.lab2rgb_opt)
            visual_ret['fake_ab_reg'] = lab2rgb(
                torch.cat((torch.zeros_like(
                    self.real_A.type(torch.cuda.FloatTensor)),
                           self.fake_B_reg.type(torch.cuda.FloatTensor)),
                    dim=1), **self.lab2rgb_opt)

        elif self.insta_stage == 'fusion':
            visual_ret['gray'] = lab2rgb(
                torch.cat((self.full_real_A.type(
                    torch.cuda.FloatTensor), torch.zeros_like(
                    self.full_real_B).type(torch.cuda.FloatTensor)),
                    dim=1), **self.lab2rgb_opt)
            visual_ret['real'] = lab2rgb(
                torch.cat((self.full_real_A.type(torch.cuda.FloatTensor),
                           self.full_real_B.type(torch.cuda.FloatTensor)),
                          dim=1), **self.lab2rgb_opt)
            visual_ret['comp_reg'] = lab2rgb(
                torch.cat((self.full_real_A.type(torch.cuda.FloatTensor),
                           self.comp_B_reg.type(torch.cuda.FloatTensor)),
                          dim=1), **self.lab2rgb_opt)
            visual_ret['fake_reg'] = lab2rgb(
                torch.cat((self.full_real_A.type(torch.cuda.FloatTensor),
                           self.fake_B_reg.type(torch.cuda.FloatTensor)),
                          dim=1), **self.lab2rgb_opt)

            self.instance_mask = torch.nn.functional.interpolate(
                torch.zeros([1, 1, 176, 176]),
                size=visual_ret['gray'].shape[2:],
                mode='bilinear').type(torch.cuda.FloatTensor)
            visual_ret['box_mask'] = torch.cat(
                (self.instance_mask, self.instance_mask, self.instance_mask),
                1)
            visual_ret['real_ab'] = lab2rgb(
                torch.cat((torch.zeros_like(
                    self.full_real_A.type(torch.cuda.FloatTensor)),
                           self.full_real_B.type(torch.cuda.FloatTensor)),
                    dim=1), **self.lab2rgb_opt)
            visual_ret['comp_ab_reg'] = lab2rgb(
                torch.cat((torch.zeros_like(
                    self.full_real_A.type(torch.cuda.FloatTensor)),
                           self.comp_B_reg.type(torch.cuda.FloatTensor)),
                    dim=1), **self.lab2rgb_opt)
            visual_ret['fake_ab_reg'] = lab2rgb(
                torch.cat((torch.zeros_like(
                    self.full_real_A.type(torch.cuda.FloatTensor)),
                           self.fake_B_reg.type(torch.cuda.FloatTensor)),
                    dim=1), **self.lab2rgb_opt)
        else:
            print('Error! Wrong stage selection!')
            exit()
        return visual_ret
