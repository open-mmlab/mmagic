# Copyright (c) OpenMMLab. All rights reserved.

from collections import OrderedDict
from typing import Union

import torch
from mmengine.config import Config

from mmedit.models.utils import generation_init_weights
from mmedit.models.base_models import BaseColorization
from mmedit.registry import BACKBONES, COMPONENTS

from .util import encode_ab_ind, get_colorization_data, lab2rgb


@BACKBONES.register_module()
class INSTA(BaseColorization):

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
                 instance_model=None,
                 full_model=None,
                 fusion_model=None,
                 loss=None,
                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(INSTA, self).__init__(
            data_preprocessor=data_preprocessor,
            loss=loss,
            init_cfg=init_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg
        )

        self.ngf = ngf
        self.output_nc = output_nc
        self.avg_loss_alpha = avg_loss_alpha
        self.ab_norm = ab_norm
        self.ab_max = ab_max
        self.ab_quant = ab_quant
        self.l_norm = l_norm
        self.l_cent = l_cent
        self.sample_Ps = sample_Ps
        self.mask_cent = mask_cent
        self.which_direction = which_direction

        self.device = torch.device('cuda:{}'.format(0))

        self.instance_model = instance_model
        self.full_model = full_model
        self.fusion_model = fusion_model

        self.insta_stage = insta_stage

        if self.insta_stage == 'full' or self.insta_stage == 'instance':
            self.training = False
            self.setup_to_train()
        else:
            self.setup_to_test()

    def set_input(self, input):
        AtoB = self.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.hint_B = input['hint_B'].to(self.device)

        self.mask_B = input['mask_B'].to(self.device)
        self.mask_B_nc = self.mask_B + self.mask_cent

        self.real_B_enc = encode_ab_ind(
            self.real_B[:, :, ::4, ::4],
            ab_norm=self.ab_norm,
            ab_max=self.ab_max,
            ab_quant=self.ab_quant)

    def set_fusion_input(self, input, box_info):
        AtoB = self.which_direction == 'AtoB'
        self.full_real_A = input['A' if AtoB else 'B'].to(self.device)
        self.full_real_B = input['B' if AtoB else 'A'].to(self.device)

        self.full_hint_B = input['hint_B'].to(self.device)
        self.full_mask_B = input['mask_B'].to(self.device)

        self.full_mask_B_nc = self.full_mask_B + self.mask_cent
        self.full_real_B_enc = encode_ab_ind(
            self.full_real_B[:, :, ::4, ::4],
            ab_norm=self.ab_norm,
            ab_max=self.ab_max,
            ab_quant=self.ab_quant)
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
                                             self)

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
            self.loss_G = 10 * torch.mean(
                self.criterionL1(
                    self.fake_B_reg.type(torch.cuda.FloatTensor),
                    self.real_B.type(torch.cuda.FloatTensor)))

        elif self.insta_stage == 'fusion':
            self.loss_L1 = torch.mean(
                self.criterionL1(
                    self.fake_B_reg.type(torch.cuda.FloatTensor),
                    self.full_real_B.type(torch.cuda.FloatTensor)))
            self.loss_G = 10 * torch.mean(
                self.criterionL1(
                    self.fake_B_reg.type(torch.cuda.FloatTensor),
                    self.full_real_B.type(torch.cuda.FloatTensor)))
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
                    1 - self.avg_loss_alpha**
                    self.error_cnt) * self.avg_losses[name]

        return errors_ret

    def train_step(self, data_batch, optimizer):

        log_vars = {}

        colorization_data_opt = dict(
            ab_thresh=0,
            ab_norm=self.ab_norm,
            l_norm=self.l_norm,
            l_cent=self.l_cent,
            sample_PS=self.sample_Ps,
            mask_cent=self.mask_cent,
        )

        if self.insta_stage == 'full' or self.insta_stage == 'instance':
            data_batch['rgb_img'] = [data_batch['rgb_img']]
            data_batch['gray_img'] = [data_batch['gray_img']]

            input_data = get_colorization_data(data_batch['gray_img'],
                                               **colorization_data_opt)

            gt_data = get_colorization_data(data_batch['rgb_img'],
                                            **colorization_data_opt)

            input_data['B'] = gt_data['B']
            input_data['hint_B'] = gt_data['hint_B']
            input_data['mask_B'] = gt_data['mask_B']
            self.set_input(input_data)
            (_, self.fake_B_reg) = self.netG(self.real_A, self.hint_B,
                                             self.mask_B)

        elif self.insta_stage == 'fusion':

            data_batch['cropped_rgb'] = torch.stack(
                data_batch['cropped_rgb_list'])
            data_batch['cropped_gray'] = torch.stack(
                data_batch['cropped_gray_list'])
            data_batch['full_rgb'] = torch.stack(data_batch['full_rgb_list'])
            data_batch['full_gray'] = torch.stack(data_batch['full_gray_list'])
            data_batch['box_info'] = torch.from_numpy(
                data_batch['box_info']).type(torch.long)
            data_batch['box_info_2x'] = torch.from_numpy(
                data_batch['box_info_2x']).type(torch.long)
            data_batch['box_info_4x'] = torch.from_numpy(
                data_batch['box_info_4x']).type(torch.long)
            data_batch['box_info_8x'] = torch.from_numpy(
                data_batch['box_info_8x']).type(torch.long)

            box_info = data_batch['box_info'][0]
            box_info_2x = data_batch['box_info_2x'][0]
            box_info_4x = data_batch['box_info_4x'][0]
            box_info_8x = data_batch['box_info_8x'][0]

            cropped_input_data = get_colorization_data(
                data_batch['cropped_gray'], **colorization_data_opt)
            cropped_gt_data = get_colorization_data(data_batch['cropped_rgb'],
                                                    **colorization_data_opt)
            full_input_data = get_colorization_data(data_batch['full_gray'],
                                                    **colorization_data_opt)
            full_gt_data = get_colorization_data(data_batch['full_rgb'],
                                                 **colorization_data_opt)

            cropped_input_data['B'] = cropped_gt_data['B']
            full_input_data['B'] = full_gt_data['B']

            self.set_input(cropped_input_data)
            self.set_fusion_input(
                full_input_data,
                [box_info, box_info_2x, box_info_4x, box_info_8x])

            (_, self.comp_B_reg) = self.netGComp(self.full_real_A,
                                                 self.full_hint_B,
                                                 self.full_mask_B)
            (_, feature_map) = self.netG(self.real_A, self.hint_B, self.mask_B)
            self.fake_B_reg = self.netGF(self.full_real_A, self.full_hint_B,
                                         self.full_mask_B, feature_map,
                                         self.box_info_list)

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

        if self.insta_stage == 'full' or self.insta_stage == 'instance':
            self.model_names = ['G']
            self.netG = COMPONENTS.build(self.instance_model)
            generation_init_weights(self.netG)
            self.generator = self.netG

        elif self.insta_stage == 'fusion':
            self.model_names = ['G', 'GF', 'GComp']
            self.netG = COMPONENTS.build(self.instance_model)
            generation_init_weights(self.netG)
            self.netG.eval()

            self.netGF = COMPONENTS.build(self.fusion_model)
            generation_init_weights(self.netGF)
            self.netGF.eval()

            self.netGComp = COMPONENTS.build(self.full_model)
            generation_init_weights(self.netGComp)
            self.netGComp.eval()

            self.generator = \
                list(self.netGF.module.weight_layer.parameters()) + \
                list(self.netGF.module.weight_layer2.parameters()) + \
                list(self.netGF.module.weight_layer3.parameters()) + \
                list(self.netGF.module.weight_layer4.parameters()) + \
                list(self.netGF.module.weight_layer5.parameters()) + \
                list(self.netGF.module.weight_layer6.parameters()) + \
                list(self.netGF.module.weight_layer7.parameters()) + \
                list(self.netGF.module.weight_layer8_1.parameters()) + \
                list(self.netGF.module.weight_layer8_2.parameters()) + \
                list(self.netGF.module.weight_layer9_1.parameters()) + \
                list(self.netGF.module.weight_layer9_2.parameters()) + \
                list(self.netGF.module.weight_layer10_1.parameters()) + \
                list(self.netGF.module.weight_layer10_2.parameters()) + \
                list(self.netGF.module.model10.parameters()) + \
                list(self.netGF.module.model_out.parameters())

        else:
            # print('Error Stage!')
            # exit()
            pass

        self.criterionL1 = self.loss

        # initialize average loss values
        self.avg_losses = OrderedDict()
        # self.avg_loss_alpha = self.avg_loss_alpha
        self.error_cnt = 0
        for loss_name in self.loss_names:
            self.avg_losses[loss_name] = 0

    def get_current_visuals(self):
        from collections import OrderedDict
        visual_ret = OrderedDict()
        opt = dict(
            ab_norm=self.ab_norm, l_norm=self.l_norm, l_cent=self.l_cent)
        if self.insta_stage == 'full' or self.insta_stage == 'instance':

            visual_ret['gray'] = lab2rgb(
                torch.cat((self.real_A.type(
                    torch.cuda.FloatTensor), torch.zeros_like(
                        self.real_B).type(torch.cuda.FloatTensor)),
                          dim=1), **opt)
            visual_ret['real'] = lab2rgb(
                torch.cat((self.real_A.type(torch.cuda.FloatTensor),
                           self.real_B.type(torch.cuda.FloatTensor)),
                          dim=1), **opt)
            visual_ret['fake_reg'] = lab2rgb(
                torch.cat((self.real_A.type(torch.cuda.FloatTensor),
                           self.fake_B_reg.type(torch.cuda.FloatTensor)),
                          dim=1), **opt)

            visual_ret['hint'] = lab2rgb(
                torch.cat((self.real_A.type(torch.cuda.FloatTensor),
                           self.hint_B.type(torch.cuda.FloatTensor)),
                          dim=1), **opt)
            visual_ret['real_ab'] = lab2rgb(
                torch.cat((torch.zeros_like(
                    self.real_A.type(torch.cuda.FloatTensor)),
                           self.real_B.type(torch.cuda.FloatTensor)),
                          dim=1), **opt)
            visual_ret['fake_ab_reg'] = lab2rgb(
                torch.cat((torch.zeros_like(
                    self.real_A.type(torch.cuda.FloatTensor)),
                           self.fake_B_reg.type(torch.cuda.FloatTensor)),
                          dim=1), **opt)

        elif self.insta_stage == 'fusion':
            visual_ret['gray'] = lab2rgb(
                torch.cat((self.full_real_A.type(
                    torch.cuda.FloatTensor), torch.zeros_like(
                        self.full_real_B).type(torch.cuda.FloatTensor)),
                          dim=1), **opt)
            visual_ret['real'] = lab2rgb(
                torch.cat((self.full_real_A.type(torch.cuda.FloatTensor),
                           self.full_real_B.type(torch.cuda.FloatTensor)),
                          dim=1), **opt)
            visual_ret['comp_reg'] = lab2rgb(
                torch.cat((self.full_real_A.type(torch.cuda.FloatTensor),
                           self.comp_B_reg.type(torch.cuda.FloatTensor)),
                          dim=1),
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent)
            visual_ret['fake_reg'] = lab2rgb(
                torch.cat((self.full_real_A.type(torch.cuda.FloatTensor),
                           self.fake_B_reg.type(torch.cuda.FloatTensor)),
                          dim=1), **opt)

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
                          dim=1), **opt)
            visual_ret['comp_ab_reg'] = lab2rgb(
                torch.cat((torch.zeros_like(
                    self.full_real_A.type(torch.cuda.FloatTensor)),
                           self.comp_B_reg.type(torch.cuda.FloatTensor)),
                          dim=1), **opt)
            visual_ret['fake_ab_reg'] = lab2rgb(
                torch.cat((torch.zeros_like(
                    self.full_real_A.type(torch.cuda.FloatTensor)),
                           self.fake_B_reg.type(torch.cuda.FloatTensor)),
                          dim=1), **opt)
        else:
            print('Error! Wrong stage selection!')
            exit()
        return visual_ret

    def forward_test(self, inputs, data_samples, **kwargs):

        output = dict()
        data = data_samples[0]
        full_img= data.full_img
        if not data.empty_box:
            cropped_img = data.cropped_img
            box_info = data.box_info
            box_info_2x = data.box_info_2x
            box_info_4x = data.box_info_4x
            box_info_8x = data.box_info_8x
            cropped_data = get_colorization_data(
                cropped_img,
                ab_thresh=0,
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent,
                sample_PS=self.sample_Ps,
                mask_cent=self.mask_cent,
            )
            full_img_data = get_colorization_data(
                full_img,
                ab_thresh=0,
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent,
                sample_PS=self.sample_Ps,
                mask_cent=self.mask_cent,
            )
            self.set_input(cropped_data)
            self.set_fusion_input(
                full_img_data,
                [box_info, box_info_2x, box_info_4x, box_info_8x])
        else:
            full_img_data = get_colorization_data(
                full_img, ab_thresh=0)
            self.set_forward_without_box(full_img_data)

        (_, feature_map) = self.netG(self.real_A, self.hint_B, self.mask_B)
        self.fake_B_reg = self.netGF(self.full_real_A, self.full_hint_B,
                                     self.full_mask_B, feature_map,
                                     self.box_info_list)

        out_img = torch.clamp(
            lab2rgb(
                torch.cat((self.full_real_A.type(torch.cuda.FloatTensor),
                           self.fake_B_reg.type(torch.cuda.FloatTensor)),
                          dim=1),
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent), 0.0, 1.0)

        output['fake_img'] = out_img
        output['meta'] = None if 'meta' not in kwargs else kwargs['meta'][0]

        self.save_visualization(out_img,
                                '/mnt/ruoning/results/output_mmedit11.png')
        return output

    def setup_to_test(self):
        self.netG = COMPONENTS.build(self.instance_model)
        generation_init_weights(self.netG)

        self.netGF = COMPONENTS.build(self.fusion_model)
        generation_init_weights(self.netGF)
