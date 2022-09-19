# Copyright (c) OpenMMLab. All rights reserved.

from collections import OrderedDict

import torch

from mmedit.models.common import generation_init_weights
from ..builder import build_backbone
from ..registry import MODELS
from .basic_colorization import BaseColorization
from .util import encode_ab_ind, get_colorization_data, lab2rgb


@MODELS.register_module()
class FusionModel(BaseColorization):

    def __init__(
        self,
        ngf,
        output_nc,
        # avg_loss_alpha,
        ab_norm,
        l_norm,
        l_cent,
        sample_Ps,
        mask_cent,
        fusion_weight_path,
        resize_or_crop,
        loss=None,
        stage=None,
        init_type='normal',
        which_direction='AtoB',
        instance_model=None,
        full_model=None,
        fusion_model=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None):
        super(FusionModel, self).__init__(
            ngf,
            output_nc,
            # avg_loss_alpha,
            ab_norm,
            l_norm,
            l_cent,
            sample_Ps,
            mask_cent,
            init_type,
            fusion_weight_path,
            which_direction,
            loss,
            ab_max=110.,
            ab_quant=10.,
            gpu_ids='0',
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        self.stage = stage
        self.instance_model = instance_model
        self.full_model = full_model
        self.fusion_model = fusion_model

        if self.stage is not None:
            self.training = False
            self.setup_to_train()
        else:
            self.setup_to_test()

        if resize_or_crop != 'scale_width':
            torch.backends.cudnn.benchmark = True

    def init_net(self, net, init_type='xavier', gpu_ids=[]):
        if len(gpu_ids) > 0:
            assert (torch.cuda.is_available())
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)
        generation_init_weights(net, init_type)
        return net

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
        if self.stage == 'full' or self.stage == 'instance':
            loss_L1 = torch.mean(
                self.criterionL1(
                    self.fake_B_reg.type(torch.cuda.FloatTensor),
                    self.real_B.type(torch.cuda.FloatTensor)))
            loss_G = 10 * torch.mean(
                self.criterionL1(
                    self.fake_B_reg.type(torch.cuda.FloatTensor),
                    self.real_B.type(torch.cuda.FloatTensor)))
        elif self.stage == 'fusion':
            loss_L1 = torch.mean(
                self.criterionL1(
                    self.fake_B_reg.type(torch.cuda.FloatTensor),
                    self.full_real_B.type(torch.cuda.FloatTensor)))
            loss_G = 10 * torch.mean(
                self.criterionL1(
                    self.fake_B_reg.type(torch.cuda.FloatTensor),
                    self.full_real_B.type(torch.cuda.FloatTensor)))
        else:
            print('Error! Wrong stage selection!')
            exit()
        loss = {'loss_L1': loss_L1, 'loss_G': loss_G}

        return loss

    def forward_train_d(self):

        if self.stage == 'full' or self.stage == 'instance':
            self.loss_L1 = self.criterionL1(
                self.fake_B_reg.type(torch.cuda.FloatTensor),
                self.real_B.type(torch.cuda.FloatTensor))
            self.loss_G = 10 * self.criterionL1(
                self.fake_B_reg.type(torch.cuda.FloatTensor),
                self.real_B.type(torch.cuda.FloatTensor))
        elif self.stage == 'fusion':
            self.loss_L1 = self.criterionL1(
                self.fake_B_reg.type(torch.cuda.FloatTensor),
                self.full_real_B.type(torch.cuda.FloatTensor))
            self.loss_G = 10 * self.criterionL1(
                self.fake_B_reg.type(torch.cuda.FloatTensor),
                self.full_real_B.type(torch.cuda.FloatTensor))

        loss = dict(loss_L1=self.loss_L1, loss_G=self.loss_G)
        return loss

    def train_step(self, data_batch, optimizer):

        log_vars = {}

        if self.stage == 'full' or self.stage == 'instance':
            data_batch['rgb_img'] = [data_batch['rgb_img']]
            data_batch['gray_img'] = [data_batch['gray_img']]

            input_data = get_colorization_data(
                data_batch['gray_img'],
                ab_thresh=0,
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent,
                sample_PS=self.sample_Ps,
                mask_cent=self.mask_cent,
            )

            gt_data = get_colorization_data(
                data_batch['rgb_img'],
                ab_thresh=0,
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent,
                sample_PS=self.sample_Ps,
                mask_cent=self.mask_cent)

            input_data['B'] = gt_data['B']
            input_data['hint_B'] = gt_data['hint_B']
            input_data['mask_B'] = gt_data['mask_B']
            self.set_input(input_data)
            (_, self.fake_B_reg) = self.netG(self.real_A, self.hint_B,
                                             self.mask_B)

        elif self.stage == 'fusion':

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
                data_batch['cropped_gray'],
                b_thresh=0,
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent,
                sample_PS=self.sample_Ps,
                mask_cent=self.mask_cent)
            cropped_gt_data = get_colorization_data(
                data_batch['cropped_rgb'],
                ab_thresh=0,
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent,
                sample_PS=self.sample_Ps,
                mask_cent=self.mask_cent)
            full_input_data = get_colorization_data(
                data_batch['full_gray'],
                ab_thresh=0,
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent,
                sample_PS=self.sample_Ps,
                mask_cent=self.mask_cent)
            full_gt_data = get_colorization_data(
                data_batch['full_rgb'],
                ab_thresh=0,
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent,
                sample_PS=self.sample_Ps,
                mask_cent=self.mask_cent)

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

        loss = self.forward_train_d()

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

        if self.stage == 'full' or self.stage == 'instance':
            self.model_names = ['G']
            self.netG = self.init_net(build_backbone(self.instance_model))
            self.generator = self.netG

        elif self.stage == 'fusion':
            self.model_names = ['G', 'GF', 'GComp']
            self.netG = self.init_net(
                build_backbone(self.instance_model),
                init_type=self.init_type,
                gpu_ids=self.gpu_ids)
            self.netG.eval()

            self.netGF = self.init_net(
                build_backbone(self.fusion_model),
                init_type=self.init_type,
                gpu_ids=self.gpu_ids)
            self.netGF.eval()

            self.netGComp = self.init_net(
                build_backbone(self.full_model),
                init_type=self.init_type,
                gpu_ids=self.gpu_ids)
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
            print('Error Stage!')
            exit()

        self.criterionL1 = self.loss
        # self.criterionL1 = networks.L1Loss()

        # initialize average loss values
        self.avg_losses = OrderedDict()
        # self.avg_loss_alpha = self.avg_loss_alpha
        self.error_cnt = 0
        for loss_name in self.loss_names:
            self.avg_losses[loss_name] = 0

    def get_current_visuals(self):
        from collections import OrderedDict
        visual_ret = OrderedDict()
        if self.stage == 'full' or self.stage == 'instance':
            visual_ret['gray'] = lab2rgb(
                torch.cat((self.real_A.type(
                    torch.cuda.FloatTensor), torch.zeros_like(
                        self.real_B).type(torch.cuda.FloatTensor)),
                          dim=1),
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent)
            visual_ret['real'] = lab2rgb(
                torch.cat((self.real_A.type(torch.cuda.FloatTensor),
                           self.real_B.type(torch.cuda.FloatTensor)),
                          dim=1),
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent)
            visual_ret['fake_reg'] = lab2rgb(
                torch.cat((self.real_A.type(torch.cuda.FloatTensor),
                           self.fake_B_reg.type(torch.cuda.FloatTensor)),
                          dim=1),
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent)

            visual_ret['hint'] = lab2rgb(
                torch.cat((self.real_A.type(torch.cuda.FloatTensor),
                           self.hint_B.type(torch.cuda.FloatTensor)),
                          dim=1),
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent)
            visual_ret['real_ab'] = lab2rgb(
                torch.cat((torch.zeros_like(
                    self.real_A.type(torch.cuda.FloatTensor)),
                           self.real_B.type(torch.cuda.FloatTensor)),
                          dim=1),
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent)
            visual_ret['fake_ab_reg'] = lab2rgb(
                torch.cat((torch.zeros_like(
                    self.real_A.type(torch.cuda.FloatTensor)),
                           self.fake_B_reg.type(torch.cuda.FloatTensor)),
                          dim=1),
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent)

        elif self.stage == 'fusion':
            visual_ret['gray'] = lab2rgb(
                torch.cat((self.full_real_A.type(
                    torch.cuda.FloatTensor), torch.zeros_like(
                        self.full_real_B).type(torch.cuda.FloatTensor)),
                          dim=1),
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent)
            visual_ret['real'] = lab2rgb(
                torch.cat((self.full_real_A.type(torch.cuda.FloatTensor),
                           self.full_real_B.type(torch.cuda.FloatTensor)),
                          dim=1),
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent)
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
                          dim=1),
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent)

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
                          dim=1),
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent)
            visual_ret['comp_ab_reg'] = lab2rgb(
                torch.cat((torch.zeros_like(
                    self.full_real_A.type(torch.cuda.FloatTensor)),
                           self.comp_B_reg.type(torch.cuda.FloatTensor)),
                          dim=1),
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent)
            visual_ret['fake_ab_reg'] = lab2rgb(
                torch.cat((torch.zeros_like(
                    self.full_real_A.type(torch.cuda.FloatTensor)),
                           self.fake_B_reg.type(torch.cuda.FloatTensor)),
                          dim=1),
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent)
        else:
            print('Error! Wrong stage selection!')
            exit()
        return visual_ret

    def forward_test(self, **kwargs):
        output = dict()
        kwargs['full_img'][0] = kwargs['full_img'][0].cuda()
        if not kwargs['empty_box']:
            kwargs['cropped_img'][0] = kwargs['cropped_img'][0].cuda()
            box_info = kwargs['box_info'][0]
            box_info_2x = kwargs['box_info_2x'][0]
            box_info_4x = kwargs['box_info_4x'][0]
            box_info_8x = kwargs['box_info_8x'][0]
            cropped_data = get_colorization_data(
                kwargs['cropped_img'],
                ab_thresh=0,
                ab_norm=self.ab_norm,
                l_norm=self.l_norm,
                l_cent=self.l_cent,
                sample_PS=self.sample_Ps,
                mask_cent=self.mask_cent,
            )
            full_img_data = get_colorization_data(
                kwargs['full_img'],
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
                kwargs['full_img'], ab_thresh=0)
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

        self.netG = self.init_net(
            build_backbone(self.instance_model),
            init_type=self.init_type,
            gpu_ids=[0],
        )
        self.netG.eval()

        self.netGF = self.init_net(
            build_backbone(self.fusion_model),
            init_type=self.init_type,
            gpu_ids=[0],
        )
        self.netGF.eval()

        GF_path = '{0}/latest_net_GF.pth'.format(self.fusion_weight_path)
        print('load Fusion model from %s' % GF_path)
        GF_state_dict = torch.load(GF_path)
        self.netGF.load_state_dict(GF_state_dict, strict=False)

        G_path = '{0}/latest_net_G.pth'.format(self.fusion_weight_path)
        G_state_dict = torch.load(G_path)
        self.netG.module.load_state_dict(G_state_dict, strict=False)

        self.netGF.eval()
        self.netG.eval()
