# Copyright (c) OpenMMLab. All rights reserved.
import functools

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

from ..builder import build_backbone
from ..registry import MODELS
from .basic_colorization import BaseColorization
from .util import encode_ab_ind, get_colorization_data, lab2rgb


@MODELS.register_module()
class FusionModel(BaseColorization):

    def name(self):
        return 'FusionModel'

    @staticmethod
    def modify_commandline_ons(parser, is_train=True):
        return parser

    def __init__(self,
                 ab_norm,
                 l_norm,
                 l_cent,
                 sample_Ps,
                 mask_cent,
                 init_type,
                 fusion_weight_path,
                 full_model,
                 instance_model,
                 fusion_model,
                 which_direction,
                 ab_max=110.,
                 ab_quant=10.,
                 gpu_ids='0',
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        # BaseColorization.initialize(self)
        self.ab_norm = ab_norm
        self.l_norm = l_norm
        self.l_cent = l_cent
        self.sample_Ps = sample_Ps
        self.mask_cent = mask_cent
        self.fusion_weight_path = fusion_weight_path
        self.which_direction = which_direction
        self.ab_max = ab_max
        self.ab_quant = ab_quant
        self.model_names = ['G', 'GF']
        self.gpu_ids = gpu_ids
        self.device = torch.device('cuda:{}'.format(
            self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        # load/define networks

        self.netG = self.init_net(
            build_backbone(instance_model),
            init_type=init_type,
            gpu_ids=self.gpu_ids)
        self.netG.eval()

        self.netGF = self.init_net(
            build_backbone(fusion_model),
            init_type=init_type,
            gpu_ids=self.gpu_ids)
        self.netGF.eval()

        self.netGComp = self.init_net(
            build_backbone(full_model),
            init_type=init_type,
            gpu_ids=self.gpu_ids)
        self.netGComp.eval()

    def get_norm_layer(self, norm_type='instance'):
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        elif norm_type == 'none':
            norm_layer = None
        else:
            raise NotImplementedError('normalization layer [%s] is not found' %
                                      norm_type)
        return norm_layer

    def _init_weights(self, net, init_type='xavier', gain=0.02):

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                         or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' %
                        init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)

    def init_net(self, net, init_type='xavier', gpu_ids=[]):
        if len(gpu_ids) > 0:
            assert (torch.cuda.is_available())
            device = torch.device('cuda:0')
            net.to(device)
            net = torch.nn.DataParallel(net)
        self._init_weights(net, init_type)
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

    def forward_test(self, full_img, empty_box, cropped_img, box_info,
                     box_info_2x, box_info_4x, box_info_8x):
        self.setup_to_test()
        output = dict()
        full_img[0] = full_img[0].cuda()
        if empty_box[0] == 0:
            cropped_img[0] = cropped_img[0].cuda()
            box_info = box_info[0]
            box_info_2x = box_info_2x[0]
            box_info_4x = box_info_4x[0]
            box_info_8x = box_info_8x[0]
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
            full_img_data = get_colorization_data(full_img, ab_thresh=0)
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
        out_img = np.transpose(out_img.cpu().data.numpy()[0], (1, 2, 0))
        output['fake_img'] = out_img

        return output

    def forward(self, test_mode, full_img, empty_box, cropped_img, box_info,
                box_info_2x, box_info_4x, box_info_8x, **kwargs):
        if test_mode:
            output = self.forward_test(full_img, empty_box, cropped_img,
                                       box_info, box_info_2x, box_info_4x,
                                       box_info_8x)
        else:
            output = None
        output['meta'] = None if 'meta' not in kwargs else kwargs['meta'][0]

        return output

    def setup_to_test(self):
        GF_path = '{0}/latest_net_GF.pth'.format(self.fusion_weight_path)
        print('load Fusion model from %s' % GF_path)
        GF_state_dict = torch.load(GF_path)

        # G_path =
        #       'checkpoints/coco_finetuned_mask_256/latest_net_G.pth'
        # fine tuned on cocostuff
        G_path = '{0}/latest_net_G.pth'.format(self.fusion_weight_path)
        G_state_dict = torch.load(G_path)

        # GComp_path =
        #       'checkpoints/siggraph_retrained/latest_net_G.pth'
        # original net
        # GComp_path =
        #       'checkpoints/coco_finetuned_mask_256/latest_net_GComp.pth'
        # fine tuned on cocostuff
        GComp_path = '{0}/latest_net_GComp.pth'.format(self.fusion_weight_path)
        GComp_state_dict = torch.load(GComp_path)

        self.netGF.load_state_dict(GF_state_dict, strict=False)
        # todo 此处删除了module.load_state_dict
        self.netG.module.load_state_dict(G_state_dict, strict=False)
        self.netGComp.module.load_state_dict(GComp_state_dict, strict=False)
        self.netGF.eval()
        self.netG.eval()
        self.netGComp.eval()
