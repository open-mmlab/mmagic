# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmedit.registry import BACKBONES, COMPONENTS
from mmedit.models.utils import generation_init_weights


@BACKBONES.register_module()
class InstColorizationGenerator(nn.Module):

    def __init__(self,
                 stage,
                 instance_model=None,
                 full_model=None,
                 fusion_model=None,
                 ):

        super(InstColorizationGenerator, self).__init__()

        self.stage = stage

        if self.stage == "test":
            self.netG = COMPONENTS.build(instance_model)
            generation_init_weights(self.netG)

            self.netGF = COMPONENTS.build(fusion_model)
            generation_init_weights(self.netGF)

        elif self.stage == "instance" or stage == 'full':
            self.netG = COMPONENTS.build(instance_model)
            generation_init_weights(self.netG)

        elif self.stage == "fusion":
            self.netG = COMPONENTS.build(instance_model)
            generation_init_weights(self.netG)
            self.netG.eval()

            self.netGF = COMPONENTS.build(fusion_model)
            generation_init_weights(self.netGF)
            self.netGF.eval()

            self.netGComp = COMPONENTS.build(full_model)
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
            print('Error! Wrong stage selection!')
            exit()

    def forward(self,
                real_A,
                hint_B,
                mask_B,
                full_real_A=None,
                full_hint_B=None,
                full_mask_B=None,
                box_info_list=None
                ):
        if self.stage == 'test':
            (_, feature_map) = self.netG(real_A, hint_B, mask_B)
            fake_B_reg = self.netGF(
                full_real_A, full_hint_B, full_mask_B,
                feature_map, box_info_list
            )

            return fake_B_reg

        elif self.stage == 'full' or self.stage == 'instance':
            (_, fake_B_reg) = self.netG(real_A, hint_B, mask_B)

            return fake_B_reg

        elif self.stage == 'fusion':
            (_, self.comp_B_reg) = self.netGComp(
                full_real_A, full_hint_B, full_mask_B)

            (_, feature_map) = self.netG(real_A, hint_B, mask_B)

            fake_B_reg = self.netGF(
                full_real_A, full_hint_B, full_mask_B,
                feature_map, box_info_list)

            return fake_B_reg