# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from mmagic.configs.styleganv2.stylegan2_c2_8xb4_800kiters_ffhq_256x256 import *

model.update(loss_config=dict(r1_use_apex_amp=False, g_reg_use_apex_amp=False))

train_cfg.update(max_iters=800002)

# remain to be refactored
apex_amp = dict(mode='gan', init_args=dict(opt_level='O1', num_losses=2))
resume_from = None
