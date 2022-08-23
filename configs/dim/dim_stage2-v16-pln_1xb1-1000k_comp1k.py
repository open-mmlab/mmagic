_base_ = ['./dim_stage1-v16_1xb1-1000k_comp1k.py']
save_dir = './work_dirs/'
experiment_name = 'dim_stage2-v16-pln_1xb1-1000k_comp1k'

# model settings
model = dict(
    refiner=dict(type='PlainRefiner'),
    loss_refine=dict(type='CharbonnierLoss'),
    train_cfg=dict(train_backbone=False, train_refiner=True),
    test_cfg=dict(refine=True),
)

# load_from = \
#     'https://download.openmmlab.com/mmediting/mattors/dim/'\
#     'dim_stage1_v16_1x1_1000k_comp1k_SAD-53.8_20200605_140257-979a420f.pth'
