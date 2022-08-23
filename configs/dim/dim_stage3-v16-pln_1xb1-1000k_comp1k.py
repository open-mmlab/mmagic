_base_ = ['./dim_stage1-v16_1xb1-1000k_comp1k.py']

save_dir = './work_dirs'
experiment_name = 'dim_stage3-v16-pln_1xb1-1000k_comp1k'

# model settings
model = dict(
    refiner=dict(type='PlainRefiner'),
    loss_refine=dict(type='CharbonnierLoss'),
    train_cfg=dict(train_backbone=True, train_refiner=True),
    test_cfg=dict(refine=True),
)

# load_from = \
#     'https://download.openmmlab.com/mmediting/mattors/dim/'\
#     'dim_stage2_v16_pln_1x1_1000k_comp1k_SAD-52.3_20200607_171909-d83c4775.pth'
