_base_ = ['dim_stage1_v16_1x1_1000k_comp1k.py']

# model settings
model = dict(
    refiner=dict(type='PlainRefiner'),
    loss_refine=dict(type='CharbonnierLoss'),
    train_cfg=dict(train_backbone=True, train_refiner=True),
    test_cfg=dict(refine=True),
)

load_from = './checkpoints/dim_stage2_v16_pln_1x1_1000k_comp1k_SAD-52.3_20200607_171909-d83c4775.pth'  # noqa: E501
