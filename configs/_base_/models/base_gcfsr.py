_base_ = '../default_runtime.py'

# DistributedDataParallel
model_wrapper_cfg = dict(
    type='MMSeparateDistributedDataParallel', find_unused_parameters=True)

save_dir = './work_dirs'

metrics = [
    dict(type='PSNR'),
    dict(type='SSIM'),
    # dict(
    #     type='FrechetInceptionDistance',
    #     prefix='FID-Full-50k',
    #     fake_nums=50000,
    #     inception_style='StyleGAN'
    # ),
]

# evaluator
val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=800000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = dict(
    type='MultiStepLR', by_epoch=False, milestones=[800000], gamma=0.5)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        save_optimizer=True,
        by_epoch=False,
        out_dir=save_dir,
        # save_best=['FID-Full-50k/fid', 'PSNR', 'SSIM'],
        save_best=['PSNR', 'SSIM'],
        # rule=['less', 'greater', 'greater']),
        rule=['greater', 'greater']),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='EditVisualizationHook', bgr_order=True),
)
