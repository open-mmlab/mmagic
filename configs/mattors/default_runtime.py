default_scope = 'mmedit'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10000, by_epoch=False),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),
    dist_cfg=dict(backend='nccl'),
)

# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
# custom_hooks = [dict(type='DetVisualizationHook', interval=10)]

log_level = 'INFO'
log_processor = dict(type='LogProcessor', by_epoch=False)

load_from = None
resume = False

# TODO: support auto scaling lr
