default_scope = 'mmagic'

# configure for default hooks
default_hooks = dict(
    # record time of every iteration.
    timer=dict(type='IterTimerHook'),
    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),
    # save checkpoint per 10000 iterations
    checkpoint=dict(
        type='CheckpointHook',
        interval=10000,
        by_epoch=False,
        max_keep_ckpts=20,
        save_optimizer=True))

# config for environment
env_cfg = dict(
    # whether to enable cudnn benchmark.
    cudnn_benchmark=True,
    # set multi process parameters.
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters.
    dist_cfg=dict(backend='nccl'))

# set log level
log_level = 'INFO'
log_processor = dict(type='LogProcessor', by_epoch=True)

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = None

# config for model wrapper
model_wrapper_cfg = dict(
    type='MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=False)

# set visualizer
vis_backends = [dict(type='VisBackend')]
visualizer = dict(type='Visualizer', vis_backends=vis_backends)

randomness = dict(seed=None, deterministic=False)
