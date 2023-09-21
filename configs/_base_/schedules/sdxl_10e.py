optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',
    optimizer=dict(
        type='Adafactor',
        lr=1e-5,
        weight_decay=1e-2,
        scale_parameter=False,
        relative_step=False),
    clip_grad=dict(max_norm=1.0),
    accumulative_counts=1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=10)
val_cfg = None
test_cfg = None
