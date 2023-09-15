_base_ = [
    '../_base_/models/stable_diffusion/stable_diffusion_v15.py',
    '../_base_/datasets/pokemon_blip.py', '../_base_/schedules/sd_50e.py',
    '../_base_/sd_default_runtime.py'
]

val_prompts = ['yoda pokemon'] * 4

model = dict(val_prompts=val_prompts)

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=3,
    ))
custom_hooks = [
    dict(
        type='VisualizationHook',
        by_epoch=True,
        interval=1,
        fixed_input=True,
        # visualize train dataset
        vis_kwargs_list=dict(type='Data', name='fake_img'),
        n_samples=1)
]
