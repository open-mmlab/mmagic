_base_ = [
    '../_base_/models/stable_diffusion_xl/stable_diffusion_xl_lora.py',
    '../_base_/datasets/pokemon_blip_xl.py', '../_base_/schedules/sd_10e.py',
    '../_base_/sd_default_runtime.py'
]

val_prompts = ['yoda pokemon'] * 4

model = dict(val_prompts=val_prompts)

train_dataloader = dict(batch_size=4, num_workers=4)

# hooks
custom_hooks = [
    dict(
        type='VisualizationHook',
        by_epoch=True,
        interval=1,
        fixed_input=True,
        # visualize train dataset
        vis_kwargs_list=dict(type='Data', name='fake_img'),
        n_samples=1),
    dict(type='LoRACheckpointToSaveHook')
]
