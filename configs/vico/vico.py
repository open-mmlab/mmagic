_base_ = '../_base_/gen_default_runtime.py'

# config for model
stable_diffusion_v15_url = 'runwayml/stable-diffusion-v1-5'

# val_prompts = [
#     'a sks dog in basket', 'a sks dog on the mountain',
#     'a sks dog beside a swimming pool', 'a sks dog on the desk',
#     'a sleeping sks dog', 'a screaming sks dog', 'a man in the garden'
# ]

data_root = './data/vico'
concept_dir = 'wooden_pot'
image_cross_layers = [
    # down blocks (2x transformer block) * (3x down blocks) = 6
    0,
    0,
    0,
    0,
    0,
    0,
    # mid block (1x transformer block) * (1x mid block)= 1
    0,
    # up blocks (3x transformer block) * (3x up blocks) = 9
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
]
reg_loss_weight: float = 5e-4
placeholder: str = 'S*'
initialize_token: str = 'pot'
num_vectors_per_token: int = 1

model = dict(
    type='ViCo',
    vae=dict(
        type='AutoencoderKL',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='vae'),
    unet=dict(
        type='UNet2DConditionModel',
        subfolder='unet',
        from_pretrained=stable_diffusion_v15_url),
    text_encoder=dict(
        type='ClipWrapper',
        clip_type='huggingface',
        pretrained_model_name_or_path=stable_diffusion_v15_url,
        subfolder='text_encoder'),
    tokenizer=stable_diffusion_v15_url,
    scheduler=dict(
        type='DDPMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    test_scheduler=dict(
        type='DDIMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    data_preprocessor=dict(type='DataPreprocessor', data_keys=None),
    image_cross_layers=image_cross_layers,
    reg_loss_weight=reg_loss_weight,
    placeholder=placeholder,
    initialize_token=initialize_token,
    num_vectors_per_token=num_vectors_per_token,
)

train_cfg = dict(max_iters=400)

parawise_cfg = dict(
    custom_keys={
        '.*image_cross_attention': dict(lr_mult=2e-3),
        '.*trainable_embeddings': dict(lr_mult=1.0)
    })
optim_wrapper = dict(
    # modules='.*image_cross_attention|.*trainable_embeddings',
    optimizer=dict(type='AdamW', lr=5e-3, weight_decay=0.01),
    parawise_cfg=parawise_cfg,
    accumulative_counts=1)

# optim_wrapper = {
#     ".*image_cross_attention": {
#         'type': 'OptimWrapper',
#         'optimizer': {
#             'type': 'AdamW',
#             'lr': 1e-5,
#             'betas': (0.9, 0.99),
#             'weight_decay': 0.01
#         }
#     },
#     ".*trainable_embeddings": {
#         'type': 'OptimWrapper',
#         'optimizer': {
#             'type': 'AdamW',
#             'lr': 0.005,
#             'betas': (0.9, 0.99),
#             'weight_decay': 0.01
#         }
#     }
# }

pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='img_ref', channel_order='rgb'),
    dict(type='Resize', keys=['img', 'img_ref'], scale=(512, 512)),
    dict(
        type='PackInputs',
        keys=['img', 'img_ref'],
        data_keys='prompt',
        meta_keys=[
            'img_channel_order', 'img_color_type', 'img_ref_channel_order',
            'img_ref_color_type'
        ])
]
dataset = dict(
    type='ViCoDataset',
    data_root=data_root,
    # TODO: rename to instance
    concept_dir=concept_dir,
    placeholder=placeholder,
    pipeline=pipeline)
train_dataloader = dict(
    dataset=dataset,
    num_workers=16,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    persistent_workers=True,
    batch_size=4)
val_cfg = val_evaluator = val_dataloader = None
test_cfg = test_evaluator = test_dataloader = None

# hooks
default_hooks = dict(logger=dict(interval=10))
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=50,
        fixed_input=True,
        # visualize train dataset
        vis_kwargs_list=dict(type='Data', name='fake_img'),
        n_samples=1)
]
