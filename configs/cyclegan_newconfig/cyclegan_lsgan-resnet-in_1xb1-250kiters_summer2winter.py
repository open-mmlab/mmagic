from mmcv import KeyMapper
from mmengine.config import read_base
from torch.optim import Adam

from mmagic.datasets.transforms.formatting import PackInputs
from mmagic.engine.hooks import VisualizationHook
from mmagic.engine.schedulers.linear_lr_scheduler_with_interval import \
    LinearLrInterval
from mmagic.evaluation.metrics import TransFID, TransIS

with read_base():
    from .._base_.datasets.unpaired_imgs_256x256 import *
    from .._base_.gen_default_runtime import *
    from .._base_.models.base_cyclegan import *
train_cfg.update(dict(max_iters=250000))
domain_a = 'summer'
domain_b = 'winter'
model.update(
    dict(
        loss_config=dict(cycle_loss_weight=10., id_loss_weight=0.5),
        default_domain=domain_b,
        reachable_domains=[domain_a, domain_b],
        related_domains=[domain_a, domain_b],
        data_preprocessor=dict(
            data_keys=[f'img_{domain_a}', f'img_{domain_b}'])))
dataroot = './data/cyclegan/summer2winter_yosemite'
train_pipeline = train_dataloader['dataset']['pipeline']
val_pipeline = val_dataloader['dataset']['pipeline']
test_pipeline = test_dataloader['dataset']['pipeline']
key_mapping = dict(
    type=KeyMapper,
    mapping={
        f'img_{domain_a}': 'img_A',
        f'img_{domain_b}': 'img_B'
    },
    remapping={
        f'img_{domain_a}': f'img_{domain_a}',
        f'img_{domain_b}': f'img_{domain_b}'
    })
pack_input = dict(
    type=PackInputs,
    keys=[f'img_{domain_a}', f'img_{domain_b}'],
    data_keys=[f'img_{domain_a}', f'img_{domain_b}'])

train_pipeline += [key_mapping, pack_input]
val_pipeline += [key_mapping, pack_input]
test_pipeline += [key_mapping, pack_input]
train_dataloader.update(dict(dataset=dict(data_root=dataroot)))
val_dataloader.update(dict(dataset=dict(data_root=dataroot, test_mode=True)))
test_dataloader = val_dataloader

optim_wrapper.update(
    dict(
        generators=dict(
            optimizer=dict(type=Adam, lr=0.0002, betas=(0.5, 0.999))),
        discriminators=dict(
            optimizer=dict(type=Adam, lr=0.0002, betas=(0.5, 0.999)))))
# learning policy
param_scheduler = dict(
    type=LinearLrInterval,
    interval=1250,
    by_epoch=False,
    start_factor=0.0002,
    end_factor=0,
    begin=125000,
    end=250000)

custom_hooks = [
    dict(
        type=VisualizationHook,
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=[
            dict(type='Translation', name='trans'),
            dict(type='TranslationVal', name='trans_val')
        ])
]

num_images_a = 309
num_images_b = 238
metrics = [
    dict(
        type=TransIS,
        prefix=f'IS-{domain_a}-to-{domain_b}',
        fake_nums=num_images_b,
        fake_key=f'fake_{domain_b}',
        use_pillow_resize=False,
        resize_method='bilinear',
        inception_style='PyTorch'),
    dict(
        type=TransIS,
        prefix=f'IS-{domain_b}-to-{domain_a}',
        fake_nums=num_images_a,
        fake_key=f'fake_{domain_a}',
        use_pillow_resize=False,
        resize_method='bilinear',
        inception_style='PyTorch'),
    dict(
        type=TransFID,
        prefix=f'FID-{domain_a}-to-{domain_b}',
        fake_nums=num_images_b,
        inception_style='PyTorch',
        real_key=f'img_{domain_b}',
        fake_key=f'fake_{domain_b}'),
    dict(
        type=TransFID,
        prefix=f'FID-{domain_b}-to-{domain_a}',
        fake_nums=num_images_a,
        inception_style='PyTorch',
        real_key=f'img_{domain_a}',
        fake_key=f'fake_{domain_a}')
]
val_evaluator.update(dict(metrics=metrics))
test_evaluator.update(dict(metrics=metrics))
