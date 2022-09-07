_base_ = [
    '../_base_/models/base_cyclegan.py',
    '../_base_/datasets/unpaired_imgs_256x256.py',
    '../_base_/gen_default_runtime.py'
]
train_cfg = dict(max_iters=270000)

domain_a = 'horse'
domain_b = 'zebra'

model = dict(
    loss_config=dict(cycle_loss_weight=10., id_loss_weight=0.),
    default_domain=domain_b,
    reachable_domains=[domain_a, domain_b],
    related_domains=[domain_a, domain_b])

dataroot = './data/cyclegan/horse2zebra'
train_pipeline = _base_.train_dataloader.dataset.pipeline
val_pipeline = _base_.val_dataloader.dataset.pipeline
test_pipeline = _base_.test_dataloader.dataset.pipeline

key_mapping = dict(
    type='KeyMapper',
    mapping={
        f'img_{domain_a}': 'img_A',
        f'img_{domain_b}': 'img_B'
    },
    remapping={
        f'img_{domain_a}': f'img_{domain_a}',
        f'img_{domain_b}': f'img_{domain_b}'
    })
pack_input = dict(
    type='PackEditInputs', keys=[f'img_{domain_a}', f'img_{domain_b}'])

train_pipeline += [key_mapping, pack_input]
val_pipeline += [key_mapping, pack_input]
test_pipeline += [key_mapping, pack_input]

train_dataloader = dict(dataset=dict(data_root=dataroot))
val_dataloader = dict(dataset=dict(data_root=dataroot, test_mode=True))
test_dataloader = val_dataloader

optim_wrapper = dict(
    generators=dict(
        optimizer=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999))),
    discriminators=dict(
        optimizer=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999))))

param_scheduler = dict(
    type='LinearLrInterval',
    interval=1350,
    by_epoch=False,
    start_factor=0.0002,
    end_factor=0,
    begin=135000,
    end=270000)

custom_hooks = [
    dict(
        type='GenVisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=[
            dict(type='Translation', name='trans'),
            dict(type='TranslationVal', name='trans_val')
        ])
]

num_images = 140
metrics = [
    dict(
        type='TransIS',
        prefix='IS-Full',
        fake_nums=num_images,
        fake_key='fake_zebra',
        inception_style='PyTorch'),
    dict(
        type='TransFID',
        prefix='FID-Full',
        fake_nums=num_images,
        inception_style='PyTorch',
        real_key='img_zebra',
        fake_key='fake_zebra')
]

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
