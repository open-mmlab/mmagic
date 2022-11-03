_base_ = [
    '../_base_/models/base_pix2pix.py',
    '../_base_/datasets/paired_imgs_256x256_crop.py',
    '../_base_/gen_default_runtime.py'
]
# deterministic training can improve the performance of Pix2Pix
randomness = dict(deterministic=True)

source_domain = domain_a = 'aerial'
target_domain = domain_b = 'map'
# model settings
model = dict(
    default_domain=target_domain,
    reachable_domains=[target_domain],
    related_domains=[target_domain, source_domain])

train_cfg = dict(max_iters=220000)

# dataset settings
dataroot = 'data/pix2pix/maps'
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

train_dataloader = dict(dataset=dict(data_root=dataroot, test_dir='val'))
val_dataloader = dict(
    dataset=dict(data_root=dataroot, test_dir='val', test_mode=True))
test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    generators=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999))),
    discriminators=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999))))

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

# save multi best checkpoints
default_hooks = dict(checkpoint=dict(save_best='FID-Full/fid', rule='less'))

fake_nums = 1098
metrics = [
    dict(
        type='TransIS',
        prefix='IS-Full',
        fake_nums=fake_nums,
        fake_key=f'fake_{target_domain}',
        inception_style='PyTorch',
        sample_model='orig'),
    dict(
        type='TransFID',
        prefix='FID-Full',
        fake_nums=fake_nums,
        inception_style='PyTorch',
        real_key=f'img_{target_domain}',
        fake_key=f'fake_{target_domain}',
        sample_model='orig')
]

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
