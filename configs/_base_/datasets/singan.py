dataset_type = 'SinGANDataset'

pipeline = [dict(type='PackEditInputs')]
dataset = dict(
    type=dataset_type,
    data_root=None,
    min_size=25,
    max_size=250,
    scale_factor_init=0.75,
    pipeline=pipeline)

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=dataset,
    sampler=None,
    persistent_workers=False)
