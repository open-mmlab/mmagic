_base_ = './basicvsr_vimeo90k_bd.py'

experiment_name = 'basicvsr_vimeo90k_bi'
work_dir = f'./work_dirs/{experiment_name}'

train_dataloader = dict(
    dataset=dict(
        type='BasicFramesDataset', data_prefix=dict(img='BIx4', gt='GT')))

val_dataloader = dict(
    dataset=dict(
        type='BasicFramesDataset', data_prefix=dict(img='BIx4', gt='GT')))

test_dataloader = dict(
    dataset=dict(
        type='BasicFramesDataset', data_prefix=dict(img='BIx4', gt='GT')))
