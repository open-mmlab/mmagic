_base_ = './basicvsr_vimeo90k-bd_2xb4.py'

experiment_name = 'basicvsr_vimeo90k-bi_2xb4'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs'

train_dataloader = dict(
    dataset=dict(
        type='BasicFramesDataset', data_prefix=dict(img='BIx4', gt='GT')))

val_dataloader = dict(
    dataset=dict(
        type='BasicFramesDataset', data_prefix=dict(img='BIx4', gt='GT')))

test_dataloader = dict(
    dataset=dict(
        type='BasicFramesDataset', data_prefix=dict(img='BIx4', gt='GT')))

find_unused_parameters = True
