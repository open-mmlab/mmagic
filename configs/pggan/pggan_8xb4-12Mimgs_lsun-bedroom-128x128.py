_base_ = ['./pggan_8xb4-12Mimgs_celeba-cropped-128x128.py']

# Overwrite data configs
data_roots = {'128': './data/lsun/images/bedroom_train'}
train_dataloader = dict(batch_size=64, dataset=dict(data_roots=data_roots))
test_dataloader = dict(dataset=dict(data_root=data_roots['128']))
