_base_ = ['./wgangp_GN_1xb64-160kiters_celeba-cropped-128x128.py']

loss_config = dict(gp_norm_mode='HWC', gp_loss_weight=50)
model = dict(loss_config=loss_config)

batch_size = 64
data_root = './data/lsun/images/bedroom_train'
train_dataloader = dict(
    batch_size=batch_size, dataset=dict(data_root=data_root))

val_dataloader = dict(batch_size=batch_size, dataset=dict(data_root=data_root))

test_dataloader = dict(
    batch_size=batch_size, dataset=dict(data_root=data_root))
