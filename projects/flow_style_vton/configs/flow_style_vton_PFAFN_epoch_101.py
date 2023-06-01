model = dict(
    type='FlowStyleVTON',
    warp_model=dict(type='AFWM', input_nc=3),
    gen_model=dict(
        type='ResUnetGenerator', input_nc=7, output_nc=4, num_downs=5),
    pretrained_cfgs=dict(
        warp_model=dict(ckpt_path='ckp/aug/PFAFN_warp_epoch_101.pth'),
        gen_model=dict(ckpt_path='ckp/aug/PFAFN_gen_epoch_101.pth')))
