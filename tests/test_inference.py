import torch

from mmedit.apis import restoration_video_inference
from mmedit.models import build_model


def test_restoration_video_inference():
    if torch.cuda.is_available():
        # recurrent framework (BasicVSR)
        model = build_model(
            dict(
                type='BasicVSR',
                generator=dict(
                    type='BasicVSRNet',
                    mid_channels=64,
                    num_blocks=30,
                    spynet_pretrained='https://download.openmmlab.com/'
                    'mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth'
                ),
                pixel_loss=dict(
                    type='CharbonnierLoss', loss_weight=1.0,
                    reduction='mean'))).cuda()
        img_dir = './tests/data/vimeo90k/00001/0266'
        window_size = 0
        start_idx = 1
        filename_tmpl = 'im{}.png'

        output = restoration_video_inference(model, img_dir, window_size,
                                             start_idx, filename_tmpl)
        assert output.shape == (1, 7, 3, 256, 448)

        # sliding-window framework (EDVR)
        window_size = 5
        model = build_model(
            dict(
                type='EDVR',
                generator=dict(
                    type='EDVRNet',
                    in_channels=3,
                    out_channels=3,
                    mid_channels=64,
                    num_frames=5,
                    deform_groups=8,
                    num_blocks_extraction=5,
                    num_blocks_reconstruction=10,
                    center_frame_idx=2,
                    with_tsa=False),
                pixel_loss=dict(
                    type='CharbonnierLoss', loss_weight=1.0,
                    reduction='sum'))).cuda()
        output = restoration_video_inference(model, img_dir, window_size,
                                             start_idx, filename_tmpl)
        assert output.shape == (1, 7, 3, 256, 448)
