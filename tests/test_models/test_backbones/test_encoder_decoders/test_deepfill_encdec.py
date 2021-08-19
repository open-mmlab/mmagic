# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.backbones import DeepFillEncoderDecoder, GLEncoderDecoder
from mmedit.models.components import DeepFillRefiner


def test_deepfill_encdec():
    encdec = DeepFillEncoderDecoder()
    assert isinstance(encdec.stage1, GLEncoderDecoder)
    assert isinstance(encdec.stage2, DeepFillRefiner)

    if torch.cuda.is_available():
        img = torch.rand((2, 3, 256, 256)).cuda()
        mask = img.new_zeros((2, 1, 256, 256))
        mask[..., 20:100, 30:120] = 1.
        input_x = torch.cat([img, torch.ones_like(mask), mask], dim=1)
        encdec.cuda()
        stage1_res, stage2_res = encdec(input_x)
        assert stage1_res.shape == (2, 3, 256, 256)
        assert stage2_res.shape == (2, 3, 256, 256)
        encdec = DeepFillEncoderDecoder(return_offset=True).cuda()
        stage1_res, stage2_res, offset = encdec(input_x)
        assert offset.shape == (2, 32, 32, 32, 32)
