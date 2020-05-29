import torch
from mmedit.models.components import TMADPatchDiscriminator


def test_tmad_patch_disc():
    x = torch.rand((2, 3, 32, 32))
    disc = TMADPatchDiscriminator(
        3, 16, num_enc_blocks=2, num_dilation_blocks=2)

    res = disc(x)

    assert res.shape == (2, 1, 8, 8)
    assert not disc.output_conv.with_activation
    assert not disc.output_conv.with_norm
    assert not disc.output_conv.with_spectral_norm

    if torch.cuda.is_available():
        x = x.cuda()
        disc = TMADPatchDiscriminator(
            3, 16, num_enc_blocks=2, num_dilation_blocks=2).cuda()

        res = disc(x)

        assert res.shape == (2, 1, 8, 8)
        assert not disc.output_conv.with_activation
        assert not disc.output_conv.with_norm
        assert not disc.output_conv.with_spectral_norm
