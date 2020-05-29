from .deepfill_disc import DeepFillv1Discriminators
from .gl_disc import GLDiscs
from .modified_vgg import ModifiedVGG
from .multi_layer_disc import MultiLayerDiscriminator
from .tmad_patch_disc import TMADPatchDiscriminator

__all__ = [
    'GLDiscs', 'ModifiedVGG', 'MultiLayerDiscriminator',
    'DeepFillv1Discriminators', 'TMADPatchDiscriminator'
]
