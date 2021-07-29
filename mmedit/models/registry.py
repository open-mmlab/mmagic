import warnings

from .builder import BACKBONES, COMPONENTS, LOSSES, MODELS

__all__ = ['BACKBONES', 'COMPONENTS', 'LOSSES', 'MODELS']

warnings.simplefilter('once', DeprecationWarning)
warnings.warn(
    'Registries (BACKBONES, NECKS, HEADS, LOSSES, POSENETS) have '
    'been moved to mmedit.models.builder. Importing from '
    'mmedit.models.registry will be deprecated in the future.',
    DeprecationWarning)
