import warnings

from .builder import DATASETS, PIPELINES

__all__ = ['DATASETS', 'PIPELINES']

warnings.simplefilter('once', DeprecationWarning)
warnings.warn(
    'Registries (DATASETS, PIPELINES) have been moved to '
    'mmedit.datasets.builder. Importing from '
    'mmedit.models.registry will be deprecated in the future.',
    DeprecationWarning)
