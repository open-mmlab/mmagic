import pytest


def test_old_fashion_registry_importing():
    with pytest.warns(DeprecationWarning):
        from mmedit.models.registry import (  # noqa: F401
            BACKBONES, COMPONENTS, LOSSES, MODELS)
    with pytest.warns(DeprecationWarning):
        from mmedit.datasets.registry import DATASETS, PIPELINES  # noqa: F401
