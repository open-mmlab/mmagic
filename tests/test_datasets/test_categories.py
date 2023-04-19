# Copyright (c) OpenMMLab. All rights reserved.
from mmagic.datasets.categories import CIFAR10_CATEGORIES, IMAGENET_CATEGORIES


def test_cifar10_categories():
    assert len(CIFAR10_CATEGORIES) == 10


def test_imagenet_categories():
    assert len(IMAGENET_CATEGORIES) == 1000
