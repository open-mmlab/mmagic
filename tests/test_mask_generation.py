import numpy as np
import pytest

from mmedit.core.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask,
                              random_bbox)


def test_bbox_mask():
    # default config for random bbox mask
    cfg = dict(
        img_shape=(256, 256),
        max_bbox_shape=100,
        max_bbox_delta=10,
        min_margin=10)

    bbox = random_bbox(**cfg)
    mask_bbox = bbox2mask(cfg['img_shape'], bbox)
    assert mask_bbox.shape == (256, 256, 1)
    zero_area = np.sum((mask_bbox == 0).astype(np.uint8))
    ones_area = np.sum((mask_bbox == 1).astype(np.uint8))
    assert zero_area + ones_area == 256 * 256
    assert mask_bbox.dtype == np.uint8

    with pytest.raises(ValueError):
        cfg_ = cfg.copy()
        cfg_['max_bbox_shape'] = 300
        bbox = random_bbox(**cfg_)

    with pytest.raises(ValueError):
        cfg_ = cfg.copy()
        cfg_['max_bbox_delta'] = 300
        bbox = random_bbox(**cfg_)

    with pytest.raises(ValueError):
        cfg_ = cfg.copy()
        cfg_['max_bbox_shape'] = 254
        bbox = random_bbox(**cfg_)

    cfg_ = cfg.copy()
    cfg_['max_bbox_delta'] = 1
    bbox = random_bbox(**cfg_)
    mask_bbox = bbox2mask(cfg['img_shape'], bbox)
    assert mask_bbox.shape == (256, 256, 1)


def test_free_form_mask():
    img_shape = (256, 256, 3)
    for _ in range(10):
        mask = brush_stroke_mask(img_shape)
        assert mask.shape == (256, 256, 1)

    img_shape = (256, 256, 3)
    mask = brush_stroke_mask(img_shape, num_vertexes=8)
    assert mask.shape == (256, 256, 1)
    zero_area = np.sum((mask == 0).astype(np.uint8))
    ones_area = np.sum((mask == 1).astype(np.uint8))
    assert zero_area + ones_area == 256 * 256
    assert mask.dtype == np.uint8

    img_shape = (256, 256, 3)
    mask = brush_stroke_mask(img_shape, brush_width=10)
    assert mask.shape == (256, 256, 1)

    with pytest.raises(TypeError):
        mask = brush_stroke_mask(img_shape, num_vertexes=dict())

    with pytest.raises(TypeError):
        mask = brush_stroke_mask(img_shape, brush_width=dict())


def test_irregular_mask():
    img_shape = (256, 256)
    for _ in range(10):
        mask = get_irregular_mask(img_shape)
        assert mask.shape == (256, 256, 1)
        assert 0.15 < (np.sum(mask) / (img_shape[0] * img_shape[1])) < 0.50
        zero_area = np.sum((mask == 0).astype(np.uint8))
        ones_area = np.sum((mask == 1).astype(np.uint8))
        assert zero_area + ones_area == 256 * 256
        assert mask.dtype == np.uint8

    with pytest.raises(TypeError):
        mask = get_irregular_mask(img_shape, brush_width=dict())

    with pytest.raises(TypeError):
        mask = get_irregular_mask(img_shape, length_range=dict())

    with pytest.raises(TypeError):
        mask = get_irregular_mask(img_shape, num_vertexes=dict())

    mask = get_irregular_mask(img_shape, brush_width=10)
    assert mask.shape == (256, 256, 1)

    mask = get_irregular_mask(img_shape, length_range=10)
    assert mask.shape == (256, 256, 1)

    mask = get_irregular_mask(img_shape, num_vertexes=10)
    assert mask.shape == (256, 256, 1)
