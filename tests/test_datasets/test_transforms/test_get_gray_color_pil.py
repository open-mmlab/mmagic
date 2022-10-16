# Copyright (c) OpenMMLab. All rights reserved.
import cv2 as cv

from mmedit.datasets.transforms import GenGrayColorPil


def test_get_gray_color_pil():
    img = cv.imread("../../data/image/gt/baboon.png")
    test_class = GenGrayColorPil(
        stage='test', keys=['rgb_img', 'gray_img']
    )

    results = test_class.transform(dict(img=img))

    assert 'rgb_img' in results.keys() and 'gray_img' in results.keys()
    assert results['gray_img'].shape == img.shape