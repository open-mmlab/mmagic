# Copyright (c) OpenMMLab. All rights reserved.
from mmedit.edit import MMEdit
from mmedit.utils import register_all_modules

register_all_modules()


def test_edit():
    mmedit_instance = MMEdit('biggan', model_ckpt='')
    inference_result = mmedit_instance.infer(label=1)
    result_img = inference_result[1]
    assert result_img.shape == (4, 3, 32, 32)


if __name__ == '__main__':
    test_edit()
