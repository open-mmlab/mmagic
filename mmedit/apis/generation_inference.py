import numpy as np
import torch
from mmcv.parallel import collate, scatter

from mmedit.core import tensor2img
from mmedit.datasets.pipelines import Compose


def generation_inference(model, img, img_unpaired=None):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): File path of input image.
        img_unpaired (str, optional): File path of the unpaired image.
            If not None, perform unpaired image generation. Default: None.

    Returns:
        np.ndarray: The predicted generation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)
    # prepare data
    if img_unpaired is None:
        data = dict(pair_path=img)
    else:
        data = dict(img_a_path=img, img_b_path=img_unpaired)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    with torch.no_grad():
        results = model(test_mode=True, **data)
    # process generation shown mode
    if img_unpaired is None:
        if model.show_input:
            output = np.concatenate([
                tensor2img(results['real_a'], min_max=(-1, 1)),
                tensor2img(results['fake_b'], min_max=(-1, 1)),
                tensor2img(results['real_b'], min_max=(-1, 1))
            ],
                                    axis=1)
        else:
            output = tensor2img(results['fake_b'], min_max=(-1, 1))
    else:
        if model.show_input:
            output = np.concatenate([
                tensor2img(results['real_a'], min_max=(-1, 1)),
                tensor2img(results['fake_b'], min_max=(-1, 1)),
                tensor2img(results['real_b'], min_max=(-1, 1)),
                tensor2img(results['fake_a'], min_max=(-1, 1))
            ],
                                    axis=1)
        else:
            if model.test_direction == 'a2b':
                output = tensor2img(results['fake_b'], min_max=(-1, 1))
            else:
                output = tensor2img(results['fake_a'], min_max=(-1, 1))
    return output
