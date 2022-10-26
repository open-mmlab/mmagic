# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.dataset import Compose
from mmengine.dataset.utils import default_collate as collate

from mmedit.models.base_models import BaseTranslationModel


def sample_img2img_model(model, image_path, target_domain=None, **kwargs):
    """Sampling from translation models.

    Args:
        model (nn.Module): The loaded model.
        image_path (str): File path of input image.
        style (str): Target style of output image.
    Returns:
        Tensor: Translated image tensor.
    """
    assert isinstance(model, BaseTranslationModel)

    # get source domain and target domain
    if target_domain is None:
        target_domain = model._default_domain
    source_domain = model.get_other_domains(target_domain)[0]

    cfg = model.cfg
    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)

    # prepare data
    data = dict()
    # dirty code to deal with test data pipeline
    data['pair_path'] = image_path
    data[f'img_{source_domain}_path'] = image_path
    data[f'img_{target_domain}_path'] = image_path

    data = collate([test_pipeline(data)])
    data = model.data_preprocessor(data, False)
    inputs_dict = data['inputs']

    source_image = inputs_dict[f'img_{source_domain}']

    # forward the model
    with torch.no_grad():
        results = model(
            source_image,
            test_mode=True,
            target_domain=target_domain,
            **kwargs)
    output = results['target']
    return output
