# Copyright (c) OpenMMLab. All rights reserved.
# import warnings
from typing import List, Optional, Sequence

from mmeval.metrics import MultiScaleStructureSimilarity as MS_SSIM_MMEVAL

from mmedit.registry import METRICS
from .base_gen_metric import GenerativeMetric


@METRICS.register_module('MS_SSIM')
@METRICS.register_module()
class MultiScaleStructureSimilarity(MS_SSIM_MMEVAL, GenerativeMetric):
    """MS-SSIM (Multi-Scale Structure Similarity) metric.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py # noqa

    Args:
        fake_nums (int): Numbers of the generated image need for the metric.
        fake_key (Optional[str]): Key for get fake images of the output dict.
            Defaults to None.
        real_key (Optional[str]): Key for get real images from the input dict.
            Defaults to 'img'.
        need_cond_input (bool): If true, the sampler will return the
            conditional input randomly sampled from the original dataset.
            This require the dataset implement `get_data_info` and field
            `gt_label` must be contained in the return value of
            `get_data_info`. Noted that, for unconditional models, set
            `need_cond_input` as True may influence the result of evaluation
            results since the conditional inputs are sampled from the dataset
            distribution; otherwise will be sampled from the uniform
            distribution. Defaults to False.

        sample_model (str): Sampling mode for the generative model. Support
            'orig' and 'ema'. Defaults to 'ema'.
        collect_device (str, optional): Device name used for collecting results
            from different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    name = 'MS-SSIM'

    def __init__(self,
                 fake_nums: int,
                 fake_key: Optional[str] = None,
                 input_order: str = 'CHW',
                 max_val: int = 255,
                 filter_size: int = 11,
                 filter_sigma: float = 1.5,
                 k1: float = 0.01,
                 k2: float = 0.03,
                 weights: List[float] = [
                     0.0448, 0.2856, 0.3001, 0.2363, 0.1333
                 ],
                 need_cond_input: bool = False,
                 sample_model: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 dist_backend='torch_cuda',
                 **kwargs) -> None:

        MS_SSIM_MMEVAL.__init__(
            self,
            input_order,
            max_val,
            filter_size,
            filter_sigma,
            k1,
            k2,
            weights,
            dist_backend=dist_backend,
            **kwargs)
        GenerativeMetric.__init__(self, fake_nums, 0, fake_key, None,
                                  need_cond_input, sample_model,
                                  collect_device, prefix)

        assert fake_nums % 2 == 0
        self.num_pairs = fake_nums // 2

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Feed data to the metric.

        Args:
            data_batch (dict): Real images from dataloader. Do not be used
                in this metric.
            data_samples (Sequence[dict]): Generated images.
        """
        if len(self._results) >= (self.fake_nums_per_device // 2):
            return

        fake_imgs = []
        for pred in data_samples:
            fake_img_ = pred
            # get ema/orig results
            if self.sample_model in fake_img_:
                fake_img_ = fake_img_[self.sample_model]
            # get specific fake_keys
            if (self.fake_key is not None and self.fake_key in fake_img_):
                fake_img_ = fake_img_[self.fake_key]['data']
            else:
                # get img tensor
                fake_img_ = fake_img_['fake_img']['data']

            # NOTE: here we convert [-1, 1] to [0, 255] manually, in the future
            # we can remove this operation since we call destruct in
            # base_gan.forward_test
            fake_img_ = ((fake_img_ + 1) / 2).clamp_(0, 1) * 255.
            fake_imgs.append(fake_img_.cpu().numpy().astype('uint8'))

        self.add(fake_imgs)

    def evaluate(self, *args, **kwargs) -> dict():
        metric_results = self.compute(*args, **kwargs)
        self.reset()

        key_template = f'{self.prefix}/{{}}' if self.prefix else '{}'
        return {key_template.format(k): v for k, v in metric_results.items()}
