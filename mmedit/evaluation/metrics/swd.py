# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmeval.metrics import SlicedWassersteinDistance as SWD_MMEVAL

from mmedit.registry import METRICS
from .base_gen_metric import GenMetric


@METRICS.register_module('SWD')
@METRICS.register_module()
class SlicedWassersteinDistance(SWD_MMEVAL, GenMetric):
    """SWD (Sliced Wasserstein distance) metric. We calculate the SWD of two
    sets of images in the following way. In every 'feed', we obtain the
    Laplacian pyramids of every images and extract patches from the Laplacian
    pyramids as descriptors. In 'summary', we normalize these descriptors along
    channel, and reshape them so that we can use these descriptors to represent
    the distribution of real/fake images. And we can calculate the sliced
    Wasserstein distance of the real and fake descriptors as the SWD of the
    real and fake images.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/sliced_wasserstein.py # noqa

    Args:
        fake_nums (int): Numbers of the generated image need for the metric.
        image_shape (tuple): Image shape in order "CHW".
        fake_key (Optional[str]): Key for get fake images of the output dict.
            Defaults to None.
        real_key (Optional[str]): Key for get real images from the input dict.
            Defaults to 'img'.
        sample_model (str): Sampling mode for the generative model. Support
            'orig' and 'ema'. Defaults to 'ema'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    name = 'SWD'

    def __init__(self,
                 fake_nums: int,
                 resolution: int,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 input_order: str = 'CHW',
                 sample_model: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 dist_backend: str = 'torch_cuda',
                 **kwargs):

        SWD_MMEVAL.__init__(
            self, resolution, input_order, dist_backend=dist_backend, **kwargs)
        GenMetric.__init__(self, fake_nums, fake_nums, fake_key, real_key,
                           sample_model, collect_device, prefix)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.fake_results`` and
        ``self.real_results``, which will be used to compute the metrics when
        all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        # calculate processed images maually
        if not self._results:
            _num_processed = 0
        else:
            assert len(self._results) == 2, ('results length should be 2')
            assert len(self._results[0]) == len(
                self._results[1]), ('feature of real and fake should be same.')
            _num_processed = len(self._results[0])

        if self.fake_nums != -1 and (_num_processed >=
                                     self.fake_nums_per_device):
            return

        real_imgs, fake_imgs = [], []
        for data in data_samples:
            # parse real images
            real_img_ = data['gt_img']['data']
            if real_img_.shape[1] == 1:
                real_img_ = real_img_.repeat(3, 1, 1)
            real_imgs.append(data['gt_img']['data'])
            # parse fake images
            fake_img_ = data
            # get ema/orig results
            if self.sample_model in fake_img_:
                fake_img_ = fake_img_[self.sample_model]
            # get specific fake_keys
            if (self.fake_key is not None and self.fake_key in fake_img_):
                fake_img_ = fake_img_[self.fake_key]['data']
            else:
                # get img tensor
                fake_img_ = fake_img_['fake_img']['data']
            if fake_img_.shape[1] == 1:
                fake_img_ = fake_img_.repeat(3, 1, 1)
            fake_imgs.append(fake_img_)

        # safety checking for input shape
        assert all([img.shape[-1] == self.resolution for img in real_imgs])
        assert all([img.shape[-1] == self.resolution for img in fake_imgs])

        # convert to numpy + [0, 255], in future, we can remove the following
        # code since SWD will support torch backend and we will call destruct
        # in BaseGAN.forward
        real_imgs = [((img + 1) / 2 * 255.).cpu().numpy() for img in real_imgs]
        fake_imgs = [((img + 1) / 2 * 255.).cpu().numpy() for img in fake_imgs]

        self.add(fake_imgs, real_imgs)

    def evaluate(self, *args, **kwargs) -> dict:
        metric_results = self.compute(*args, **kwargs)
        self.reset()

        key_template = f'{self.prefix}/{{}}' if self.prefix else '{}'
        return {key_template.format(k): v for k, v in metric_results.items()}
