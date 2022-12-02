# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import print_log
from PIL import Image
from scipy.stats import entropy
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

from mmedit.registry import METRICS
# from .inception_utils import disable_gpu_fuser_on_pt19, load_inception
from ..functional import disable_gpu_fuser_on_pt19, load_inception
from .base_gen_metric import GenerativeMetric


@METRICS.register_module('IS')
@METRICS.register_module()
class InceptionScore(GenerativeMetric):
    """IS (Inception Score) metric. The images are split into groups, and the
    inception score is calculated on each group of images, then the mean and
    standard deviation of the score is reported. The calculation of the
    inception score on a group of images involves first using the inception v3
    model to calculate the conditional probability for each image (p(y|x)). The
    marginal probability is then calculated as the average of the conditional
    probabilities for the images in the group (p(y)). The KL divergence is then
    calculated for each image as the conditional probability multiplied by the
    log of the conditional probability minus the log of the marginal
    probability. The KL divergence is then summed over all images and averaged
    over all classes and the exponent of the result is calculated to give the
    final score.

    Ref: https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py  # noqa

    Note that we highly recommend that users should download the Inception V3
    script module from the following address. Then, the `inception_pkl` can
    be set with user's local path. If not given, we will use the Inception V3
    from pytorch model zoo. However, this may bring significant different in
    the final results.

    Tero's Inception V3: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt  # noqa

    Args:
        fake_nums (int): Numbers of the generated image need for the metric.
        resize (bool, optional): Whether resize image to 299x299. Defaults to
            True.
        splits (int, optional): The number of groups. Defaults to 10.
        inception_style (str): The target inception style want to load. If the
            given style cannot be loaded successful, will attempt to load a
            valid one. Defaults to 'StyleGAN'.
        inception_path (str, optional): Path the the pretrain Inception
            network. Defaults to None.
        resize_method (str): Resize method. If `resize` is False, this will be
            ignored. Defaults to 'bicubic'.
        use_pil_resize (bool): Whether use Bicubic interpolation with
            Pillow's backend. If set as True, the evaluation process may be a
            little bit slow, but achieve a more accurate IS result. Defaults
            to False.
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
            'orig' and 'ema'. Defaults to 'orig'.
        collect_device (str, optional): Device name used for collecting results
            from different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    name = 'IS'

    pil_resize_method_mapping = {
        'bicubic': Image.BICUBIC,
        'bilinear': Image.BILINEAR,
        'nearest': Image.NEAREST,
        'box': Image.BOX
    }

    def __init__(self,
                 fake_nums: int = 5e4,
                 resize: bool = True,
                 splits: int = 10,
                 inception_style: str = 'StyleGAN',
                 inception_path: Optional[str] = None,
                 resize_method='bicubic',
                 use_pillow_resize: bool = True,
                 fake_key: Optional[str] = None,
                 need_cond_input: bool = False,
                 sample_model='orig',
                 collect_device: str = 'cpu',
                 prefix: str = None):
        super().__init__(fake_nums, 0, fake_key, None, need_cond_input,
                         sample_model, collect_device, prefix)

        self.resize = resize
        self.splits = splits
        self.device = 'cpu'

        if not use_pillow_resize:
            print_log(
                'We strongly recommend to use the bicubic resize with '
                'Pillow backend. Otherwise, the results maybe '
                'unreliable', 'current')
        self.use_pillow_resize = use_pillow_resize

        if self.use_pillow_resize:
            allowed_resize_method = list(self.pil_resize_method_mapping.keys())
            assert resize_method in self.pil_resize_method_mapping, (
                f'\'resize_method\' (\'{resize_method}\') is not supported '
                'for PIL resize. Please select resize method in '
                f'{allowed_resize_method}.')
            self.resize_method = self.pil_resize_method_mapping[resize_method]
        else:
            self.resize_method = resize_method

        self.inception, self.inception_style = self._load_inception(
            inception_style, inception_path)

    def prepare(self, module: nn.Module, dataloader: DataLoader) -> None:
        """Prepare for the pre-calculating items of the metric. Defaults to do
        nothing.

        Args:
            module (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for the real images.
        """
        self.device = module.data_preprocessor.device
        self.inception.to(self.device)

    def _load_inception(self, inception_style: str,
                        inception_path: Optional[str]
                        ) -> Tuple[nn.Module, str]:
        """Load pretrain model of inception network.
        Args:
            inception_style (str): Target style of Inception network want to
                load.
            inception_path (Optional[str]): The path to the inception.

        Returns:
            Tuple[nn.Module, str]: The actually loaded inception network and
                corresponding style.
        """
        inception, style = load_inception(
            dict(type=inception_style, inception_path=inception_path), 'IS')
        inception.eval()
        return inception, style

    def _preprocess(self, image: Tensor) -> Tensor:
        """Preprocess image before pass to the Inception. Preprocess operations
        contain channel conversion and resize.

        Args:
            image (Tensor): Image tensor before preprocess.

        Returns:
            Tensor: Image tensor after resize and channel conversion
                (if need.)
        """
        # image must passed in 'bgr'
        image = image[:, [2, 1, 0]]
        if not self.resize:
            return image
        if self.use_pillow_resize:
            image = (image.clone() * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            x_np = [x_.permute(1, 2, 0).detach().cpu().numpy() for x_ in image]

            # use bicubic resize as default
            x_pil = [
                Image.fromarray(x_).resize((299, 299),
                                           resample=self.resize_method)
                for x_ in x_np
            ]
            x_ten = torch.cat(
                [torch.FloatTensor(np.array(x_)[None, ...]) for x_ in x_pil])
            x_ten = (x_ten / 127.5 - 1).to(torch.float)
            return x_ten.permute(0, 3, 1, 2)
        else:
            return F.interpolate(
                image, size=(299, 299), mode=self.resize_method)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.fake_results``, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        if len(self.fake_results) >= self.fake_nums_per_device:
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
            fake_imgs.append(fake_img_)
        fake_imgs = torch.stack(fake_imgs, dim=0)
        fake_imgs = self._preprocess(fake_imgs).to(self.device)

        if self.inception_style == 'StyleGAN':
            fake_imgs = (fake_imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            with disable_gpu_fuser_on_pt19():
                feat = self.inception(fake_imgs, no_output_bias=True)
        else:
            feat = F.softmax(self.inception(fake_imgs), dim=1)

        # NOTE: feat is shape like (bz, 1000), convert to a list
        self.fake_results += list(torch.split(feat, 1))

    def compute_metrics(self, fake_results: list) -> dict:
        """Compute the results of Inception Score metric.

        Args:
            fake_results (list): List of image feature of fake images.

        Returns:
            dict: A dict of the computed IS metric and its standard error
        """
        split_scores = []
        preds = torch.cat(fake_results, dim=0).cpu().numpy()
        # check for the size
        assert preds.shape[0] >= self.fake_nums
        preds = preds[:self.fake_nums]
        for k in range(self.splits):
            part = preds[k * (self.fake_nums // self.splits):(k + 1) *
                         (self.fake_nums // self.splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        mean, std = np.mean(split_scores), np.std(split_scores)

        return {'is': float(mean), 'is_std': float(std)}


@METRICS.register_module()
class TransIS(InceptionScore):
    """IS (Inception Score) metric. The images are split into groups, and the
    inception score is calculated on each group of images, then the mean and
    standard deviation of the score is reported. The calculation of the
    inception score on a group of images involves first using the inception v3
    model to calculate the conditional probability for each image (p(y|x)). The
    marginal probability is then calculated as the average of the conditional
    probabilities for the images in the group (p(y)). The KL divergence is then
    calculated for each image as the conditional probability multiplied by the
    log of the conditional probability minus the log of the marginal
    probability. The KL divergence is then summed over all images and averaged
    over all classes and the exponent of the result is calculated to give the
    final score.

    Ref: https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py  # noqa

    Note that we highly recommend that users should download the Inception V3
    script module from the following address. Then, the `inception_pkl` can
    be set with user's local path. If not given, we will use the Inception V3
    from pytorch model zoo. However, this may bring significant different in
    the final results.

    Tero's Inception V3: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt  # noqa

    Args:
        fake_nums (int): Numbers of the generated image need for the metric.
        resize (bool, optional): Whether resize image to 299x299. Defaults to
            True.
        splits (int, optional): The number of groups. Defaults to 10.
        inception_style (str): The target inception style want to load. If the
            given style cannot be loaded successful, will attempt to load a
            valid one. Defaults to 'StyleGAN'.
        inception_path (str, optional): Path the the pretrain Inception
            network. Defaults to None.
        resize_method (str): Resize method. If `resize` is False, this will be
            ignored. Defaults to 'bicubic'.
        use_pil_resize (bool): Whether use Bicubic interpolation with
            Pillow's backend. If set as True, the evaluation process may be a
            little bit slow, but achieve a more accurate IS result. Defaults
            to False.
        fake_key (Optional[str]): Key for get fake images of the output dict.
            Defaults to None.
        real_key (Optional[str]): Key for get real images from the input dict.
            Defaults to 'img'.
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

    def __init__(self,
                 fake_nums: int = 50000,
                 resize: bool = True,
                 splits: int = 10,
                 inception_style: str = 'StyleGAN',
                 inception_path: Optional[str] = None,
                 resize_method='bicubic',
                 use_pillow_resize: bool = True,
                 fake_key: Optional[str] = None,
                 sample_model='ema',
                 collect_device: str = 'cpu',
                 prefix: str = None):
        # NOTE: set `need_cond` as False since we direct return the original
        # dataloader as sampler
        super().__init__(fake_nums, resize, splits, inception_style,
                         inception_path, resize_method, use_pillow_resize,
                         fake_key, False, sample_model, collect_device, prefix)
        self.SAMPLER_MODE = 'normal'

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.fake_results``, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from the model.
        """
        if len(self.fake_results) >= self.fake_nums_per_device:
            return

        fake_imgs = []
        for pred in data_samples:
            fake_img_ = pred
            # get ema/orig results
            if self.sample_model in fake_img_:
                fake_img_ = fake_img_[self.sample_model]
            # get specific fake_keys
            if (self.fake_key is not None and self.fake_key in fake_img_):
                fake_img_ = fake_img_[self.fake_key]
            # get img tensor
            fake_img_ = fake_img_['data']
            fake_imgs.append(fake_img_)
        fake_imgs = torch.stack(fake_imgs, dim=0)
        fake_imgs = self._preprocess(fake_imgs).to(self.device)

        if self.inception_style == 'StyleGAN':
            fake_imgs = (fake_imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            with disable_gpu_fuser_on_pt19():
                feat = self.inception(fake_imgs, no_output_bias=True)
        else:
            feat = F.softmax(self.inception(fake_imgs), dim=1)

        # NOTE: feat is shape like (bz, 1000), convert to a list
        self.fake_results += list(torch.split(feat, 1))

    def get_metric_sampler(self, model: nn.Module, dataloader: DataLoader,
                           metrics: List['GenerativeMetric']) -> DataLoader:
        """Get sampler for normal metrics. Directly returns the dataloader.

        Args:
            model (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for real images.
            metrics (List['GenMetric']): Metrics with the same sample mode.

        Returns:
            DataLoader: Default sampler for normal metrics.
        """
        return dataloader
