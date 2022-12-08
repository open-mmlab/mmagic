# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
from collections import defaultdict
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union

from mmengine.evaluator import BaseMetric, Evaluator
from mmengine.model import BaseModel
from torch.utils.data.dataloader import DataLoader

from mmedit.registry import EVALUATORS
from mmedit.structures import EditDataSample
from .metrics.base_gen_metric import GenMetric


@EVALUATORS.register_module()
class GenEvaluator(Evaluator):
    """Evaluator for generative models. Unlike high-level vision tasks, metrics
    for generative models have various input types. For example, Inception
    Score (IS, :class:`~mmedit.core.evaluation.InceptionScore`) only needs to
    take fake images as input. However, Frechet Inception Distance (FID,
    :class:`~mmedit.engine.evaluation.FrechetInceptionDistance`) needs to take
    both real images and fake images as input, and the numbers of real images
    and fake images can be set arbitrarily. For Perceptual path length (PPL,
    :class:`~mmedit.engine.evaluation.PerceptualPathLength.`), generator need
    to sample images along a latent path.

    In order to be compatible with different metrics, we designed two critical
    functions, :meth:`prepare_metrics` and :meth:`prepare_samplers` to support
    those requirements.

    - :meth:`prepare_metrics` set the image images' color order
      and pass the dataloader to all metrics. Therefore metrics need
      pre-processing to prepare the corresponding feature.
    - :meth:`premare_samplers` pass the dataloader and model to the metrics,
      and get the corresponding sampler of each kind of metrics. Metrics with
      same sample mode can share the sampler.

    The whole evaluation process can be found in
    :meth:~`mmedit.engine.runners.loops.GenValLoop.run` and
    :meth:~`mmedit.engine.runners.loops.GenTestLoop.run`.

    Args:
        metrics (dict or BaseMetric or Sequence): The config of metrics.
    """

    def __init__(self, metrics: Union[dict, BaseMetric, Sequence]):
        super().__init__(metrics)
        self.is_ready = False

    def prepare_metrics(self, module: BaseModel, dataloader: DataLoader):
        """Prepare for metrics before evaluation starts. Some metrics use
        pretrained model to extract feature. Some metrics use pretrained model
        to extract feature and input channel order may vary among those models.
        Therefore, we first parse the output color order from data
        preprocessor and set the color order for each metric. Then we pass the
        dataloader to each metrics to prepare pre-calculated items. (e.g.
        inception feature of the real images). If metric has no pre-calculated
        items, :meth:`metric.prepare` will be ignored. Once the function has
        been called, :attr:`self.is_ready` will be set as `True`. If
        :attr:`self.is_ready` is `True`, this function will directly return to
        avoid duplicate computation.

        Args:
            module (BaseModel): Model to evaluate.
            dataloader (DataLoader): The dataloader for real images.
        """
        if self.is_ready:
            return

        # prepare metrics
        for metric in self.metrics:
            metric.prepare(module, dataloader)
        self.is_ready = True

    @staticmethod
    def _cal_metric_hash(metric: GenMetric):
        """Calculate a unique hash value based on the `SAMPLER_MODE` and
        `sample_model`."""
        sampler_mode = metric.SAMPLER_MODE
        sample_model = metric.sample_model
        metric_dict = {
            'SAMPLER_MODE': sampler_mode,
            'sample_model': sample_model
        }
        if hasattr(metric, 'need_cond_input'):
            metric_dict['need_cond_input'] = metric.need_cond_input
        md5 = hashlib.md5(repr(metric_dict).encode('utf-8')).hexdigest()
        return md5

    def prepare_samplers(self, module: BaseModel, dataloader: DataLoader
                         ) -> List[Tuple[List[BaseMetric], Iterator]]:
        """Prepare for the sampler for metrics whose sampling mode are
        different. For generative models, different metric need image
        generated with different inputs. For example, FID, KID and IS need
        images generated with random noise, and PPL need paired images on the
        specific noise interpolation path. Therefore, we first group metrics
        with respect to their sampler's mode (refers to
        :attr:~`GenMetrics.SAMPLER_MODE`), and build a shared sampler for each
        metric group. To be noted that, the length of the shared sampler
        depends on the metric of the most images required in each group.

        Args:
            module (BaseModel): Model to evaluate. Some metrics (e.g. PPL)
                require `module` in their sampler.
            dataloader (DataLoader): The dataloader for real image.

        Returns:
            List[Tuple[List[BaseMetric], Iterator]]: A list of "metrics-shared
                sampler" pair.
        """
        # grouping metrics based on `SAMPLER_MODE` and `sample_mode`
        metric_mode_dict = defaultdict(list)
        for metric in self.metrics:
            metric_md5 = self._cal_metric_hash(metric)
            metric_mode_dict[metric_md5].append(metric)

        metrics_sampler_list = []
        for metrics in metric_mode_dict.values():
            first_metric = metrics[0]
            metrics_sampler_list.append([
                metrics,
                first_metric.get_metric_sampler(module, dataloader, metrics)
            ])

        return metrics_sampler_list

    def process(self, data_samples: Sequence[EditDataSample],
                data_batch: Optional[Any],
                metrics: Sequence[BaseMetric]) -> None:
        """Pass `data_batch` from dataloader and `predictions` (generated
        results) to corresponding `metrics`.

        Args:
            data_samples (Sequence[EditDataSample]): A batch of generated
                results from model.
            data_batch (Optional[Any]): A batch of data from the
                metrics specific sampler or the dataloader.
            metrics (Optional[Sequence[BaseMetric]]): Metrics to evaluate.
        """

        _data_samples = []
        for data_sample in data_samples:
            if isinstance(data_sample, EditDataSample):
                _data_samples.append(data_sample.to_dict())
            else:
                _data_samples.append(data_sample)

        # feed to the specifics metrics
        for metric in metrics:
            metric.process(data_batch, _data_samples)

    def evaluate(self) -> dict:
        """Invoke ``evaluate`` method of each metric and collect the metrics
        dictionary. Different from `Evaluator.evaluate`, this function does not
        take `size` as input, and elements in `self.metrics` will call their
        own `evaluate` method to calculate the metric.

        Returns:
            dict: Evaluation results of all metrics. The keys are the names
                of the metrics, and the values are corresponding results.
        """
        metrics = {}
        for metric in self.metrics:
            _results = metric.evaluate()

            # Check metric name conflicts
            for name in _results.keys():
                if name in metrics:
                    raise ValueError(
                        'There are multiple evaluation results with the same '
                        f'metric name {name}. Please make sure all metrics '
                        'have different prefixes.')

            metrics.update(_results)
        return metrics
