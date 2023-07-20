# Evaluator

## Evaluation Metrics and Evaluators

In model validation and testing, it is usually necessary to quantitatively evaluate the accuracy of the model. In mmagic, the evaluation metrics and evaluators are implemented to accomplish this functionality.

- Evaluation metrics are used to calculate specific model accuracy indicators based on test data and model prediction results. mmagic provides a variety of built-in metrics, which can be found in the metrics documentation. Additionally, metrics are decoupled from datasets and can be used for multiple datasets.
- The evaluator is the top-level module for evaluation metrics and usually contains one or more metrics. The purpose of the evaluator is to perform necessary data format conversion and call evaluation metrics to calculate the model accuracy during model evaluation. The evaluator is typically built by a `Runner` or a testing script, which are used for online evaluation and offline evaluation, respectively.

The evaluator in MMagic inherits from that in MMEngine and has a similar basic usage. For specific information, you can refer to [Model Accuracy Evaluation](https://mmengine.readthedocs.io/en/latest/tutorials/evaluation.html). However, different from other high-level vision tasks, the evaluation metrics for generative models often have multiple inputs. For example, the input for the Inception Score (IS) metric is only fake images and any number of real images, while the Perceptual Path Length (PPL) requires sampling from the latent space. To accommodate different evaluation metrics, mmagic introduces two important methods, `prepare_metrics` and `prepare_samplers` to meet the above requirements.

## prepare_metrics

```python
class Evaluator(Evaluator):
	...
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
        if self.metrics is None:
            self.is_ready = True
            return

        if self.is_ready:
            return

        # prepare metrics
        for metric in self.metrics:
            metric.prepare(module, dataloader)
        self.is_ready = True
```

The `prepare_metrics` method needs to be called before the evaluation starts. It is used to preprocess before evaluating each metric, and will sequentially call the prepare method of each metric in the evaluator to prepare any pre-calculated elements needed for that metric (such as features from hidden layers). Additionally, to avoid repeated calls, the `evaluator.is_ready` flag will be set to True after preprocessing for all metrics is completed.

```python
class GenMetric(BaseMetric):
	...
    def prepare(self, module: nn.Module, dataloader: DataLoader) -> None:
        """Prepare for the pre-calculating items of the metric. Defaults to do
        nothing.

        Args:
            module (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for the real images.
        """
        if is_model_wrapper(module):
            module = module.module
        self.data_preprocessor = module.data_preprocessor
```

## prepare_samplers

Different metrics require different inputs for generative models. For example, FID, KID, and IS only need the generated fake images, while PPL requires vectors from the latent space. Therefore, mmagic groups different evaluation metrics based on the type of input. One or more evaluation metrics in the same group share a data sampler. The sampler mode for each evaluation metric is determined by the `SAMPLER_MODE` attribute of that metric.

```python
class GenMetric(BaseMetric):
	...
    SAMPLER_MODE = 'normal'

class GenerativeMetric(GenMetric):
	...
    SAMPLER_MODE = 'Generative'
```

The `prepare_samplers` method of the evaluator is responsible for preparing the data samplers based on the sampler mode of all evaluation metrics.

```python
class Evaluator(Evaluator):
	...
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
        if self.metrics is None:
            return [[[None], []]]

        # grouping metrics based on `SAMPLER_MODE` and `sample_mode`
        metric_mode_dict = defaultdict(list)
        for metric in self.metrics:  # Specify a sampler group for each metric.
            metric_md5 = self._cal_metric_hash(metric)
            metric_mode_dict[metric_md5].append(metric)

        metrics_sampler_list = []
        for metrics in metric_mode_dict.values(): # Generate a sampler for each group.
            first_metric = metrics[0]
            metrics_sampler_list.append([
                metrics,
                first_metric.get_metric_sampler(module, dataloader, metrics)
            ])

        return metrics_sampler_list
```

The method will first check if it has any evaluation metrics to calculate: if not, it will return directly. If there are metrics to calculate, it will iterate through all the evaluation metrics and group them based on the sampler_mode and sample_model. The specific implementation is as follows: it calculates a hash code based on the sampler_mode and sample_model, and puts the evaluation metrics with the same hash code into the same list.

```python
class Evaluator(Evaluator):
	...
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
```

Finally, this method will generate a sampler for each evaluation metric group and add it to a list to return.

## Evaluation process of an evaluator

The implementation of evaluation process can be found in `mmagic.engine.runner.MultiValLoop.run` and `mmagic.engine.runner.MultiTestLoop.run`. Here we take `mmagic.engine.runner.MultiValLoop.run` as example.

```python
class MultiValLoop(BaseLoop):
	...
    def run(self):
	...
        # 1. prepare all metrics and get the total length
        metrics_sampler_lists = []
        meta_info_list = []
        dataset_name_list = []
        for evaluator, dataloader in zip(self.evaluators, self.dataloaders):
            # 1.1 prepare for metrics
            evaluator.prepare_metrics(module, dataloader)
            # 1.2 prepare for metric-sampler pair
            metrics_sampler_list = evaluator.prepare_samplers(
                module, dataloader)
            metrics_sampler_lists.append(metrics_sampler_list)
            # 1.3 update total length
            self._total_length += sum([
                len(metrics_sampler[1])
                for metrics_sampler in metrics_sampler_list
            ])
            # 1.4 save metainfo and dataset's name
            meta_info_list.append(
                getattr(dataloader.dataset, 'metainfo', None))
            dataset_name_list.append(dataloader.dataset.__class__.__name__)
```

First, the runner will perform preprocessing and obtain the necessary data samplers for evaluation using the `evaluator.prepare_metric` and `evaluator.prepare_samplers` methods. It will also update the total length of samples obtained using the samplers. As the evaluation metrics and dataset in mmagic are separated, some meta_info required for evaluation also needs to be saved and passed to the evaluator.

```python
class MultiValLoop(BaseLoop):
	...
    def run(self):
	...
        # 2. run evaluation
        for idx in range(len(self.evaluators)):
            # 2.1 set self.evaluator for run_iter
            self.evaluator = self.evaluators[idx]
            self.dataloader = self.dataloaders[idx]

            # 2.2 update metainfo for evaluator and visualizer
            meta_info = meta_info_list[idx]
            dataset_name = dataset_name_list[idx]
            if meta_info:
                self.evaluator.dataset_meta = meta_info
                self._runner.visualizer.dataset_meta = meta_info
            else:
                warnings.warn(
                    f'Dataset {dataset_name} has no metainfo. `dataset_meta` '
                    'in evaluator, metric and visualizer will be None.')

            # 2.3 generate images
            metrics_sampler_list = metrics_sampler_lists[idx]
            for metrics, sampler in metrics_sampler_list:
                for data in sampler:
                    self.run_iter(idx_counter, data, metrics)
                    idx_counter += 1

            # 2.4 evaluate metrics and update multi_metric
            metrics = self.evaluator.evaluate()
            if multi_metric and metrics.keys() & multi_metric.keys():
                raise ValueError('Please set different prefix for different'
                                 ' datasets in `val_evaluator`')
            else:
                multi_metric.update(metrics)
        # 3. finish evaluation and call hooks
        self._runner.call_hook('after_val_epoch', metrics=multi_metric)
        self._runner.call_hook('after_val')
```

After the preparation for evaluation is completed, the runner will iterate through all the evaluators and perform the evaluation one by one. Each evaluator needs to correspond to a data loader to complete the evaluation work for a dataset. Specifically, during the evaluation process for each evaluator, it is necessary to pass the required meta_info to the evaluator, then iterate through all the metrics_samplers of this evaluator to generate the images needed for evaluation, and finally complete the evaluation.
