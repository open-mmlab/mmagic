# 评估器

## 评测指标与评测器

在模型的验证和测试中，通常需要对模型的精度进行定量的评测。在mmagic中实现了评测指标(metric)和评测器(evaluator)来完成这一功能。

- 评测指标(metric)用于根据测试数据和模型预测结果，特定模型精度指标的计算。在mmagic中内置了多种metric，详见[评价指标](https://mmagic.readthedocs.io/zh_CN/latest/user_guides/metrics.html)。同时metric和数据集解耦，每种metric可以用于多个数据集。
- 评测器(evaluator)是评测指标的上层模块，通常需要包含一个或者多个指标。评测器的作用是在模型评测时完成必要的数据格式转换，并调用评测指标来计算模型精度。评测器通常由[执行器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html)或测试脚本构建，分别用于在线评测和离线评测。

mmagic中的评测器继承自mmengine中的评测器，基本使用方法也与mmengine中的评测器类似，具体可以参见[模型精度评测](https://mmengine.readthedocs.io/zh_CN/latest/design/evaluation.html)。但不同于其他上层视觉任务，生成模型的评估指标往往具有多种输入。例如Inception Score（IS）指标的输入仅为虚假图片和任意数量的真实图片；Perceptual path length（PPL) 则需要从隐空间中进行采样。为了对不同的评测指标进行兼容，mmagic设计了两个重要的方法prepare_metrics和prepare_samplers来实现上述要求。

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

prepare_metrics方法需要在评测开始之前调用。它被用于在每个评测指标开始评测之前进行预处理，会依次调用evaluator的所有评测指标的prepare方法来准备该评测指标的需要预先计算好的元素(例如一些隐藏层的特征)。同时为了避免多次重复调用，在所有评测指标预处理完成之后，evaluator.is_ready 标志位会被设置为True。

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

对于生成模型而言，不同的metric需要不同的输入。例如FID, KID, IS只需要生成的fake images，而PPL则需要隐空间的向量。因此mmagic将不同的评估指标按照输入的类型进行了分组，属于同一个组的一个或者多个评测指标共享一个数据的采样器，每个评测指标的sampler mode由该评测指标的SAMPLER_MODE属性决定。

```python
class GenMetric(BaseMetric):
	...
    SAMPLER_MODE = 'normal'

class GenerativeMetric(GenMetric):
	...
    SAMPLER_MODE = 'Generative'
```

而evaluator的prepare_samplers 方法就是根据所有评测指标的sampler mode来准备好data sampler。

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
        for metric in self.metrics:  # 为每个metric指定sampler group
            metric_md5 = self._cal_metric_hash(metric)
            metric_mode_dict[metric_md5].append(metric)

        metrics_sampler_list = []
        for metrics in metric_mode_dict.values(): #为每个group生成sampler
            first_metric = metrics[0]
            metrics_sampler_list.append([
                metrics,
                first_metric.get_metric_sampler(module, dataloader, metrics)
            ])

        return metrics_sampler_list
```

该方法会首先检查自身是否有需要计算的评测指标：如果没有直接返回，如果有则会遍历所有评测指标，对所有采样指标根据sampler_mode和sample_model进行分组, 具体实现方式为根据sampler_mode和sample_model计算hash码，将具有相同hash码的评测指标放入同一个列表里。

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

最后该方法会为每一个评测指标组生成一个sampler采样器，添加到列表返回。

## 评测器评测流程

整个评测器的评测流程在方法mmagic.engine.runner.MultiValLoop.run和mmagic.engine.runner.MultiTestLoop.run中实现。以mmagic.engine.runner.MultiTestLoop.run为例：

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

runner首先会通过evaluator.prepare_metrics和evaluator.prepare_samplers两个方法来进行评测所需要的预处理工作和获取评测所需要的数据采样器；同时更新所有采样器的采样总长度。由于mmagic的评测指标和数据集进行了分离，因此一些在评测时所需要的meta_info也需要进行保存并传递给评测器。

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

在完成了评测前的准备之后，runner会遍历所有evaluator，依次进行评估，每个evaluator需要对应一个dataloader，完成一个数据集的评测工作。具体在对每个evaluator进行评测的过程中，首先需要将评测所需要的meta_info传递给评测器，随后遍历该evaluator的所有metrics_sampler，生成评测所需要的图像，最后再完成评测。
