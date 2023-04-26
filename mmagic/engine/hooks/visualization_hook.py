# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from mmengine import MessageHub
from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner
from mmengine.structures import BaseDataElement
from mmengine.utils import is_list_of
from mmengine.visualization import Visualizer

from mmagic.structures import DataSample
from mmagic.utils import get_sampler


@HOOKS.register_module()
class BasicVisualizationHook(Hook):
    """Basic hook that invoke visualizers during validation and test.

    Args:
        interval (int | dict): Visualization interval. Default: {}.
        on_train (bool): Whether to call hook during train. Default to False.
        on_val (bool): Whether to call hook during validation. Default to True.
        on_test (bool): Whether to call hook during test. Default to True.
    """
    priority = 'NORMAL'

    def __init__(self,
                 interval: dict = {},
                 on_train=False,
                 on_val=True,
                 on_test=True):
        self._interval = interval
        self._sample_counter = 0
        self._vis_dir = None
        self._on_train = on_train
        self._on_val = on_val
        self._on_test = on_test

    def _after_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: Optional[Sequence[dict]],
        outputs: Optional[Sequence[BaseDataElement]],
        mode=None,
    ) -> None:
        """Show or Write the predicted results.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (Sequence[dict], optional): Data
                from dataloader. Defaults to None.
            outputs (Sequence[BaseDataElement], optional): Outputs from model.
                Defaults to None.
        """
        if mode == 'train' and (not self._on_train):
            return
        elif mode == 'val' and (not self._on_val):
            return
        elif mode == 'test' and (not self._on_test):
            return

        if isinstance(self._interval, int):
            interval = self._interval
        else:
            interval = self._interval.get(mode, 1)

        if self.every_n_inner_iters(batch_idx, interval):
            for data_sample in outputs:
                runner.visualizer.add_datasample(data_sample, step=runner.iter)


@HOOKS.register_module()
class VisualizationHook(Hook):
    """MMagic Visualization Hook. Used to visual output samples in training,
    validation and testing. In this hook, we use a list called
    `sample_kwargs_list` to control how to generate samples and how to
    visualize them. Each element in `sample_kwargs_list`, called
    `sample_kwargs`, may contains the following keywords:

    - Required key words:
        - 'type': Value must be string. Denotes what kind of sampler is used to
            generate image. Refers to :meth:`~mmagic.utils.get_sampler`.
    - Optional key words (If not passed, will use the default value):
        - 'n_row': Value must be int. The number of images in one row.
        - 'num_samples': Value must be int. The number of samples to visualize.
        - 'vis_mode': Value must be string. How to visualize the generated
            samples (e.g. image, gif).
        - 'fixed_input': Value must be bool. Whether use the fixed input
            during the loop.
        - 'draw_gt': Value must be bool. Whether save the real images.
        - 'target_keys': Value must be string or list of string. The keys of
            the target image to visualize.
        - 'name': Value must be string. If not passed, will use
            `sample_kwargs['type']` as default.

    For convenience, we also define a group of alias of samplers' type for
    models supported in MMagic. Refers to
    `:attr:self.SAMPLER_TYPE_MAPPING`.

    Example:
        >>> # for GAN models
        >>> custom_hooks = [
        >>>     dict(
        >>>         type='VisualizationHook',
        >>>         interval=1000,
        >>>         fixed_input=True,
        >>>         vis_kwargs_list=dict(type='GAN', name='fake_img'))]
        >>> # for Translation models
        >>> custom_hooks = [
        >>>     dict(
        >>>         type='VisualizationHook',
        >>>         interval=10,
        >>>         fixed_input=False,
        >>>         vis_kwargs_list=[dict(type='Translation',
        >>>                                  name='translation_train',
        >>>                                  n_samples=6, draw_gt=True,
        >>>                                  n_row=3),
        >>>                             dict(type='TranslationVal',
        >>>                                  name='translation_val',
        >>>                                  n_samples=16, draw_gt=True,
        >>>                                  n_row=4)])]

    # NOTE: user-defined vis_kwargs > vis_kwargs_mapping > hook init args

    Args:
        interval (int): Visualization interval. Default: 1000.
        sampler_kwargs_list (Tuple[List[dict], dict]): The list of sampling
            behavior to generate images.
        fixed_input (bool): The default action of whether use fixed input to
            generate samples during the loop. Defaults to True.
        n_samples (Optional[int]): The default value of number of samples to
            visualize. Defaults to 64.
        n_row (Optional[int]): The default value of number of images in each
            row in the visualization results. Defaults to None.
        message_hub_vis_kwargs (Optional[Tuple[str, dict, List[str],
            List[Dict]]]): Key arguments visualize images in message hub.
            Defaults to None.
        save_at_test (bool): Whether save images during test. Defaults to True.
        max_save_at_test (int): Maximum number of samples saved at test time.
            If None is passed, all samples will be saved. Defaults to 100.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
    """

    priority = 'NORMAL'

    VIS_KWARGS_MAPPING = dict(
        GAN=dict(type='Noise'),
        SinGAN=dict(type='Arguments', forward_kwargs=dict(mode='rand')),
        Translation=dict(type='Data'),
        TranslationVal=dict(type='ValData'),
        TranslationTest=dict(type='TestData'),
        DDPMDenoising=dict(
            type='Arguments',
            name='ddpm_sample',
            n_samples=16,
            n_row=4,
            vis_mode='gif',
            n_skip=100,
            forward_kwargs=dict(
                forward_mode='sampling',
                sample_kwargs=dict(show_pbar=True, save_intermedia=True))))

    def __init__(self,
                 interval: int = 1000,
                 vis_kwargs_list: Tuple[List[dict], dict] = None,
                 fixed_input: bool = True,
                 n_samples: Optional[int] = 64,
                 n_row: Optional[int] = None,
                 message_hub_vis_kwargs: Optional[Tuple[str, dict, List[str],
                                                        List[Dict]]] = None,
                 save_at_test: bool = True,
                 max_save_at_test: int = 100,
                 test_vis_keys: Optional[Union[str, List[str]]] = None,
                 show: bool = False,
                 wait_time: float = 0):

        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval

        self.vis_kwargs_list = deepcopy(vis_kwargs_list)
        if isinstance(self.vis_kwargs_list, dict):
            self.vis_kwargs_list = [self.vis_kwargs_list]

        self.fixed_input = fixed_input
        self.inputs_buffer = defaultdict(list)

        self.n_samples = n_samples
        self.n_row = n_row

        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.save_at_test = save_at_test
        self.test_vis_keys_list = test_vis_keys
        self.max_save_at_test = max_save_at_test
        self.message_vis_kwargs = message_hub_vis_kwargs

    @master_only
    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs) -> None:
        """:class:`VisualizationHook` do not support visualize during
        validation.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs: outputs of the generation model
        """
        return

    @master_only
    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs):
        """Visualize samples after test iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict, optional): Data from dataloader.
                Defaults to None.
            outputs: outputs of the generation model Defaults to None.
        """
        if not self.save_at_test:
            return

        for idx, sample in enumerate(outputs):
            curr_idx = batch_idx * len(outputs) + idx
            if (self.max_save_at_test is not None
                    and curr_idx >= self.max_save_at_test):
                continue
            # NOTE: only support visualize image tensors (ndim == 3)
            if self.test_vis_keys_list is None:
                target_keys = [
                    k for k, v in sample.items() if not k.startswith('_')
                    and isinstance(v, torch.Tensor) and v.ndim == 3
                ]
                assert len(target_keys), (
                    'Cannot found Tensor in outputs. Please specific '
                    '\'vis_test_keys_list\'.')
            elif isinstance(self.test_vis_keys_list, str):
                target_keys = [self.test_vis_keys_list]
            else:
                assert is_list_of(self.test_vis_keys_list, str), (
                    'test_vis_keys_list must be str or list of str or None.')
                target_keys = self.test_vis_keys_list

            for key in target_keys:
                name = key.replace('.', '_')
                self._visualizer.add_datasample(
                    name=f'test_{name}',
                    gen_samples=[sample],
                    step=curr_idx,
                    target_keys=key,
                    n_row=1)

    @master_only
    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: dict = None,
                         outputs: Optional[dict] = None) -> None:
        """Visualize samples after train iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model. Defaults to None.
        """
        if self.every_n_inner_iters(batch_idx, self.interval):
            self.vis_sample(runner, batch_idx, data_batch, outputs)

    @torch.no_grad()
    def vis_sample(self,
                   runner: Runner,
                   batch_idx: int,
                   data_batch: dict,
                   outputs: Optional[dict] = None) -> None:
        """Visualize samples.

        Args:
            runner (Runner): The runner contains model to visualize.
            batch_idx (int): The index of the current batch in loop.
            data_batch (dict): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model. Defaults to None.
        """
        # this function will only called in training process
        num_batches = runner.train_dataloader.batch_size

        module = runner.model
        module.eval()
        if hasattr(module, 'module'):
            module = module.module

        forward_func = module.val_step

        for vis_kwargs in self.vis_kwargs_list:
            # pop the sample-unrelated values
            vis_kwargs_ = deepcopy(vis_kwargs)
            sampler_type = vis_kwargs_['type']

            # replace with alias
            for alias in self.VIS_KWARGS_MAPPING.keys():
                if alias.upper() == sampler_type.upper():
                    sampler_alias = deepcopy(self.VIS_KWARGS_MAPPING[alias])
                    vis_kwargs_['type'] = sampler_alias.pop('type')
                    for default_k, default_v in sampler_alias.items():
                        vis_kwargs_.setdefault(default_k, default_v)
                    break
            # sampler_type = vis_kwargs_.pop('type')

            name = vis_kwargs_.pop('name', None)
            if not name:
                name = sampler_type.lower()

            n_samples = vis_kwargs_.pop('n_samples', self.n_samples)
            n_row = vis_kwargs_.pop('n_row', self.n_row)

            num_iters = math.ceil(n_samples / num_batches)
            vis_kwargs_['max_times'] = num_iters
            vis_kwargs_['num_batches'] = num_batches
            fixed_input = vis_kwargs_.pop('fixed_input', self.fixed_input)
            target_keys = vis_kwargs_.pop('target_keys', None)
            vis_mode = vis_kwargs_.pop('vis_mode', None)

            output_list = []
            if fixed_input and self.inputs_buffer[sampler_type]:
                sampler = self.inputs_buffer[sampler_type]
            else:
                sampler = get_sampler(vis_kwargs_, runner)
            need_save = fixed_input and not self.inputs_buffer[sampler_type]

            for inputs in sampler:
                output = forward_func(inputs)
                if len(output) != num_batches:
                    # one sample contains multiple elements
                    output_list.append(output)
                    contain_mul_elements = True
                else:
                    output_list += [out for out in forward_func(inputs)]
                    contain_mul_elements = False

                # save inputs
                if need_save:
                    self.inputs_buffer[sampler_type].append(inputs)

            output_list = output_list[:n_samples]
            if contain_mul_elements:
                output_to_vis = []
                for output in output_list:
                    output_to_vis += output
            else:
                output_to_vis = output_list
            n_row = min(n_row, len(output_to_vis)) if n_row else None

            self._visualizer.add_datasample(
                name=name,
                gen_samples=output_to_vis,
                target_keys=target_keys,
                vis_mode=vis_mode,
                n_row=n_row,
                show=self.show,
                wait_time=self.wait_time,
                step=batch_idx + 1,
                **vis_kwargs_)

        # save images in message_hub
        self.vis_from_message_hub(batch_idx)

        module.train()

    def vis_from_message_hub(self, batch_idx: int):
        """Visualize samples from message hub.

        Args:
            batch_idx (int): The index of the current batch in the test loop.
            color_order (str): The color order of generated images.
            target_mean (Sequence[Union[float, int]]): The original mean
                of the image tensor before preprocessing. Image will be
                re-shifted to ``target_mean`` before visualizing.
            target_std (Sequence[Union[float, int]]): The original std of the
                image tensor before preprocessing. Image will be re-scaled to
                ``target_std`` before visualizing.
        """
        # TODO: add destruct in this function
        if self.message_vis_kwargs is None:
            return

        message_hub = MessageHub.get_current_instance()
        if 'vis_results' not in message_hub.runtime_info:
            raise RuntimeError('Cannot find \'vis_results\' in '
                               '\'message_hub.runtime_info\'. Cannot perform '
                               'visualization from messageHub.')

        vis_results = message_hub.get_info('vis_results')
        if isinstance(self.message_vis_kwargs, str):
            target_keys, vis_modes = [self.message_vis_kwargs], [None]
        elif isinstance(self.message_vis_kwargs, dict):
            target_keys = [self.message_vis_kwargs['key']]
            vis_modes = [self.message_vis_kwargs['vis_mode']]
        elif is_list_of(self.message_vis_kwargs, str):
            target_keys = self.message_vis_kwargs
            vis_modes = [None for _ in range(len(target_keys))]
        else:
            # list of dict
            target_keys = [kwargs['key'] for kwargs in self.message_vis_kwargs]
            vis_modes = [
                kwargs.pop('vis_mode', None)
                for kwargs in deepcopy(self.message_vis_kwargs)
            ]

        for key, vis_mode in zip(target_keys, vis_modes):
            if key not in vis_results:
                raise RuntimeError(
                    f'Cannot find \'{key}\' in '
                    'message_hub.runtime_info[\'vis_results\'].')

            value = vis_results[key]
            # pack to list of DataSample
            if isinstance(value, torch.Tensor):
                gen_samples = []
                num_batches = value.shape[0]
                for idx in range(num_batches):
                    gen_sample = DataSample()
                    setattr(gen_sample, key, value[idx])
                    gen_samples.append(gen_sample)
            elif is_list_of(value, BaseDataElement):
                # already packed
                gen_samples = value
                num_batches = len(gen_samples)
            else:
                raise TypeError(
                    'Only support to visualize Tensor or list of DataSample '
                    f'in MessageHub. But \'{key}\' is \'{type(value)}\'.')

            self._visualizer.add_datasample(
                name=f'train_{key}',
                gen_samples=gen_samples,
                target_keys=key,
                vis_mode=vis_mode,
                n_row=min(self.n_row, num_batches) if self.n_row else None,
                show=self.show,
                step=batch_idx)
