# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmengine.config import Config, ConfigDict
from mmengine.runner import load_checkpoint
from mmengine.structures import BaseDataElement

from mmedit.registry import MODELS
from mmedit.utils import ConfigType, SampleList
from .inference_functions import set_random_seed

InputType = Union[str, int, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[BaseDataElement, SampleList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict], BaseDataElement, List[BaseDataElement]]


class BaseMMEditInferencer:
    """Base inferencer.

    Args:
        config (str or ConfigType): Model config or the path to it.
        ckpt (str, optional): Path to the checkpoint.
        device (str, optional): Device to run inference. If None, the best
            device will be automatically used.
        result_out_dir (str): Output directory of images. Defaults to ''.
    """

    func_kwargs = dict(
        preprocess=[],
        forward=[],
        visualize=['result_out_dir'],
        postprocess=['get_datasample'])
    func_order = dict(preprocess=0, forward=1, visualize=2, postprocess=3)

    extra_parameters = dict()

    def __init__(self,
                 config: Union[ConfigType, str],
                 ckpt: Optional[str],
                 device: Optional[str] = None,
                 extra_parameters: Optional[Dict] = None,
                 seed: int = 2022,
                 **kwargs) -> None:
        # Load config to cfg
        if isinstance(config, str):
            config = Config.fromfile(config)
        elif not isinstance(config, (ConfigDict, Config)):
            raise TypeError('config must be a filename or any ConfigType'
                            f'object, but got {type(config)}')
        self.cfg = config

        if config.model.get('pretrained'):
            config.model.pretrained = None

        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self._init_model(config, ckpt, device)
        self._init_extra_parameters(extra_parameters)
        self.base_params = self._dispatch_kwargs(**kwargs)
        self.seed = seed
        set_random_seed(self.seed)

    def _init_model(self, cfg: Union[ConfigType, str], ckpt: Optional[str],
                    device: str) -> None:
        """Initialize the model with the given config and checkpoint on the
        specific device."""
        model = MODELS.build(cfg.model)
        if ckpt is not None and ckpt != '':
            ckpt = load_checkpoint(model, ckpt, map_location='cpu')
        model.cfg = cfg
        model.to(device)
        model.eval()
        self.model = model

    def _init_extra_parameters(self, extra_parameters: Dict) -> None:
        """Initialize extra_parameters of each kind of inferencer."""
        if extra_parameters is not None:
            for key in self.extra_parameters.keys():
                if key in extra_parameters.keys():
                    self.extra_parameters[key] = extra_parameters[key]

    def _update_extra_parameters(self, **kwargs) -> None:
        """update extra_parameters during run time."""
        if 'extra_parameters' in kwargs:
            input_extra_parameters = kwargs['extra_parameters']
            if input_extra_parameters is not None:
                for key in self.extra_parameters.keys():
                    if key in input_extra_parameters.keys():
                        self.extra_parameters[key] = \
                            input_extra_parameters[key]

    def _dispatch_kwargs(self, **kwargs) -> Tuple[Dict, Dict, Dict, Dict]:
        """Dispatch kwargs to preprocess(), forward(), visualize() and
        postprocess() according to the actual demands."""
        results = [{}, {}, {}, {}]
        dispatched_kwargs = set()

        # Dispatch kwargs according to self.func_kwargs
        for func_name, func_kwargs in self.func_kwargs.items():
            for func_kwarg in func_kwargs:
                if func_kwarg in kwargs:
                    dispatched_kwargs.add(func_kwarg)
                    results[self.func_order[func_name]][func_kwarg] = kwargs[
                        func_kwarg]

        return results

    @torch.no_grad()
    def __call__(self, **kwargs) -> Union[Dict, List[Dict]]:
        """Call the inferencer.

        Args:
            kwargs: Keyword arguments for the inferencer.

        Returns:
            Union[Dict, List[Dict]]: Results of inference pipeline.
        """
        self._update_extra_parameters(**kwargs)

        params = self._dispatch_kwargs(**kwargs)
        preprocess_kwargs = self.base_params[0].copy()
        preprocess_kwargs.update(params[0])
        forward_kwargs = self.base_params[1].copy()
        forward_kwargs.update(params[1])
        visualize_kwargs = self.base_params[2].copy()
        visualize_kwargs.update(params[2])
        postprocess_kwargs = self.base_params[3].copy()
        postprocess_kwargs.update(params[3])

        data = self.preprocess(**preprocess_kwargs)
        preds = self.forward(data, **forward_kwargs)
        imgs = self.visualize(preds, **visualize_kwargs)
        results = self.postprocess(preds, imgs, **postprocess_kwargs)
        return results

    def get_extra_parameters(self) -> List[str]:
        """Each inferencer may has its own parameters. Call this function to
        get these parameters.

        Returns:
            List[str]: List of unique parameters.
        """
        return list(self.extra_parameters.keys())

    @abstractmethod
    def preprocess(self, inputs: InputsType) -> Dict:
        """Process the inputs into a model-feedable format.

        Args:
            inputs (List[Union[str, np.ndarray]]): Inputs for the inferencer.

        Returns:
            Dict: Result of preprocess
        """

    @abstractmethod
    def forward(self, inputs: InputsType) -> PredType:
        """Forward the inputs to the model."""

    @abstractmethod
    def visualize(self,
                  inputs: InputsType,
                  preds: PredType,
                  result_out_dir: str = None) -> List[np.ndarray]:
        """Visualize predictions.

        Args:
            inputs (List[Union[str, np.ndarray]]): Inputs for the inferencer.
            preds (List[Dict]): Predictions of the model.
            result_out_dir (str): Output directory of images. Defaults to ''.

        Returns:
            List[np.ndarray]: Result of visualize
        """

    def postprocess(
        self,
        preds: PredType,
        imgs: Optional[List[np.ndarray]] = None,
        is_batch: bool = False,
        get_datasample: bool = False,
    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        """Postprocess predictions.

        Args:
            preds (List[Dict]): Predictions of the model.
            imgs (Optional[np.ndarray]): Visualized predictions.
            is_batch (bool): Whether the inputs are in a batch.
                Defaults to False.
            get_datasample (bool): Whether to use Datasample to store
                inference results. If False, dict will be used.

        Returns:
            result (Dict): Inference results as a dict.
            imgs (torch.Tensor): Image result of inference as a tensor or
                tensor list.
        """
        results = preds
        if not get_datasample:
            results = []
            for pred in preds:
                result = self._pred2dict(pred)
                results.append(result)
        if not is_batch:
            results = results[0]
        return results, imgs

    def _pred2dict(self, pred_tensor: torch.Tensor) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary. It's better to contain only basic data elements such as
        strings and numbers in order to guarantee it's json-serializable.

        Args:
            pred_tensor (torch.Tensor): The tensor to be converted.

        Returns:
            dict: The output dictionary.
        """
        result = {}
        result['infer_results'] = pred_tensor
        return result
