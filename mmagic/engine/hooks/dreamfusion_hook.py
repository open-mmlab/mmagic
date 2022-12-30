# Copyright (c) OpenMMLab. All rights reserved.
import random

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmedit.registry import HOOKS


@HOOKS.register_module()
class DreamFusionTrainingHook(Hook):

    def __init__(self, albedo_iters: int):
        super().__init__()
        self.albedo_iters = albedo_iters

        self.shading_test = 'albedo'
        self.ambident_ratio_test = 1.0

    def set_shading_and_ambient(self, runner, shading: str,
                                ambient_ratio: str) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        renderer = model.renderer
        if is_model_wrapper(renderer):
            renderer = renderer.module
        renderer.set_shading(shading)
        renderer.set_ambient_ratio(ambient_ratio)

    def after_train_iter(self, runner, batch_idx: int, *args,
                         **kwargs) -> None:
        if batch_idx < self.albedo_iters or self.albedo_iters == -1:
            shading = 'albedo'
            ambient_ratio = 1.0
        else:
            rand = random.random()
            if rand > 0.8:  # NOTE: this should be 0.75 in paper
                shading = 'albedo'
                ambient_ratio = 1.0
            elif rand > 0.4:  # NOTE: this should be 0.75 * 0.5 = 0.325
                shading = 'textureless'
                ambient_ratio = 0.1
            else:
                shading = 'lambertian'
                ambient_ratio = 0.1
        self.set_shading_and_ambient(runner, shading, ambient_ratio)

    def before_test(self, runner) -> None:
        self.set_shading_and_ambient(runner, self.shading_test,
                                     self.ambident_ratio_test)

    def before_val(self, runner) -> None:
        self.set_shading_and_ambient(runner, self.shading_test,
                                     self.ambident_ratio_test)
