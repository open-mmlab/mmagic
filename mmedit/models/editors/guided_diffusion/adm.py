import mmengine
import torch
from mmengine.model import BaseModel
from mmengine.runner.checkpoint import _load_checkpoint_with_prefix
from tqdm import tqdm

from mmedit.registry import DIFFUSERS, MODELS, MODULES


@MODELS.register_module('ADM')
class AblatedDiffusionModel(BaseModel):

    def __init__(self,
                 data_preprocessor,
                 unet,
                 diffuser,
                 classifier=None,
                 use_fp16=False,
                 pretrained_cfgs=None):
        super().__init__(data_preprocessor=data_preprocessor)
        self.unet = MODULES.build(unet)
        self.diffuser = DIFFUSERS.build(diffuser)
        if classifier:
            self.classifier = MODULES.build(unet)
        if pretrained_cfgs:
            self.load_pretrained_models(pretrained_cfgs)
        if use_fp16:
            self.convert_to_fp16()

    def load_pretrained_models(self, pretrained_cfgs):
        for key, ckpt_cfg in pretrained_cfgs.items():
            prefix = ckpt_cfg.get('prefix', '')
            map_location = ckpt_cfg.get('map_location', 'cpu')
            strict = ckpt_cfg.get('strict', True)
            ckpt_path = ckpt_cfg.get('ckpt_path')
            state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path,
                                                      map_location)
            getattr(self, key).load_state_dict(state_dict, strict=strict)
            mmengine.print_log(f'Load pretrained {key} from {ckpt_path}')

    def convert_to_fp16(self):
        pass

    @property
    def device(self):
        """Get current device of the model.

        Returns:
            torch.device: The current device of the model.
        """
        return next(self.parameters()).device

    def infer(self,
              batch_size=1,
              num_inference_steps=1000,
              label_id=-1,
              show_progress=False):
        # Sample gaussian noise to begin loop
        image = torch.randn((batch_size, self.unet.in_channels,
                             self.unet.image_size, self.unet.image_size))
        image = image.to(self.device)
        labels = torch.randint(
            low=0,
            high=self.unet.num_classes,
            size=(batch_size, ),
            device=self.device)

        # set step values
        if num_inference_steps > 0:
            self.diffuser.set_timesteps(num_inference_steps)

        timesteps = self.diffuser.timesteps
        if show_progress:
            timesteps = tqdm(timesteps)
        for t in timesteps:
            # 1. predict noise model_output
            model_output = self.unet(image, t, label=labels)["outputs"]

            # 2. compute previous image: x_t -> t_t-1
            image = self.diffuser.step(model_output, t, image)["prev_sample"]

        return {"samples": image}

    def forward(self, x):
        pass
