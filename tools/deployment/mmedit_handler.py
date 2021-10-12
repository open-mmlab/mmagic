# Copyright (c) OpenMMLab. All rights reserved.
import os

import torch
from ts.torch_handler.base_handler import BaseHandler

from mmedit.apis import init_model, restoration_inference
from mmedit.core import tensor2img


class MMEditHandler(BaseHandler):

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')) if torch.cuda.
                                   is_available() else self.map_location)
        self.manifest = context.manifest

        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        checkpoint = os.path.join(model_dir, serialized_file)
        self.config_file = os.path.join(model_dir, 'config.py')

        self.model = init_model(self.config_file, checkpoint, self.device)
        self.initialized = True

    def preprocess(self, data, *args, **kwargs):
        # data preprocess is in inference.
        return data

    def inference(self, data, *args, **kwargs):
        results = restoration_inference(self.model, data)
        return results

    def postprocess(self, data):
        # convert torch tensor to numpy and then covert to bytes
        output_list = []
        for data_ in data:
            data_np = tensor2img(data_)
            data_byte = data_np.tobytes()
            output_list.append(data_byte)

        return output_list
