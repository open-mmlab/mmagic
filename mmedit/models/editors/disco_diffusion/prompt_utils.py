# Copyright (c) OpenMMLab. All rights reserved.
import torchvision.transforms as T

normalize = T.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711])
