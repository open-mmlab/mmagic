# Copyright (c) OpenMMLab. All rights reserved.
import os

from mmagic.datasets import TextualInversionDataset

data_dir = os.path.join(__file__, '../', '../', 'data', 'controlnet')
concept_dir = os.path.join(data_dir, 'source')
placeholder = 'S*'
template = os.path.join(__file__, '../', '../', 'data', 'textual_inversion',
                        'imagenet_templates_small.txt')
with_image_reference = True


def test_textual_inversion_dataset():
    print(os.path.abspath(data_dir))
    dataset = TextualInversionDataset(
        data_root=data_dir,
        concept_dir=concept_dir,
        placeholder=placeholder,
        template=template,
        with_image_reference=with_image_reference)
    assert len(dataset) == 2
