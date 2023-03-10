# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmedit.evaluation import MultiScaleStructureSimilarity
from mmedit.structures import EditDataSample
from mmedit.utils import register_all_modules

register_all_modules()


class TestMS_SSIM(TestCase):

    def test_init(self):
        MS_SSIM = MultiScaleStructureSimilarity(
            fake_nums=10, fake_key='fake', sample_model='ema')
        self.assertEqual(MS_SSIM.num_pairs, 5)

        with self.assertRaises(AssertionError):
            MultiScaleStructureSimilarity(fake_nums=9)

    def test_process_and_evaluate(self):
        MS_SSIM = MultiScaleStructureSimilarity(
            fake_nums=4, fake_key='fake', sample_model='ema')

        input_batch_size = 6
        input_pairs = 6 // 2
        gen_images = torch.randint(0, 255, (input_batch_size, 3, 32, 32))
        gen_samples = [
            EditDataSample(fake_img=img).to_dict() for img in gen_images
        ]

        MS_SSIM.process(None, gen_samples)
        MS_SSIM.process(None, gen_samples)
        self.assertEqual(len(MS_SSIM.fake_results), input_pairs)
        metric_1 = MS_SSIM.evaluate()
        self.assertTrue('avg' in metric_1)

        MS_SSIM.fake_results.clear()
        MS_SSIM.process(None, gen_samples[:4])
        self.assertEqual(len(MS_SSIM.fake_results), 4 // 2)
        metric_2 = MS_SSIM.evaluate()
        self.assertTrue('avg' in metric_2)

        MS_SSIM.fake_results.clear()
        gen_samples = [
            EditDataSample(
                ema=EditDataSample(
                    fake_img=torch.randint(0, 255, (3, 32, 32))),
                orig=EditDataSample(
                    fake_img=torch.randint(0, 255, (3, 32, 32)))).to_dict()
        ] * 2
        MS_SSIM.process(None, gen_samples)

        gen_samples = [
            EditDataSample(
                ema=EditDataSample(
                    fake=torch.randint(0, 255, (3, 32, 32)))).to_dict()
        ] * 2
        MS_SSIM.process(None, gen_samples)

        # test prefix
        MS_SSIM = MultiScaleStructureSimilarity(
            fake_nums=4, fake_key='fake', sample_model='ema', prefix='ms-ssim')
