# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmagic.models.archs import TokenizerWrapper

PREFIX = '<|startoftext|>'
SUFFIX = ' <|endoftext|>'


class TestTokenizerWrapper(TestCase):

    def setUp(self):
        """Test add placeholder tokens in this function."""
        tokenizer = TokenizerWrapper('openai/clip-vit-base-patch32')

        # 'Goodbye' in kiswahili
        tokenizer.add_placeholder_token('kwaheri', num_vec_per_token=1)
        # 'how much' in kiswahili
        tokenizer.add_placeholder_token('ngapi', num_vec_per_token=4)

        with self.assertRaises(AssertionError):
            tokenizer.add_placeholder_token('hello', num_vec_per_token=1)

        self.tokenizer = tokenizer

    def test_encode_and_decode_and_call(self):
        # test single token
        text = 'Nice bro, kwaheri!'
        input_ids = self.tokenizer.encode(text).input_ids
        self.assertEqual(input_ids[-3:-2],
                         self.tokenizer.encode('kwaheri').input_ids[1:-1])
        text_recon = self.tokenizer.decode(input_ids)
        text_recon_raw = self.tokenizer.decode(input_ids, return_raw=True)
        self.assertEqual(text_recon, f'{PREFIX}{text}{SUFFIX}'.lower())
        self.assertEqual(text_recon_raw, f'{PREFIX}{text}{SUFFIX}'.lower())
        self.assertEqual(input_ids, self.tokenizer(text).input_ids)

        # test multi token
        text = 'This apple seems delicious, ngapi?'
        input_ids = self.tokenizer.encode(text).input_ids
        self.assertEqual(input_ids[6:-2],
                         self.tokenizer.encode('ngapi').input_ids[1:-1])
        text_recon = self.tokenizer.decode(input_ids)
        text_recon_raw = self.tokenizer.decode(input_ids, return_raw=True)
        self.assertEqual(text_recon, f'{PREFIX}{text}{SUFFIX}'.lower())
        for idx in range(4):
            self.assertIn(f'ngapi_{idx}', text_recon_raw)
        self.assertEqual(input_ids, self.tokenizer(text).input_ids)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
