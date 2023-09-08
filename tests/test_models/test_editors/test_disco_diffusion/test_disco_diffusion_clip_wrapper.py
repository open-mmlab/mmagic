# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from mmagic.models.archs import TokenizerWrapper
from mmagic.models.editors import ClipWrapper
from mmagic.models.editors.disco_diffusion.clip_wrapper import \
    EmbeddingLayerWithFixes


class TestClipWrapper(TestCase):

    def test_clip_not_installed(self):
        with patch.dict('sys.modules', {'clip': None}):
            with self.assertRaises(ImportError):
                ClipWrapper('clip')

    def test_open_clip_not_installed(self):
        with patch.dict('sys.modules', {'open_clip': None}):
            with self.assertRaises(ImportError):
                ClipWrapper('open_clip')

    def test_transformers_not_installed(self):
        with patch.dict('sys.modules', {'transformers': None}):
            with self.assertRaises(ImportError):
                ClipWrapper('huggingface')

    @patch('clip.load')
    def test_clip_load(self, mock_clip_load):
        mock_model = MagicMock()
        mock_clip_load.return_value = (mock_model, None)

        model = ClipWrapper('clip', name='test_model')

        mock_clip_load.assert_called_once_with(name='test_model')
        self.assertEqual(model.model, mock_model)

    def test_open_clip_load(self):
        mock_model = MagicMock()
        create_model_mock = MagicMock()
        create_model_mock.return_value = mock_model

        open_clip_mock = MagicMock()
        open_clip_mock.create_model = create_model_mock

        with patch.dict('sys.modules', {'open_clip': open_clip_mock}):
            model = ClipWrapper('open_clip', model_name='RN50')
            create_model_mock.assert_called_once_with(model_name='RN50')
            self.assertEqual(model.model, mock_model)

    @patch('transformers.CLIPTextModel.from_pretrained')
    def test_huggingface_load(self, mock_from_pretrained):
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_from_pretrained.return_value = mock_model

        model = ClipWrapper(
            'huggingface',
            pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5',
            subfolder='text_encoder')
        mock_from_pretrained.assert_called_once_with(
            pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5',
            subfolder='text_encoder')
        self.assertEqual(model.model, mock_model)
        self.assertEqual(model.model.config, mock_model.config)


def test_embedding_layer_with_fixes():
    embedding_layer = nn.Embedding(10, 15)
    print(embedding_layer.weight.shape)
    embedding_layer_wrapper = EmbeddingLayerWithFixes(embedding_layer)

    assert embedding_layer_wrapper.external_embeddings == []
    # test naive forward
    input_ids = torch.randint(0, 10, (3, 20)).type(torch.long)
    out_feat = embedding_layer_wrapper(input_ids)
    print(out_feat.shape)

    tokenizer = TokenizerWrapper('openai/clip-vit-base-patch32')
    # 'Goodbye' in kiswahili
    tokenizer.add_placeholder_token('kwaheri', num_vec_per_token=1)
    # 'how much' in kiswahili
    tokenizer.add_placeholder_token('ngapi', num_vec_per_token=4)

    # test add single embedding
    new_embedding = {
        'name': 'kwaheri',  # 'Goodbye' in kiswahili
        'embedding': torch.ones(1, 15) * 4,
        'start': tokenizer.get_token_info('kwaheri')['start'],
        'end': tokenizer.get_token_info('kwaheri')['end']
    }
    embedding_layer_wrapper.add_embeddings(new_embedding)
    input_text = ['hello world, kwaheri!', 'hello world', 'kwaheri']
    input_ids = tokenizer(
        input_text, padding='max_length', truncation=True,
        return_tensors='pt')['input_ids']
    out_feat = embedding_layer_wrapper(input_ids)
    assert (out_feat[0, 4] == 4).all()
    assert (out_feat[2, 1] == 4).all()

    new_embedding = {
        'name': 'ngapi',  # 'how much' in kiswahili
        'embedding': torch.ones(4, 15) * 2.3,
        'start': tokenizer.get_token_info('ngapi')['start'],
        'end': tokenizer.get_token_info('ngapi')['end']
    }
    embedding_layer_wrapper.add_embeddings(new_embedding)
    input_text = ['hello, ngapi!', 'hello world', 'kwaheri my friend, ngapi?']
    input_ids = tokenizer(
        input_text, padding='max_length', truncation=True,
        return_tensors='pt')['input_ids']
    out_feat = embedding_layer_wrapper(input_ids)
    assert (out_feat[0, 3:7] == 2.3).all()
    assert (out_feat[2, 5:9] == 2.3).all()


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
