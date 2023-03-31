# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock, patch

from mmedit.models.editors import ClipWrapper


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
