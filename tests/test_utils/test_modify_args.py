# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from unittest.mock import patch

from mmedit.utils import modify_args


def test_modify_args():

    def _parse_args():
        parser = argparse.ArgumentParser(description='Generation demo')
        parser.add_argument('--config-path', help='test config file path')
        args = parser.parse_args()
        return args

    with patch('argparse._sys.argv', ['test.py', '--config_path=config.py']):
        modify_args()
        args = _parse_args()
        assert args.config_path == 'config.py'
