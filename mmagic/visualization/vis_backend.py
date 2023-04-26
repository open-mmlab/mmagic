# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import Optional, Union

import cv2
import imageio
import numpy as np
import torch
from mmengine import MessageHub
from mmengine.config import Config
from mmengine.fileio import dump, get_file_backend
from mmengine.visualization import BaseVisBackend
from mmengine.visualization import \
    TensorboardVisBackend as BaseTensorboardVisBackend
from mmengine.visualization import WandbVisBackend as BaseWandbVisBackend
from mmengine.visualization.vis_backend import force_init_env

from mmagic.registry import VISBACKENDS


@VISBACKENDS.register_module()
class VisBackend(BaseVisBackend):
    """MMagic visualization backend class. It can write image, config, scalars,
    etc. to the local hard disk and ceph path. You can get the drawing backend
    through the experiment property for custom drawing.

    Examples:
        >>> from mmagic.visualization import VisBackend
        >>> import numpy as np
        >>> vis_backend = VisBackend(save_dir='temp_dir',
        >>>                          ceph_path='s3://temp-bucket')
        >>> img = np.random.randint(0, 256, size=(10, 10, 3))
        >>> vis_backend.add_image('img', img)
        >>> vis_backend.add_scalar('mAP', 0.6)
        >>> vis_backend.add_scalars({'loss': [1, 2, 3], 'acc': 0.8})
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> vis_backend.add_config(cfg)
    Args:
        save_dir (str): The root directory to save the files produced by the
            visualizer.
        img_save_dir (str): The directory to save images.
            Default to 'vis_image'.
        config_save_file (str): The file name to save config.
            Default to 'config.py'.
        scalar_save_file (str):  The file name to save scalar values.
            Default to 'scalars.json'.
        ceph_path (Optional[str]): The remote path of Ceph cloud storage.
            Defaults to None.
        delete_local (bool): Whether delete local after uploading to ceph or
            not. If ``ceph_path`` is None, this will be ignored. Defaults to
            True.
    """

    def __init__(self,
                 save_dir: str,
                 img_save_dir: str = 'vis_image',
                 config_save_file: str = 'config.py',
                 scalar_save_file: str = 'scalars.json',
                 ceph_path: Optional[str] = None,
                 delete_local_image: bool = True):
        assert config_save_file.split('.')[-1] == 'py'
        assert scalar_save_file.split('.')[-1] == 'json'
        super().__init__(save_dir)
        self._img_save_dir = img_save_dir
        self._config_save_file = config_save_file
        self._scalar_save_file = scalar_save_file

        self._ceph_path = ceph_path
        self._file_client = None
        self._delete_local_image = delete_local_image

        self._cfg = None

    def _init_env(self):
        if self._env_initialized:
            return
        self._env_initialized = True
        """Init save dir."""
        os.makedirs(self._save_dir, exist_ok=True)
        self._img_save_dir = osp.join(
            self._save_dir,  # type: ignore
            self._img_save_dir)
        self._config_save_file = osp.join(
            self._save_dir,  # type: ignore
            self._config_save_file)
        self._scalar_save_file = osp.join(
            self._save_dir,  # type: ignore
            self._scalar_save_file)

        if self._ceph_path is not None:
            # work_dir: A/B/.../C/D
            # ceph_path: s3://a/b
            # local_files:  A/B/.../C/D/TIME_STAMP/vis_data/
            # remote files: s3://a/b/D/TIME_STAMP/vis_data/
            if self._cfg is None or self._cfg.get('work_dir', None) is None:
                message_hub = MessageHub.get_current_instance()
                cfg_str = message_hub.get_info('cfg')
                self._cfg = Config.fromstring(cfg_str, '.py')
            full_work_dir = osp.abspath(self._cfg['work_dir'])
            if full_work_dir.endswith('/'):
                full_work_dir = full_work_dir[:-1]

            # NOTE: handle src_path with `os.sep`, because may windows
            # and linux may have different separate.
            src_path = os.sep.join(full_work_dir.split(os.sep)[:-1])
            # NOTE: handle tar_path with '/', because ceph use linux
            # environment
            tar_path = self._ceph_path[:-1] if \
                self._ceph_path.endswith('/') else self._ceph_path

            backend_args = dict(
                backend='petrel', path_mapping={src_path: tar_path})
            self._file_client = get_file_backend(backend_args=backend_args)

    @property  # type: ignore
    @force_init_env
    def experiment(self) -> 'VisBackend':
        """Return the experiment object associated with this visualization
        backend."""
        return self

    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config to disk.

        Args:
            config (Config): The Config object
        """
        assert isinstance(config, Config)
        self._cfg = config
        self._init_env()
        config.dump(self._config_save_file)
        self._upload(self._config_save_file)

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.array,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to disk.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB. Default to None.
            step (int): Global step value to record. Default to 0.
        """
        assert image.dtype == np.uint8
        os.makedirs(self._img_save_dir, exist_ok=True)
        if image.ndim == 3:
            drawn_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            save_file_name = f'{name}_{step}.png'
            cv2.imwrite(
                osp.join(self._img_save_dir, save_file_name), drawn_image)
        elif image.ndim == 4:
            n_skip = kwargs.get('n_skip', 1)
            fps = kwargs.get('fps', 60)
            save_file_name = f'{name}_{step}.gif'
            save_file_path = osp.join(self._img_save_dir, save_file_name)

            frames_list = []
            for frame in image[::n_skip]:
                frames_list.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if not (image.shape[0] % n_skip == 0):
                frames_list.append(image[-1])
            imageio.mimsave(
                save_file_path, frames_list, 'GIF', duration=1000. / fps)
        else:
            raise ValueError(
                'Only support visualize image with dimension of 3 or 4. But '
                f'receive input with shape \'{image.shape}\'.')
        self._upload(
            osp.join(self._img_save_dir, save_file_name),
            self._delete_local_image)

    @force_init_env
    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data to disk.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Global step value to record. Default to 0.
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        self._dump({name: value, 'step': step}, self._scalar_save_file, 'json')

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalars to disk.

        The scalar dict will be written to the default and
        specified files if ``file_path`` is specified.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values. The value must be dumped
                into json format.
            step (int): Global step value to record. Default to 0.
            file_path (str, optional): The scalar's data will be
                saved to the ``file_path`` file at the same time
                if the ``file_path`` parameter is specified.
                Default to None.
        """
        assert isinstance(scalar_dict, dict)
        scalar_dict_new = dict()
        for k, v in scalar_dict.items():
            if isinstance(v, torch.Tensor):
                scalar_dict_new[k] = v.item()
            else:
                scalar_dict_new[k] = v
        scalar_dict_new.setdefault('step', step)

        if file_path is not None:
            assert file_path.split('.')[-1] == 'json'
            new_save_file_path = osp.join(
                self._save_dir,  # type: ignore
                file_path)
            assert new_save_file_path != self._scalar_save_file, \
                '``file_path`` and ``scalar_save_file`` have the ' \
                'same name, please set ``file_path`` to another value'
            self._dump(scalar_dict_new, new_save_file_path, 'json')
        self._dump(scalar_dict_new, self._scalar_save_file, 'json')
        self._upload(self._scalar_save_file)

    def _dump(self, value_dict: dict, file_path: str,
              file_format: str) -> None:
        """dump dict to file.

        Args:
           value_dict (dict) : The dict data to saved.
           file_path (str): The file path to save data.
           file_format (str): The file format to save data.
        """
        with open(file_path, 'a+') as f:
            dump(value_dict, f, file_format=file_format)
            f.write('\n')

    def _upload(self, path: str, delete_local=False) -> None:
        """Upload file at path to remote.

        Args:
            path (str): Path of file to upload.
        """
        if self._file_client is None:
            return
        with open(path, 'rb') as file:
            self._file_client.put(file, path)
        if delete_local:
            os.remove(path)


@VISBACKENDS.register_module()
class TensorboardVisBackend(BaseTensorboardVisBackend):

    @force_init_env
    def add_image(self, name: str, image: np.array, step: int = 0, **kwargs):
        """Record the image to Tensorboard. Additional support upload gif
        files.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB.
            step (int): Useless parameter. Wandb does not
                need this parameter. Default to 0.
        """

        if image.ndim == 4:
            n_skip = kwargs.get('n_skip', 1)
            fps = kwargs.get('fps', 60)

            frames_list = []
            for frame in image[::n_skip]:
                frames_list.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if not (image.shape[0] % n_skip == 0):
                frames_list.append(image[-1])

            frames_np = np.transpose(
                np.stack(frames_list, axis=0), (0, 3, 1, 2))
            frames_tensor = torch.from_numpy(frames_np)[None, ...]
            self._tensorboard.add_video(
                name, frames_tensor, global_step=step, fps=fps)
        else:
            # write normal image
            self._tensorboard.add_image(name, image, step, dataformats='HWC')


@VISBACKENDS.register_module()
class PaviVisBackend(BaseVisBackend):
    """Visualization backend for Pavi."""

    def __init__(self,
                 save_dir: str,
                 exp_name: Optional[str] = None,
                 labels: Optional[str] = None,
                 project: Optional[str] = None,
                 model: Optional[str] = None,
                 description: Optional[str] = None):
        self.save_dir = save_dir

        self._name = exp_name
        self._labels = labels
        self._project = project
        self._model = model
        self._description = description

    def _init_env(self):
        """Init save dir."""
        try:
            import pavi
        except ImportError:
            raise ImportError(
                'To use \'PaviVisBackend\' Pavi must be installed.')
        self._pavi = pavi.SummaryWriter(
            name=self._name,
            labels=self._labels,
            project=self._project,
            model=self._model,
            description=self._description,
            log_dir=self.save_dir)

    @property  # type: ignore
    @force_init_env
    def experiment(self) -> 'VisBackend':
        """Return the experiment object associated with this visualization
        backend."""
        return self._pavi

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.array,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to Pavi.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB. Default to None.
            step (int): Global step value to record. Default to 0.
        """
        assert image.dtype == np.uint8
        drawn_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self._pavi.add_image(name, drawn_image, step)

    @force_init_env
    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data to Pavi.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Global step value to record. Default to 0.
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        self._pavi.add_scalar(name, value, step)

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalars to Pavi.

        The scalar dict will be written to the default and
        specified files if ``file_path`` is specified.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values. The value must be dumped
                into json format.
            step (int): Global step value to record. Default to 0.
            file_path (str, optional): The scalar's data will be
                saved to the ``file_path`` file at the same time
                if the ``file_path`` parameter is specified.
                Default to None.
        """
        assert isinstance(scalar_dict, dict)
        for name, value in scalar_dict.items():
            self.add_scalar(name, value, step)


@VISBACKENDS.register_module()
class WandbVisBackend(BaseWandbVisBackend):
    """Wandb visualization backend for MMagic."""

    def _init_env(self):
        """Setup env for wandb."""
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir, exist_ok=True)  # type: ignore
        if self._init_kwargs is None:
            self._init_kwargs = {'dir': self._save_dir}
        else:
            self._init_kwargs.setdefault('dir', self._save_dir)
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')

        # add timestamp at the end of name
        timestamp = self._save_dir.split('/')[-2]
        orig_name = self._init_kwargs.get('name', None)
        if orig_name:
            self._init_kwargs['name'] = f'{orig_name}_{timestamp}'
        wandb.init(**self._init_kwargs)
        self._wandb = wandb

    @force_init_env
    def add_image(self, name: str, image: np.array, step: int = 0, **kwargs):
        """Record the image to wandb. Additional support upload gif files.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB.
            step (int): Useless parameter. Wandb does not
                need this parameter. Default to 0.
        """
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')

        if image.ndim == 4:
            n_skip = kwargs.get('n_skip', 1)
            fps = kwargs.get('fps', 60)

            frames_list = []
            for frame in image[::n_skip]:
                frames_list.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if not (image.shape[0] % n_skip == 0):
                frames_list.append(image[-1])

            frames_np = np.transpose(
                np.stack(frames_list, axis=0), (0, 3, 1, 2))
            self._wandb.log(
                {name: wandb.Video(frames_np, fps=fps, format='gif')},
                commit=self._commit)
        else:
            # write normal image
            self._wandb.log({name: wandb.Image(image)}, commit=self._commit)
