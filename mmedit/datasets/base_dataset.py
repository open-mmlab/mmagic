import copy
from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset

from .pipelines import Compose


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for datasets.

    All datasets should subclass it.
    All subclasses should overwrite:

        ``load_annotations``, supporting to load information and generate
        image lists.

    Args:
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): If True, the dataset will work in test mode.
            Otherwise, in train mode.
    """

    def __init__(self, pipeline, test_mode=False):
        super(BaseDataset, self).__init__()
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)

    @abstractmethod
    def load_annotations(self):
        """Abstract function for loading annotation.

        All subclasses should overwrite this function
        """

    def prepare_train_data(self, idx):
        """Prepare training data.

        Args:
            idx (int): Index of the training batch data.

        Returns:
            dict: Returned training batch.
        """
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        """Prepare testing data.

        Args:
            idx (int): Index for getting each testing batch.

        Returns:
            Tensor: Returned testing batch.
        """
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data_infos)

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        if not self.test_mode:
            return self.prepare_train_data(idx)
        else:
            return self.prepare_test_data(idx)
