import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets.folder import default_loader, make_dataset

from easyfl.datasets.simulation import data_simulation, SIMULATE_IID

logger = logging.getLogger(__name__)

TEST_IN_SERVER = "test_in_server"
TEST_IN_CLIENT = "test_in_client"

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

DEFAULT_MERGED_ID = "Merged"


def default_process_x(raw_x_batch):
    return torch.tensor(raw_x_batch)


def default_process_y(raw_y_batch):
    return torch.tensor(raw_y_batch)


class FederatedDataset(ABC):
    """The abstract class of federated dataset for EasyFL."""

    def __init__(self):
        pass

    @abstractmethod
    def loader(self, batch_size, shuffle=True):
        """Get data loader.

        Args:
            batch_size (int): The batch size of the data loader.
            shuffle (bool): Whether shuffle the data in the loader.
        """
        raise NotImplementedError("Data loader not implemented")

    @abstractmethod
    def size(self, cid):
        """Get dataset size.

        Args:
            cid (str): client id.
        """
        raise NotImplementedError("Size not implemented")

    @property
    def users(self):
        """Get client ids of the federated dataset."""
        raise NotImplementedError("Users not implemented")


class FederatedTorchDataset(FederatedDataset):
    """Wrapper over PyTorch dataset.

    Args:
        data (dict): A dictionary of client datasets, format {"client_id": dataset1, "client_id2": dataset2}.
    """

    def __init__(self, data, users):
        super(FederatedTorchDataset, self).__init__()
        self.data = data
        self._users = users

    def loader(self, batch_size, client_id=None, shuffle=True, drop_last=False, seed=0, num_workers=32):
        if client_id is None:
            data = self.data
        else:
            data = self.data[client_id]

        loader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
            drop_last=drop_last)
        return loader

    @property
    def users(self):
        return self._users

    @users.setter
    def users(self, value):
        self._users = value

    def size(self, cid=None):
        if cid is not None:
            return len(self.data[cid])
        else:
            return len(self.data)
