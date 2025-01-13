

import torch
import os
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from blipnet.utils.utils import generate_plot_grid
from blipnet.utils.utils import fig_to_array


blipnet_dataset_config = {
    "dataset_folder":   "data/",
    "dataset_files":    [""],
}


class BlipNetDataset(Dataset):
    """
    """
    def __init__(
        self,
        name:   str = "blipnet",
        config: dict = blipnet_dataset_config,
        meta:   dict = {}
    ):
        self.name = name
        self.config = config
        self.meta = meta

        self.process_config()

    def process_config(self):
        self.num_events = 10

        self.dataset_folder = self.config['dataset_folder']
        self.files = [
            file for file in os.listdir(path=self.dataset_folder)
            if os.path.isfile(os.path.join(self.dataset_folder, file))
        ]

        if "normalized" not in self.config:
            self.config["normalized"] = False
        self.normalized = self.config["normalized"]

        if self.normalized:
            pass

        self.dataset_type = self.config['dataset_type']

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        data = {
            'positions': torch.Tensor([[0, 1, 2]]),
            'features': torch.Tensor([[0, 1, 2, 3]]),
            'labels': torch.Tensor([[0, 1]])
        }
        if self.normalized:
            data = self.normalize(data)
        return data

    def normalize(
        self,
        data
    ):
        return data

    def unnormalize(
        self,
        data,
    ):
        return data

    def save_predictions(
        self,
        model_name,
        predictions,
        indices
    ):
        pass

    def evaluate_outputs(
        self,
        data,
        data_type='training'
    ):
        """
        Here we make plots of the distributions of gut_test/gut_true before and after the autoencorder,
        as well as different plots of the latent projections, binary variables, etc.
        """
        if self.normalized:
            data = self.unnormalize(data)