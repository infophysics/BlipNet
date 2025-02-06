

import torch
import os
import numpy as np
import h5py
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
        self.create_index_mapping()

    def process_config(self):
        self.dataset_folder = self.config['dataset_folder']
        self.files = [
            os.path.join(self.dataset_folder, file)
            for file in os.listdir(self.dataset_folder)
            if file.endswith(".h5")
        ]

        # Validate the specified files
        if self.config.get('dataset_files'):
            self.files = [
                os.path.join(self.dataset_folder, file)
                for file in self.config['dataset_files']
                if file in os.listdir(self.dataset_folder)
            ]

        self.position_indices = self.config["positions"]
        self.features_indices = self.config["features"]
        self.fragment_truth_indices = self.config["fragment_truth"]
        self.interaction_truth_indices = self.config["interaction_truth"]

        if "normalized" not in self.config:
            self.config["normalized"] = False
        self.normalized = self.config["normalized"]

        if self.normalized:
            pass

        self.dataset_type = self.config['dataset_type']

    def create_index_mapping(self):
        """
        Create a mapping from global index to (file_idx, event_idx).
        """
        self.index_map = []
        for file_idx, file in enumerate(self.files):
            with h5py.File(file, 'r') as f:
                num_events = f['event'].shape[0]
                self.index_map.extend([(file_idx, event_idx) for event_idx in range(num_events)])

        self.total_events = len(self.index_map)

    def __len__(self):
        return self.total_events

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range.")

        file_idx, event_idx = self.index_map[idx]

        # Read the data from the appropriate file
        with h5py.File(self.files[file_idx], 'r') as f:
            positions = f['positions'][event_idx, self.position_indices]
            features = f['features'][event_idx, self.features_indices]
            fragment_truth = f['fragment_truth'][event_idx, self.fragment_truth_indices]
            interaction_truth = f['interaction_truth'][event_idx, self.interaction_truth_indices]

        data = {
            'positions': torch.tensor(positions, dtype=torch.float32),
            'features': torch.tensor(features, dtype=torch.float32),
            'batch': torch.full((positions.shape[0],), idx, dtype=torch.int64),
            'fragment_truth': torch.tensor(fragment_truth, dtype=torch.int64),
            'interaction_truth': torch.tensor(interaction_truth, dtype=torch.int64),
        }

        if self.normalized:
            data = self.normalize(data)

        return data

    def normalize(self, data):
        # Implement normalization logic if needed
        return data

    def unnormalize(self, data):
        # Implement unnormalization logic if needed
        return data

    def save_predictions(self, model_name, predictions, indices):
        # Save predictions to file if needed
        pass

    def evaluate_outputs(self, data, data_type='training'):
        """
        Here we make plots of the distributions of gut_test/gut_true before and after the autoencoder,
        as well as different plots of the latent projections, binary variables, etc.
        """
        pass
