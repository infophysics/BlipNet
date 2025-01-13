"""
Implementation of the blipnet model using pytorch
"""
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from sklearn.cluster import DBSCAN

from blipnet.models import GenericModel

blipnet_config = {
}


class BlipNet(GenericModel):
    """
    """
    def __init__(
        self,
        name:   str,
        config: dict = blipnet_config,
        meta:   dict = {}
    ):
        super(BlipNet, self).__init__(
            name, config, meta
        )
        self.config = config
        # check config
        self.logger.info(f"checking blipnet architecture using config: {self.config}")
        # construct the model
        self.forward_views = {}
        self.forward_view_map = {}
        # construct the model
        self.construct_model()
        # register hooks
        self.register_forward_hooks()

    def construct_model(self):
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.
        """
        self.logger.info(f"Attempting to build blipnet architecture using config: {self.config}")

        """Set up dbscan parameters"""
        if "dbscan_eps" not in self.config:
            self.config["dbscan_eps"] = 0.3
        self.dbscan_eps = self.config["dbscan_eps"]
        if "dbscan_min_samples" not in self.config:
            self.config["dbscan_min_samples"] = 10
        self.dbscan_min_samples = self.config["dbscan_min_samples"]
        if "dbscan_metric" not in self.config:
            self.config["dbscan_metric"] = "euclidean"
        self.dbscan_metric = self.config["dbscan_metric"]

        self.dbscan = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric=self.dbscan_metric
        )

        self.fragment_node_dict = OrderedDict()
        self.fragment_edge_dict = OrderedDict()
        self.fragment_edge_classification_dict = OrderedDict()
        self.interaction_node_dict = OrderedDict()
        self.interaction_edge_dict = OrderedDict()
        self.interaction_edge_classification_dict = OrderedDict()

    def forward(
        self,
        data
    ):
        """
        Iterate over the model dictionary
        """
        """Run dbscan over each event to generate fragment nodes/edges"""
        data = self.generate_fragments(data)

        """Generate positional feature embedding"""
        data = self.generate_position_embedding(data)

        """Generate fragment nodes/edges based on dbscan clusters"""
        data = self.generate_fragment_node_embedding(data)
        data = self.generate_fragment_edge_embedding(data)
        data = self.generate_fragment_edge_classification(data)

        """Generate interaction nodes/edges based on fragment nodes/edges"""
        data = self.generate_interaction_node_embedding(data)
        data = self.generate_interaction_edge_embedding(data)
        data = self.generate_interaction_edge_classification(data)

        return data

    def generate_fragments(
        self,
        data
    ):
        """
        DBSCAN generates fragment clusters based on input parameters.
        Those clusters define nodes and edges in fragment space which
        get appended to the dataset.
        """

        positions = data['positions']
        batch = positions.batch

        cluster_labels = torch.full((positions.size(0),), -1, dtype=torch.long)  # Initialize with -1 (noise)
        unique_batches = batch.unique()

        for b in unique_batches:
            mask = batch == b
            pos_b = positions[mask].cpu().numpy()

            # Apply DBSCAN on the batch
            labels = self.dbscan.fit_predict(pos_b)

            # Assign cluster labels to the correct batch positions
            cluster_labels[mask] = torch.tensor(labels, dtype=torch.long)

    def generate_position_embedding(
        self,
        data
    ):
        pass

    def generate_fragment_node_embedding(
        self,
        data
    ):
        pass

    def generate_fragment_edge_embedding(
        self,
        data
    ):
        pass

    def generate_fragment_edge_classification(
        self,
        data
    ):
        pass

    def generate_interaction_node_embedding(
        self,
        data
    ):
        pass

    def generate_interaction_edge_embedding(
        self,
        data
    ):
        pass

    def generate_interaction_edge_classification(
        self,
        data
    ):
        pass
