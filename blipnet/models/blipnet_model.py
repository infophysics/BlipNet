"""
Implementation of the blipnet model using pytorch
"""
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from torch_geometric.nn import GCNConv
from torch_geometric.nn import EdgeConv
from torch_geometric.nn import global_mean_pool

from blipnet.models import GenericModel
from blipnet.models.layers.graph_conv import GraphConvolution
from blipnet.models.layers.edge_mlp import EdgeEmbedding
from blipnet.models.common import activations
from blipnet.utils.utils import generate_complete_edge_list

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
        if "dbscan_eps" not in self.config['clustering']:
            self.config['clustering']["dbscan_eps"] = 0.3
        self.dbscan_eps = self.config['clustering']["dbscan_eps"]
        if "dbscan_min_samples" not in self.config['clustering']:
            self.config['clustering']["dbscan_min_samples"] = 10
        self.dbscan_min_samples = self.config['clustering']["dbscan_min_samples"]
        if "dbscan_metric" not in self.config['clustering']:
            self.config['clustering']["dbscan_metric"] = "euclidean"
        self.dbscan_metric = self.config['clustering']["dbscan_metric"]

        self.dbscan = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric=self.dbscan_metric
        )

        """Set up aggregation parameters"""
        self.node_aggregation_features = self.config["aggregation"]["fragment_node_features"]
        self.edge_aggregation_features = self.config["aggregation"]["fragment_edge_features"]
        self.node_aggregation_output_label = self.config["aggregation"]["node_output_label"]
        self.edge_index_aggregation_output_label = self.config["aggregation"]["edge_index_output_label"]
        self.fragment_node_pooling = global_mean_pool
        self.fragment_edge_pooling = global_mean_pool

        """Construct layers"""
        self.construct_position_embedding()
        self.construct_fragment_node_embedding()
        self.construct_fragment_edge_embedding()
        self.construct_fragment_edge_classification()
        self.construct_interaction_node_embedding()
        self.construct_interaction_edge_embedding()
        self.construct_interaction_edge_classification()

    def construct_position_embedding(self):
        """Construct position embedding"""
        self.position_embedding = GraphConvolution(
            self.config["position_embedding"],
            device=self.device
        )

    def construct_fragment_node_embedding(self):
        """Construct fragment node embedding"""
        self.fragment_node_embedding = GraphConvolution(
            self.config["fragment_node_embedding"],
            device=self.device
        )

    def construct_fragment_edge_embedding(self):
        """Construct fragment edge embedding"""
        self.fragment_edge_embedding = EdgeEmbedding(
            self.config["fragment_edge_embedding"],
            device=self.device
        )

    def construct_fragment_edge_classification(self):
        """Construct fragment edge classification"""
        self.fragment_edge_classifier = EdgeEmbedding(
            self.config["fragment_edge_classifier"],
            device=self.device
        )

    def construct_interaction_node_embedding(self):
        """Construct interaction node embedding"""
        self.interaction_node_embedding = GraphConvolution(
            self.config["interaction_node_embedding"],
            device=self.device
        )

    def construct_interaction_edge_embedding(self):
        """Construct interaction edge embedding"""
        self.interaction_edge_embedding = EdgeEmbedding(
            self.config["interaction_edge_embedding"],
            device=self.device
        )

    def construct_interaction_edge_classification(self):
        """Construct interaction edge classification"""
        self.interaction_edge_classifier = EdgeEmbedding(
            self.config["interaction_edge_classifier"],
            device=self.device
        )

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

        """Generate empty node labels for clustering"""
        fragment_node_labels = torch.full((positions.size(0),), -1, dtype=torch.long)

        """Iterate over batches and cluster fragments from DBSCAN"""
        batch_offset = 0
        unique_batches = data['batch'].unique()
        for b in unique_batches:
            batch_mask = (data['batch'] == b)
            pos_b = positions[batch_mask].cpu().numpy()

            """DBSCAN on the batch"""
            labels = self.dbscan.fit_predict(pos_b)

            """Assign node labels to correct positions and add offset"""
            fragment_node_labels[batch_mask] = torch.tensor(labels, dtype=torch.long)
            positive_labels = (fragment_node_labels > -1) & batch_mask
            fragment_node_labels[positive_labels] += batch_offset
            batch_offset += len(np.unique(labels))

        """Create minimum spanning tree edges and batch/label info"""
        fragment_edge_batch = []
        fragment_edge_labels = []
        fragment_edge_indices = []
        fragment_node_answer = []
        interaction_node_answer = []

        """Iterate over clusters and """
        unique_labels = fragment_node_labels.unique()
        for label in unique_labels:
            """Skip noise points"""
            if label == -1:
                continue

            """Get node indices of the current cluster"""
            fragment_mask = (fragment_node_labels == label)
            fragment_indices = np.where(fragment_node_labels == label)[0]
            fragment_labels = data['fragment_truth'][fragment_mask]
            fragment_interaction_labels = data['interaction_truth'][fragment_mask]
            fragment_node_batch = data['batch'][fragment_mask][0]

            fragment_positions = positions[fragment_indices]

            """Compute the minimum spanning tree of the cluster positions"""
            dist_matrix = distance_matrix(fragment_positions, fragment_positions)
            minimum_tree = minimum_spanning_tree(dist_matrix).toarray()

            """Collect edge indices, labels from the spanning tree"""
            for i, j in zip(*np.nonzero(minimum_tree)):
                fragment_edge_indices.append((fragment_indices[i], fragment_indices[j]))
                fragment_edge_batch.append(fragment_node_batch)
                fragment_edge_labels.append(label)

                """Assign fragment node labels"""
                if fragment_labels[i] == fragment_labels[j]:
                    fragment_node_answer.append([1])
                else:
                    fragment_node_answer.append([0])
                if fragment_interaction_labels[i] == fragment_interaction_labels[j]:
                    interaction_node_answer.append([1])
                else:
                    interaction_node_answer.append([0])

        """Generate tensors"""
        fragment_edge_batch = torch.tensor(fragment_edge_batch, dtype=torch.long).t()
        fragment_edge_labels = torch.tensor(fragment_edge_labels, dtype=torch.long).t()
        fragment_edge_indices = torch.tensor(fragment_edge_indices, dtype=torch.long).t()
        fragment_node_answer = torch.tensor(fragment_node_answer, dtype=torch.long)
        interaction_node_answer = torch.tensor(interaction_node_answer, dtype=torch.long)

        data['fragment_node_labels'] = fragment_node_labels
        data['fragment_edge_batch'] = fragment_edge_batch
        data['fragment_edge_labels'] = fragment_edge_labels
        data['fragment_edge_indices'] = fragment_edge_indices
        data['fragment_node_answer'] = fragment_node_answer
        data['interaction_node_answer'] = interaction_node_answer
        return data

    def generate_position_embedding(
        self,
        data
    ):
        """
        We apply a general position embedding which takes the 3D positions
        to a larger space that we then append to the position features to be
        used with the fragment node embedding.
        """
        data = self.position_embedding(data)
        return data

    def generate_fragment_node_embedding(
        self,
        data
    ):
        """
        The fragment node embedding takes the positions, position_embedding,
        and features and concatenates them to a set of input node features.  We then apply
        a graph convolution to generate an overall set of node features.
        """
        data = self.fragment_node_embedding(data)
        return data

    def generate_fragment_edge_embedding(
        self,
        data
    ):
        """
        The edge embedding generates a set of edge features based on a graph convolution
        of each of the connected embedded node features from the previous layer.
        """
        data = self.fragment_edge_embedding(data)
        return data

    def generate_fragment_edge_classification(
        self,
        data
    ):
        """
        Edge classification is done at the fragment level which determines whether an edge
        should exist between two nodes in a fragment.
        """
        data = self.fragment_edge_classifier(data)
        return data

    def generate_interaction_node_embedding(
        self,
        data
    ):
        data = self.fragment_aggregation(data)
        data = self.interaction_node_embedding(data)
        return data

    def generate_interaction_edge_embedding(
        self,
        data
    ):
        data = self.interaction_edge_embedding(data)
        return data

    def generate_interaction_edge_classification(
        self,
        data
    ):
        data = self.interaction_edge_classifier(data)
        return data

    def fragment_aggregation(
        self,
        data
    ):
        """
        Aggregate node and edge features to form higher-level interaction node embeddings.
        """
        fragment_node_features = torch.cat(
            [data[key].to(self.device) for key in self.node_aggregation_features],
            dim=1
        )
        fragment_node_labels = data['fragment_node_labels'].to(self.device)
        fragment_node_batch = data['batch'].to(self.device)

        fragment_edge_features = torch.cat(
            [data[key].to(self.device) for key in self.edge_aggregation_features],
            dim=1
        )
        fragment_edge_labels = data['fragment_edge_labels'].to(self.device)
        fragment_edge_batch = data['fragment_edge_batch'].to(self.device)

        """New fragment and interaction embeddings"""
        fragment_embeddings = []
        interaction_batches = []

        """Iterate over batches"""
        batch_offset = 0
        unique_batches = fragment_node_batch.unique()
        for b in unique_batches:
            """Grab the current batch"""
            node_mask = (fragment_node_batch == b)
            edge_mask = (fragment_edge_batch == b)

            """Apply node features aggregation"""
            fragment_node_features_batch = fragment_node_features[node_mask]
            fragment_node_labels_batch = fragment_node_labels[node_mask]
            fragment_node_agg = self.fragment_node_pooling(
                fragment_node_features_batch,
                fragment_node_labels_batch
            )

            """Apply edge features aggregation"""
            fragment_edge_features_batch = fragment_edge_features[edge_mask]
            fragment_edge_labels_batch = fragment_edge_labels[edge_mask]
            fragment_edge_agg = self.fragment_edge_pooling(
                fragment_edge_features_batch,
                fragment_edge_labels_batch
            )

            """Combine features into pooled set"""
            fragment_embeddings_batch = torch.cat([fragment_node_agg, fragment_edge_agg], dim=1)
            interaction_index_batch = torch.Tensor([batch_offset for ii in range(len(fragment_embeddings_batch))])
            fragment_embeddings.append(fragment_embeddings_batch)
            interaction_batches.append(interaction_index_batch)
            batch_offset += 1

        data[self.node_aggregation_output_label] = torch.cat(fragment_embeddings, dim=0)
        data['interaction_batches'] = torch.cat(interaction_batches, dim=0)
        data[self.edge_index_aggregation_output_label] = generate_complete_edge_list(data['interaction_batches'])
        return data
