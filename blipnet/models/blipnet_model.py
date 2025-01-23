"""
Implementation of the blipnet model using pytorch
"""
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from torch_geometric.nn import global_mean_pool

from blipnet.models import GenericModel
from blipnet.models.layers.node_embedding import NodeEmbedding
from blipnet.models.layers.edge_embedding import EdgeEmbedding
from blipnet.utils.utils import generate_complete_edge_list
from blipnet.utils.utils import max_purity_torch

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

        """Whether data is training or testing"""
        self.training = self.config["training"]

        """Set up dbscan parameters"""
        self.fragment_dbscan_eps = self.config['fragment_clustering']["dbscan_eps"]
        self.fragment_dbscan_min_samples = self.config['fragment_clustering']["dbscan_min_samples"]
        self.fragment_dbscan_metric = self.config['fragment_clustering']["dbscan_metric"]
        self.fragment_cluster_features = self.config['fragment_clustering']["cluster_features"]
        self.fragment_dbscan = DBSCAN(
            eps=self.fragment_dbscan_eps,
            min_samples=self.fragment_dbscan_min_samples,
            metric=self.fragment_dbscan_metric
        )

        self.interaction_dbscan_eps = self.config['interaction_clustering']["dbscan_eps"]
        self.interaction_dbscan_min_samples = self.config['interaction_clustering']["dbscan_min_samples"]
        self.interaction_dbscan_metric = self.config['interaction_clustering']["dbscan_metric"]
        self.interaction_cluster_features = self.config['interaction_clustering']["cluster_features"]
        self.interaction_dbscan = DBSCAN(
            eps=self.interaction_dbscan_eps,
            min_samples=self.interaction_dbscan_min_samples,
            metric=self.interaction_dbscan_metric
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
        self.construct_fragment_node_classifier()
        self.construct_fragment_edge_classifier()
        self.construct_interaction_node_embedding()
        self.construct_interaction_edge_embedding()
        self.construct_interaction_edge_classifier()

    def construct_position_embedding(self):
        """Construct position embedding"""
        self.position_embedding = NodeEmbedding(
            self.config["position_embedding"],
            device=self.device
        )

    def construct_fragment_node_embedding(self):
        """Construct fragment node embedding"""
        self.fragment_node_embedding = NodeEmbedding(
            self.config["fragment_node_embedding"],
            device=self.device
        )

    def construct_fragment_edge_embedding(self):
        """Construct fragment edge embedding"""
        self.fragment_edge_embedding = EdgeEmbedding(
            self.config["fragment_edge_embedding"],
            device=self.device
        )

    def construct_fragment_node_classifier(self):
        """Construct fragment node classification"""
        self.fragment_node_classifier = NodeEmbedding(
            self.config["fragment_node_classifier"],
            device=self.device
        )

    def construct_fragment_edge_classifier(self):
        """Construct fragment edge classification"""
        self.fragment_edge_classifier = EdgeEmbedding(
            self.config["fragment_edge_classifier"],
            device=self.device
        )

    def construct_interaction_node_embedding(self):
        """Construct interaction node embedding"""
        self.interaction_node_embedding = NodeEmbedding(
            self.config["interaction_node_embedding"],
            device=self.device
        )

    def construct_interaction_edge_embedding(self):
        """Construct interaction edge embedding"""
        self.interaction_edge_embedding = EdgeEmbedding(
            self.config["interaction_edge_embedding"],
            device=self.device
        )

    def construct_interaction_edge_classifier(self):
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
        data = self.generate_fragment_node_classifier(data)
        data = self.generate_fragment_edge_classifier(data)

        """Aggregate clustering"""
        data = self.fragment_aggregation(data)

        """Generate interaction nodes/edges based on fragment nodes/edges"""
        data = self.generate_interaction_node_embedding(data)
        data = self.generate_interaction_edge_embedding(data)
        data = self.generate_interaction_edge_classifier(data)
        print(data)
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

    def generate_fragment_node_classifier(
        self,
        data
    ):
        """
        node classification is done at the fragment level.
        """
        data = self.fragment_node_classifier(data)
        return data

    def generate_fragment_edge_classifier(
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
        data = self.interaction_node_embedding(data)
        return data

    def generate_interaction_edge_embedding(
        self,
        data
    ):
        data = self.interaction_edge_embedding(data)
        return data

    def generate_interaction_edge_classifier(
        self,
        data
    ):
        data = self.interaction_edge_classifier(data)
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

        """Fragment clustering features"""
        fragment_cluster_features = torch.cat(
            [data[key].to(self.device) for key in self.fragment_cluster_features],
            dim=1
        )

        """Generate empty node labels for clustering"""
        fragment_cluster_labels = torch.full(
            (fragment_cluster_features.size(0),), -1, dtype=torch.long
        )

        """Iterate over batches and cluster fragments from DBSCAN"""
        batch_offset = 0
        unique_batches = data['batch'].unique()
        for b in unique_batches:
            batch_mask = (data['batch'] == b)
            cluster_features = fragment_cluster_features[batch_mask].cpu().numpy()

            """DBSCAN on the batch"""
            labels = self.fragment_dbscan.fit_predict(cluster_features)

            """Assign node labels to correct positions and add offset"""
            fragment_cluster_labels[batch_mask] = torch.tensor(labels, dtype=torch.long)
            positive_labels = (fragment_cluster_labels > -1) & batch_mask
            fragment_cluster_labels[positive_labels] += batch_offset
            batch_offset += len(np.unique(labels))

        """Create minimum spanning tree edges and batch/label info"""
        fragment_edge_batch = []
        fragment_edge_labels = []
        fragment_edge_indices = []
        fragment_edge_truth = []
        interaction_edge_truth = []

        """Iterate over clusters and construct edge indices from MST"""
        unique_labels = fragment_cluster_labels.unique()
        for label in unique_labels:
            """Get node indices of the current cluster"""
            fragment_mask = (fragment_cluster_labels == label)
            fragment_indices = np.where(fragment_cluster_labels == label)[0]
            fragment_node_batch = data['batch'][fragment_mask][0]
            fragment_positions = positions[fragment_indices]
            if self.training:
                fragment_labels = data['fragment_truth'][fragment_mask]
                fragment_interaction_labels = data['interaction_truth'][fragment_mask]

            if label == -1:
                """Treat noise points as separate clusters"""
                for node_idx in fragment_indices:
                    fragment_edge_indices.append((node_idx, node_idx))
                    fragment_edge_batch.append(data['batch'][node_idx])
                    fragment_edge_labels.append(label)
                    if self.training:
                        fragment_edge_truth.append(1)
                        interaction_edge_truth.append(1)
            else:
                """Compute the minimum spanning tree of the cluster positions"""
                dist_matrix = distance_matrix(fragment_positions, fragment_positions)
                minimum_tree = minimum_spanning_tree(dist_matrix).toarray()

                """Collect edge indices, labels from the spanning tree"""
                for i, j in zip(*np.nonzero(minimum_tree)):
                    fragment_edge_indices.append((fragment_indices[i], fragment_indices[j]))
                    fragment_edge_batch.append(fragment_node_batch)
                    fragment_edge_labels.append(label)

                    if self.training:
                        """Assign fragment node labels"""
                        if fragment_labels[i] == fragment_labels[j]:
                            fragment_edge_truth.append(1)
                        else:
                            fragment_edge_truth.append(0)
                        if fragment_interaction_labels[i] == fragment_interaction_labels[j]:
                            interaction_edge_truth.append(1)
                        else:
                            interaction_edge_truth.append(0)

        """Generate tensors"""
        fragment_edge_batch = torch.tensor(fragment_edge_batch, dtype=torch.long).t()
        fragment_edge_labels = torch.tensor(fragment_edge_labels, dtype=torch.long).t()
        fragment_edge_indices = torch.tensor(fragment_edge_indices, dtype=torch.long).t()
        if self.training:
            fragment_edge_truth = torch.tensor(fragment_edge_truth, dtype=torch.long)
            interaction_edge_truth = torch.tensor(interaction_edge_truth, dtype=torch.long)

        data['fragment_cluster_labels'] = fragment_cluster_labels
        data['fragment_edge_batch'] = fragment_edge_batch
        data['fragment_edge_labels'] = fragment_edge_labels
        data['fragment_edge_indices'] = fragment_edge_indices
        if self.training:
            data['fragment_edge_truth'] = fragment_edge_truth
            data['interaction_edge_truth'] = interaction_edge_truth
        return data

    def fragment_aggregation(
        self,
        data,
    ):
        """
        Aggregate node and edge features to form higher-level interaction node embeddings
        using DBSCAN clustering on fragment_edge_embedding.
        """
        fragment_node_features = torch.cat(
            [data[key].to(self.device) for key in self.node_aggregation_features],
            dim=1
        )
        fragment_node_batch = data['batch'].to(self.device)

        fragment_edge_features = torch.cat(
            [data[key].to(self.device) for key in self.edge_aggregation_features],
            dim=1
        )
        fragment_edge_batch = data['fragment_edge_batch'].to(self.device)
        fragment_edge_indices = data['fragment_edge_indices'].to(self.device)

        """Interaction clustering features"""
        interaction_cluster_features = torch.cat(
            [data[key].to(self.device) for key in self.interaction_cluster_features],
            dim=1
        )

        interaction_truth = data['interaction_truth'].to(self.device)

        """New fragment and interaction embeddings"""
        fragment_embeddings = []
        interaction_batches = []
        interaction_edge_indices = []
        batch_complete_edge_lists = []
        interaction_edge_purity = []

        """Iterate over batches"""
        batch_offset = 0
        cluster_offset = 0
        unique_batches = fragment_node_batch.unique()
        for b in unique_batches:
            """Grab the current batch"""
            node_mask = (fragment_node_batch == b)
            edge_mask = (fragment_edge_batch == b)

            """Apply node features aggregation"""
            fragment_edge_features_batch = fragment_edge_features[edge_mask]
            fragment_edge_indices_batch_start = fragment_edge_indices[0][edge_mask]
            fragment_edge_indices_batch_end = fragment_edge_indices[1][edge_mask]

            """Apply edge features aggregation"""
            interaction_cluster_features_batch = interaction_cluster_features[edge_mask]

            interaction_truth_batch = interaction_truth[node_mask].clone().detach().cpu()

            """Run DBSCAN on edge features"""
            edge_embeddings_np = interaction_cluster_features_batch.clone().detach().cpu().numpy()
            cluster_labels = self.interaction_dbscan.fit_predict(edge_embeddings_np)

            """Create higher-level nodes based on clusters"""
            unique_clusters = torch.unique(torch.tensor(cluster_labels))
            cluster_embeddings = []
            cluster_edge_indices = []

            """Iterate over unique clusters in cluster feature space"""
            for cluster_id in unique_clusters:
                cluster_embedding_indices = np.where(cluster_labels == cluster_id)[0]
                if cluster_id == -1:
                    """Treat noise points as separate clusters"""
                    for cluster_index in cluster_embedding_indices:
                        """
                        The cluster_index here corresponds to the index in the batched
                        interaction_cluster_features space, which is an edge embedding.
                        Therefore, to grab the correct node features we need the
                        fragment_edge_indices which give the indices in the fragment
                        space.
                        """
                        fragment_node_index = fragment_edge_indices_batch_start[cluster_index]
                        cluster_node_features = fragment_node_features[fragment_node_index]
                        cluster_edge_features = fragment_edge_features_batch[cluster_index]
                        cluster_embedding = torch.cat([cluster_node_features, cluster_edge_features], dim=0)
                        cluster_embeddings.append(cluster_embedding)

                        """Add node to interaction edge indices"""
                        cluster_edge_indices.append([cluster_offset, fragment_node_index.item()])
                        cluster_offset += 1
                else:
                    """Now for clustered points"""
                    fragment_node_indices_start = fragment_edge_indices_batch_start[cluster_embedding_indices]
                    fragment_node_indices_end = fragment_edge_indices_batch_end[cluster_embedding_indices]
                    fragment_node_indices = torch.unique(
                        torch.cat((fragment_node_indices_start, fragment_node_indices_end)), dim=0
                    )
                    cluster_node_features = fragment_node_features[fragment_node_indices]
                    batch_tensor = torch.zeros(
                        cluster_node_features.size(0),
                        dtype=torch.long,
                        device=cluster_node_features.device
                    )
                    cluster_node_agg = self.fragment_node_pooling(cluster_node_features, batch_tensor)

                    cluster_edge_features = fragment_edge_features_batch[cluster_embedding_indices]
                    batch_tensor = torch.zeros(
                        cluster_edge_features.size(0),
                        dtype=torch.long,
                        device=cluster_edge_features.device
                    )
                    cluster_edge_agg = self.fragment_edge_pooling(cluster_edge_features, batch_tensor)

                    cluster_embedding = torch.cat([cluster_node_agg, cluster_edge_agg], dim=1)
                    cluster_embeddings.append(cluster_embedding[0])

                    """Add node to interaction edge indices"""
                    for fragment_node_index in torch.unique(fragment_node_indices):
                        cluster_edge_indices.append([cluster_offset, fragment_node_index.item()])
                    cluster_offset += 1
            """Combine cluster embeddings"""
            cluster_embeddings_batch = torch.stack(cluster_embeddings).squeeze(1)
            interaction_index_batch = torch.Tensor([
                batch_offset for _ in range(len(cluster_embeddings_batch))
            ])
            fragment_embeddings.append(cluster_embeddings_batch)
            interaction_batches.append(interaction_index_batch)
            interaction_edge_indices.extend(cluster_edge_indices)

            """Compute the minimum spanning tree of the cluster positions"""
            dist_matrix = distance_matrix(
                cluster_embeddings_batch.clone().detach().cpu(),
                cluster_embeddings_batch.clone().detach().cpu()
            )
            minimum_tree = minimum_spanning_tree(dist_matrix).toarray()
            """Collect edge indices, labels from the spanning tree"""
            batch_edges = []
            for i, j in zip(*np.nonzero(minimum_tree)):
                batch_edges.append((i + batch_offset, j + batch_offset))
                batch_complete_edge_lists.append((i + batch_offset, j + batch_offset))
            """Compute interaction edge purity"""
            for edge in batch_edges:
                src, dst = edge
                src_nodes = [
                    cluster_node_map[1]
                    for ii, cluster_node_map in enumerate(cluster_edge_indices)
                    if cluster_node_map[0] == src
                ]
                dst_nodes = [
                    cluster_node_map[1]
                    for ii, cluster_node_map in enumerate(cluster_edge_indices)
                    if cluster_node_map[0] == dst
                ]
                src = interaction_truth[src_nodes]
                dst = interaction_truth[dst_nodes]
                purity = max_purity_torch(src, dst)
                interaction_edge_purity.append(purity)

            batch_offset += cluster_offset
        """Store results in data"""
        batch_complete_edge_lists = torch.tensor(batch_complete_edge_lists, dtype=torch.long).t()
        data[self.node_aggregation_output_label] = torch.cat(fragment_embeddings, dim=0)
        data['interaction_batches'] = torch.cat(interaction_batches, dim=0)
        data['interaction_edge_indices'] = torch.tensor(interaction_edge_indices).t()  # Shape (2, N)
        data[self.edge_index_aggregation_output_label] = batch_complete_edge_lists.clone()
        data['interaction_edge_purity'] = torch.tensor(interaction_edge_purity, dtype=torch.float32)
        return data
