"""
Implementation of the blipnet model using pytorch
"""
import torch
import torch.nn as nn
from collections import OrderedDict

from blipnet.models.common import activations


class EdgeEmbedding(nn.Module):
    """
    A nn.Linear layer with batch norm, activations,
    optional residual layers and dropout.
    """
    def __init__(
        self,
        config,
        device,
    ):
        super(EdgeEmbedding, self).__init__()
        self.config = config
        self.device = device
        self.construct_layer()

    def construct_layer(self):
        """Construct temporary dictionary of layers"""
        _edge_embedding = OrderedDict()

        """EdgeEmbedding type"""
        self.embedding_type = self.config["type"]

        """input and output dimensions"""
        self.input_dimension = self.config["input_dimension"]
        self.output_dimension = self.config["output_dimension"]
        self.residual_connection = self.config["residual_connection"]

        """data dictionary labels"""
        self.input_position_label = self.config["input_position_label"]
        self.input_edge_index_label = self.config["input_edge_index_label"]
        self.output_features_label = self.config["output_features_label"]

        """Residual connection"""
        if (
            self.residual_connection &
            (self.input_dimension != self.output_dimension)
        ):
            self.residual_proj = nn.Linear(self.input_dimension, self.output_dimension)
        else:
            self.residual_proj = None

        """Dropout"""
        self.dropout = self.config["dropout"]
        self.dropout_amount = self.config["dropout_amount"]

        """iterate over mlp layers to construct Linear"""
        for ii, dimension in enumerate(self.config['layers']):
            """Add Linear layer"""
            _edge_embedding[f'layer_{ii}'] = nn.Linear(
                self.input_dimension,
                dimension,
            )

            """Batch norm"""
            _edge_embedding[f'layer_{ii}_batchnorm'] = nn.BatchNorm1d(
                num_features=dimension
            )

            """Activation"""
            _edge_embedding[f'layer_{ii}_activation'] = activations[
                self.config['activation']
            ](**self.config['activation_params'])

            """Add dropout"""
            if self.dropout:
                _edge_embedding[f'layer_{ii}_dropout'] = nn.Dropout(self.dropout_amount)

            self.input_dimension = dimension

        """Save model dictionary"""
        self.edge_embedding = nn.ModuleDict(_edge_embedding)

    def forward(
        self,
        data
    ):
        """
        The nn.Linear takes the set of input features and concatenates them
        to a set of input node features.  We then apply a graph convolution
        to generate an overall set of node features.
        """

        """Grab input features"""
        if isinstance(self.input_position_label, list):
            features = torch.cat(
                [data[key].to(self.device) for key in self.input_position_label],
                dim=1
            )
        else:
            features = data[self.input_position_label].to(self.device)

        """If type is 'node_neighbors', then concatenate node features"""
        if self.embedding_type == 'node_neighbors':
            """Grab input indices"""
            source_indices, target_indices = data[
                self.input_edge_index_label
            ].to(self.device)

            """Collect neighbor node features and concatenate"""
            source_features = features[source_indices]
            target_features = features[target_indices]
            features = torch.cat([source_features, target_features], dim=1)

        """create residual"""
        if self.residual_connection:
            if self.residual_proj is None:
                residual = features.clone()
            else:
                residual = self.residual_proj(features)

        """Iterate over layers"""
        for name, layer in self.edge_embedding.items():
            features = layer(features)

        """Add residual"""
        if self.residual_connection:
            features += residual

        data[self.output_features_label] = features
        return data
