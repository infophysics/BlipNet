"""
Implementation of the blipnet model using pytorch
"""
import torch
import torch.nn as nn
from collections import OrderedDict
from torch_geometric.nn import GCNConv

from blipnet.models.common import activations


class NodeEmbedding(nn.Module):
    """
    A GCNConv layer with batch norm, activations,
    optional residual layers and dropout.
    """
    def __init__(
        self,
        config,
        device,
    ):
        super(NodeEmbedding, self).__init__()
        self.config = config
        self.device = device
        self.construct_layer()

    def construct_layer(self):
        """Construct temporary dictionary of layers"""
        _graph_convolution = OrderedDict()

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

        """iterate over convolution layers to construct gcnconv"""
        for ii, dimension in enumerate(self.config['layers']):
            """Add GCNConv layer"""
            _graph_convolution[f'layer_{ii}'] = GCNConv(
                in_channels=self.input_dimension,
                out_channels=dimension,
            )

            """Batch norm"""
            _graph_convolution[f'layer_{ii}_batchnorm'] = nn.BatchNorm1d(
                num_features=dimension
            )

            """Activation"""
            _graph_convolution[f'layer_{ii}_activation'] = activations[
                self.config['activation']
            ](**self.config['activation_params'])

            """Add dropout"""
            if self.dropout:
                _graph_convolution[f'layer_{ii}_dropout'] = nn.Dropout(self.dropout_amount)

            self.input_dimension = dimension

        """Save model dictionary"""
        self.graph_convolution = nn.ModuleDict(_graph_convolution)

    def forward(
        self,
        data
    ):
        """
        The GCNConv takes the set of input features and concatenates them
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

        """create residual"""
        if self.residual_connection:
            if self.residual_proj is None:
                residual = features.clone()
            else:
                residual = self.residual_proj(features)

        """Grab input indices"""
        edge_indices = data[self.input_edge_index_label].to(self.device)
        for name, layer in self.graph_convolution.items():
            if ('batchnorm' in name) or ('activation' in name) or ('dropout' in name):
                features = layer(features)
            else:
                features = layer(features, edge_indices)

        """Add residual"""
        if self.residual_connection:
            features += residual

        data[self.output_features_label] = features
        return data
