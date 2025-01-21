"""
Wrapper for L2 loss
"""
import torch
import torch.nn as nn

from blipnet.losses import GenericLoss


class BlipNetLoss(GenericLoss):
    """
    """
    def __init__(
        self,
        name:           str = 'blipnet_loss',
        alpha:          float = 0.0,
        meta:           dict = {}
    ):
        super(BlipNetLoss, self).__init__(
            name, alpha, meta
        )
        self.fragment_edge_loss = nn.BCELoss(reduction='mean')

    def _loss(
        self,
        data
    ):
        fragment_answer = data['fragment_answer'].to(self.device).float()
        fragment_edge_class = data['fragment_edge_class'].to(self.device)
        data['fragment_edge_loss'] = self.fragment_edge_loss(fragment_edge_class, fragment_answer)
        data['blipnet_loss'] = self.alpha * (data['fragment_edge_loss'])
        return data
