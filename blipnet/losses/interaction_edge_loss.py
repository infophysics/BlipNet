"""
"""
import torch.nn as nn

from blipnet.losses import GenericLoss


class InteractionEdgeLoss(GenericLoss):
    """
    """
    def __init__(
        self,
        name: str = 'interaction_edge_loss',
        alpha: float = 0.0,
        target: str = '',
        output: str = '',
        meta: dict = {}
    ):
        super(InteractionEdgeLoss, self).__init__(
            name, alpha, meta
        )
        self.target = target
        self.output = output
        self.interaction_edge_loss = nn.BCELoss(reduction='mean')

    def _loss(
        self,
        data
    ):
        interaction_edge_answer = data[self.target].to(self.device).float()
        interaction_edge_output = data[self.output].to(self.device)
        data['interaction_edge_loss'] = self.alpha * (self.interaction_edge_loss(
            interaction_edge_output, interaction_edge_answer
        ))
        return data
