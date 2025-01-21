"""
"""
import torch.nn as nn

from blipnet.losses import GenericLoss


class InteractionNodeLoss(GenericLoss):
    """
    """
    def __init__(
        self,
        name: str = 'interaction_node_loss',
        alpha: float = 0.0,
        target: str = '',
        output: str = '',
        meta: dict = {}
    ):
        super(InteractionNodeLoss, self).__init__(
            name, alpha, meta
        )
        self.target = target
        self.output = output
        self.interaction_node_loss = nn.BCELoss(reduction='mean')

    def _loss(
        self,
        data
    ):
        interaction_node_answer = data[self.target].to(self.device).float()
        interaction_node_output = data[self.output].to(self.device)
        data['interaction_node_loss'] = self.alpha * (self.interaction_node_loss(
            interaction_node_output, interaction_node_answer
        ))
        return data
