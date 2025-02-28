"""
"""
import torch.nn as nn

from blipnet.losses import GenericLoss


class FragmentEdgeLoss(GenericLoss):
    """
    """
    def __init__(
        self,
        name: str = 'fragment_edge_loss',
        alpha: float = 0.0,
        target: str = '',
        output: str = '',
        meta: dict = {}
    ):
        super(FragmentEdgeLoss, self).__init__(
            name, alpha, meta
        )
        self.target = target
        self.output = output
        self.fragment_edge_loss = nn.BCELoss(reduction='mean')

    def _loss(
        self,
        data
    ):
        fragment_loss_answer = data[self.target].to(self.device).float().unsqueeze(1)
        fragment_loss_output = data[self.output].to(self.device)
        data['fragment_edge_loss'] = self.alpha * (self.fragment_edge_loss(
            fragment_loss_output, fragment_loss_answer
        ))
        return data
