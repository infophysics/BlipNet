"""
"""
import torch.nn as nn

from blipnet.losses import GenericLoss


class FragmentNodeLoss(GenericLoss):
    """
    """
    def __init__(
        self,
        name: str = 'fragment_node_loss',
        alpha: float = 0.0,
        target: str = '',
        output: str = '',
        meta: dict = {}
    ):
        super(FragmentNodeLoss, self).__init__(
            name, alpha, meta
        )
        self.target = target
        self.output = output
        self.fragment_node_loss = nn.BCELoss(reduction='mean')

    def _loss(
        self,
        data
    ):
        fragment_node_answer = data[self.target].to(self.device).float()
        fragment_node_output = data[self.output].to(self.device)
        data['fragment_node_loss'] = self.alpha * (self.fragment_node_loss(
            fragment_node_output, fragment_node_answer
        ))
        return data
