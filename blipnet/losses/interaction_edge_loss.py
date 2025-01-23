import torch
import torch.nn as nn
import torch.nn.functional as F
from blipnet.losses import GenericLoss


class InteractionEdgeLoss(GenericLoss):
    """
    Computes the binary cross-entropy loss for interaction edge purity prediction.
    """
    def __init__(
        self,
        name: str = 'interaction_edge_loss',
        alpha: float = 0.0,
        target: str = '',
        output: str = '',
        meta: dict = {}
    ):
        super(InteractionEdgeLoss, self).__init__(name, alpha, meta)
        self.target = target
        self.output = output
        self.bce_loss = nn.BCELoss(reduction='mean')  # Binary cross-entropy loss

    def _loss(
        self,
        data
    ):
        # Retrieve target and output tensors
        interaction_edge_purity = data[self.target].to(self.device).float()  # (N,) purity values
        interaction_edge_output = data[self.output].to(self.device).float()  # (N,) predicted purity

        # Ensure predictions are between 0 and 1 (apply sigmoid if needed)
        predicted_purity = torch.sigmoid(interaction_edge_output)

        # Compute binary cross-entropy loss
        bce_loss = self.bce_loss(predicted_purity, interaction_edge_purity.unsqueeze(1))

        # Scale the loss by alpha and store it in the data dictionary
        data['interaction_edge_loss'] = self.alpha * bce_loss
        return data