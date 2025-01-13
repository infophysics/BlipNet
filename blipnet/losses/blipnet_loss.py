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

    def _loss(
        self,
        data
    ):
        return data
