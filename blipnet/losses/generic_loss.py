"""
Generic losses for blipnet.
"""
import torch

from blipnet.utils.logger import Logger


class GenericLoss:
    """
    Abstract base class for blipnet losses.  The inputs are
        1. name - a unique name for the loss function.
        2. alpha - a coefficient (should be from 0.0-1.0) for the strength of the influence of the loss.
        3. meta - meta information from the module.
    """
    def __init__(
        self,
        name:           str = 'generic_loss',
        alpha:          float = 0.0,
        meta:           dict = {}
    ):
        self.name = name
        self.logger = Logger(self.name, output="both", file_mode="w")
        self.alpha = alpha
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']

        # construct batch loss dictionaries
        self.batch_loss = torch.empty(size=(0, 1), dtype=torch.float, device=self.device)

    def reset_batch(self):
        self.batch_loss = torch.empty(size=(0, 1), dtype=torch.float, device=self.device)

    def set_device(
        self,
        device
    ):
        self.device = device
        self.batch_loss = torch.empty(size=(0, 1), dtype=torch.float, device=self.device)

    def _loss(
        self,
        data,
    ):
        self.logger.error('"_loss" not implemented in Loss!')
