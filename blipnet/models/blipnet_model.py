"""
Implementation of the blipnet model using pytorch
"""
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from blipnet.models import GenericModel

blipnet_config = {
}


class BlipNet(GenericModel):
    """
    """
    def __init__(
        self,
        name:   str,
        config: dict = blipnet_config,
        meta:   dict = {}
    ):
        super(BlipNet, self).__init__(
            name, config, meta
        )
        self.config = config
        # check config
        self.logger.info(f"checking blipnet architecture using config: {self.config}")
        # construct the model
        self.forward_views = {}
        self.forward_view_map = {}
        # construct the model
        self.construct_model()
        # register hooks
        self.register_forward_hooks()

    def construct_model(self):
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.
        """
        self.logger.info(f"Attempting to build blipnet architecture using config: {self.config}")

        _dict = OrderedDict()

    def forward(
        self,
        data
    ):
        """
        Iterate over the model dictionary
        """

        return data
