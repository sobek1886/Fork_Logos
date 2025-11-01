# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from mmaction.registry import MODELS
from .base import BaseWeightedLoss

@MODELS.register_module()
class MSELoss(BaseWeightedLoss):
    """MSELoss.

    Args:
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)

    def _forward(self,
                pred,
                target):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        loss = F.mse_loss(pred, target)
        return loss
