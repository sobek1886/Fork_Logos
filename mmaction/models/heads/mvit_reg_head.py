# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union, Dict, Optional
import math
import pickle

import torch
import torch.nn.functional as F
from mmengine.model.weight_init import constant_init, trunc_normal_init
from torch import Tensor, nn

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, ForwardResults, SampleList

from .base import BaseHead
from .mvit_head import MViTHead


@MODELS.register_module()
class MViTRegHead(MViTHead):
    """Head for Multi-scale ViT for classification and boundary regression.

    A PyTorch implement of : `MViTv2: Improved Multiscale Vision Transformers
    for Classification and Detection <https://arxiv.org/abs/2112.01526>`_

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Defaults to `dict(type='CrossEntropyLoss')`.
        loss_bounds(dict): Config for building loss.
            Defaults to `dict(type='MSELoss')`.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Defaults to 0.02.
        init_scale (float): Scale factor for Initiation parameters. Default: 1.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 loss_bounds: ConfigType = dict(type='MSELoss'),
                 dropout_ratio: float = 0.5,
                 init_std: float = 0.02,
                 init_scale: float = 1.0,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, dropout_ratio, init_std, init_scale, **kwargs)
        assert loss_bounds, 'loss_bounds must be passed'
        self.loss_bounds = MODELS.build(loss_bounds)
        self.fc_bounds = nn.Linear(self.in_channels, 2)

    def loss_by_feat(self, output: torch.Tensor,
                     data_samples: SampleList) -> Dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            output (torch.Tensor): Classification scores and predictions of bounds for input samples).
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        cls_scores, bounds_preds, vis_features = output
        losses = super().loss_by_feat(cls_scores, data_samples)

        gt_boundaries = [x.gt_boundaries for x in data_samples]
        gt_boundaries = torch.stack(gt_boundaries).to(bounds_preds.device)
            
        loss_bounds = self.loss_bounds(bounds_preds, gt_boundaries)
        if isinstance(loss_bounds, dict):
            losses.update(loss_bounds)
        else:
            losses['loss_bounds'] = loss_bounds
        return losses

    def predict(self, feats: Union[torch.Tensor, Tuple[torch.Tensor]],
                data_samples: SampleList, **kwargs) -> SampleList:
        """Perform forward propagation of head and predict recognition results
        on the features of the upstream network.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
             list[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        output, _, _ = self(feats, **kwargs)
        return self.predict_by_feat(output, data_samples)
        
    def forward(self, x: Tuple[List[Tensor]], **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tuple[List[Tensor]]): The input data.

        Returns:
            Tuple: Classification scores and predictions of bounds for input samples.
        """
        x = self.pre_logits(x)
        if self.dropout is not None:
            x = self.dropout(x)
        vis_features = x
        cls_score = self.fc_cls(x)
        bounds_preds = torch.sigmoid(self.fc_bounds(x)) * 2 - 1
        return cls_score, bounds_preds, vis_features
