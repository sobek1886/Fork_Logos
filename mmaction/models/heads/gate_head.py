# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import defaultdict
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from mmengine.model import BaseModule

from mmaction.registry import MODELS
from mmaction.utils import ForwardResults, SampleList, ActionDataSample, LabelData

@MODELS.register_module()
class GateHead(BaseModule, metaclass=ABCMeta):
    """Head built over set of heads that are used depending on data sample dataset_key attribute
    """

    def __init__(self, heads,
                 **kwargs) -> None:
        """
        heads: dict {dataset key: head config} or list of head configs. Must confirm with MergeDataset keys.
        """
        super(GateHead, self).__init__()
        if isinstance(heads, (list, tuple)):
            heads = {i: hd for i, hd in enumerate(heads)}
        assert isinstance(heads, dict)
        assert len(heads)
        self.heads = torch.nn.ModuleDict() # key: head  # list of dicts with "key", "dataset", "ratio" keys
        for k, head_desc in heads.items():
            self.heads[k] = MODELS.build(head_desc)

    def init_weights(self) -> None:
        for head in self.heads.values():
            head.init_weights()

    def forward(self, x, **kwargs) -> ForwardResults:
        raise NotImplementedError  # call to  loss( ) or predict( ) is expected

    def _calc_original_idx_to_key_idx(self, data_samples: SampleList) -> List[Tuple[Union[str, int], int]]:
        """Calculate list of (key, index within key) for data_samples 

        Args:
            data_samples (_type_): _description_

        Returns:
            original_idx_to_key_idx, key_to_original_idx (tuple): where
                original_idx_to_key_idx (list): list of pairs (key, index within the key) for original indices
                key_to_original_idx (dict): dict key: list of original indices for that key
        """
        key_to_original_idx = defaultdict(list)
        original_idx_to_key_idx = list()  # orig. index -> (key, index within key)
        for original_idx, s in enumerate(data_samples):
            key = s.dataset_key
            assert key in self.heads.keys(), f'GateHead: no head is defined for dataset key {key}'
            key_idx_list = key_to_original_idx[key]
            key_idx = len(key_idx_list)
            original_idx_to_key_idx.append((key, key_idx,))
            key_idx_list.append(original_idx)
        return  original_idx_to_key_idx, key_to_original_idx    

    def _extract(self, x, indices: List[int]):
        """Extract sub-batch from x using list of indicces
        """
        if isinstance(x, torch.Tensor):
            return x[indices, ...]
        else:
            assert isinstance(x, (list, tuple))
            return [self._extract(x_i, indices) for x_i in x]
        
    def _merge_outputs(self, x, original_idx_to_key_idx, key_to_original_idx):
        """Gather results from several sub-heads output"""
        template_x = next(iter(x.values()))
        if isinstance(template_x, dict):
            return {
                kx: self._merge_outputs( {k: x_k[kx] for k, x_k in x.items()}, original_idx_to_key_idx, key_to_original_idx)
                for kx in template_x.keys()
            }
        elif isinstance(template_x, (list, tuple)):
            return [
                self._merge_outputs( {k: x_k[i] for k, x_k in x.items()}, original_idx_to_key_idx, key_to_original_idx)
                for i in range(len(template_x))
            ]
        else:
            assert isinstance(template_x, torch.Tensor)
            assert not template_x.size()  # loss is scalar
            res = 0
            for i, (key, key_indices) in enumerate(key_to_original_idx.items()):
                if key_indices:
                    weight = len(key_indices)/len(original_idx_to_key_idx)
                    res = res + weight*x[key]
            return res


    def loss(self, feats: Union[torch.Tensor, Tuple[torch.Tensor]],
             data_samples: SampleList, **kwargs) -> Dict:
        """Runs appropriate heads for data depending on data_sample.dataset_key.
        Merges losses from heads as:
        - sum if loss is scalar
        - gather over batch dimension if 1st dimension of tensor is > 1

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        original_idx_to_key_idx, key_to_original_idx = self._calc_original_idx_to_key_idx(data_samples)
        results = dict()
        for k, head in self.heads.items():
            key_indices = key_to_original_idx[k]
            if not key_indices:
                continue
            feats_subset = self._extract(feats, key_indices)
            samples_subset = [data_samples[i] for i in key_indices]
            res = head.loss(feats_subset, samples_subset, **kwargs)
            results[k] = res

        results = self._merge_outputs(results, original_idx_to_key_idx, key_to_original_idx)
        return results


    def predict(self, feats: Union[torch.Tensor, Tuple[torch.Tensor]],
                data_samples: SampleList, **kwargs) -> SampleList:
        """Runs appropriate heads for data depending on data_sample.dataset_key
        Merges results into one list

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
             list[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        original_idx_to_key_idx, key_to_original_idx = self._calc_original_idx_to_key_idx(data_samples)
        results = dict()  # key: list of results for k, indexed inside key
        for k, head in self.heads.items():
            key_indices = key_to_original_idx[k]
            if not key_indices:
                continue
            res = head.predict(self._extract(feats, key_indices), [data_samples[i] for i in key_indices], **kwargs)
            results[k] = res
        result = [results[key][idx] for key, idx in original_idx_to_key_idx]
        return result
