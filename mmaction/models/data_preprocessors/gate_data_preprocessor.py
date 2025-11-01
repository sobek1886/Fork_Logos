from collections import defaultdict
import torch
from typing import List, Optional, Sequence, Tuple, Union

from mmaction.registry import MODELS
from mmaction.utils import SampleList
from mmaction.models.data_preprocessors.data_preprocessor import BaseActionDataPreprocessor, stack_batch

@MODELS.register_module()
class GateActionDataPreprocessor(BaseActionDataPreprocessor):

    def __init__(self, preprocessors, **kwargs) -> None:
        super().__init__()
        if isinstance(preprocessors, (list, tuple)):
            preprocessors = {i: pr for i, pr in enumerate(preprocessors)}
        assert isinstance(preprocessors, dict)
        assert len(preprocessors)
        self.preprocessors = torch.nn.ModuleDict() # key: head  # list of dicts with "key", "dataset", "ratio" keys
        for k, pr in preprocessors.items():
            self.preprocessors[k] = MODELS.build(pr)
            
    def _calc_original_idx_to_key_idx(self, data_samples: SampleList) -> List[Tuple[Union[str, int], int]]:
        """Calculate list of (key, index within key) for data_samples 

        Args:
            data_samples (_type_): _description_

        Returns:
            original_idx_to_key_idx, key_to_original_idx (tuple): where
                original_idx_to_key_idx (list): list of pairs (key, index within the key) for original indices
                key_to_original_idx (dict): dict key: list of original indices for that key
        """
        # TODO: it is similar to GateHead. Make parent class
        key_to_original_idx = defaultdict(list)
        original_idx_to_key_idx = list()  # orig. index -> (key, index within key)
        for original_idx, s in enumerate(data_samples):
            key = s.dataset_key
            assert key in self.preprocessors.keys(), f'GateActionDataPreprocessor: no preprocessor is defined for dataset key {key}'
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
            key_start_indices = dict()
            concat_list = list()
            start_index = 0
            for k in key_to_original_idx.keys():
                if k in x.keys():
                    x_k = x[k]
                    concat_list.append(x_k)
                    key_start_indices[k] = start_index
                    start_index += len(x_k)
            original_idx_to_concat_idx = [key_start_indices[key] + idx for (key, idx) in original_idx_to_key_idx]
            concat_x = torch.cat(concat_list, dim=0)
            return concat_x[original_idx_to_concat_idx, :]
        
    def preprocess(self,
                inputs: List[torch.Tensor],
                data_samples: SampleList,
                training: bool = False) -> Tuple:
        assert isinstance(inputs, list)  # list of tensors
        assert len(inputs) == len(data_samples)
        original_idx_to_key_idx, key_to_original_idx = self._calc_original_idx_to_key_idx(data_samples)
        inputs_results = dict()
        samples_results = dict()
        for k, preprocessor in self.preprocessors.items():
            key_indices = key_to_original_idx[k]
            if not key_indices:
                continue
            inputs_subset = [inputs[i] for i in key_indices]
            samples_subset = [data_samples[i] for i in key_indices]
            inputs_res, samples_res = preprocessor.preprocess(inputs_subset, samples_subset, training)
            inputs_results[k] = inputs_res  # tensor list converted to tensor[batch, ...] here
            samples_results[k] = samples_res
        merged_inputs_results = self._merge_outputs(inputs_results, original_idx_to_key_idx, key_to_original_idx)    
        merged_samples_results = [samples_results[key][idx] for key, idx in original_idx_to_key_idx]
        return merged_inputs_results, merged_samples_results