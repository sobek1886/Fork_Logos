from abc import ABCMeta
from dataclasses import dataclass

from torch.utils.data import Dataset

from mmaction.registry import DATASETS

@dataclass
class DatasetRec:
    key: str
    dataset_cfg: dict
    ratio: float
    dataset: Dataset = None
    inner_len: int = 0
    outer_len: int = 0
    index_conversion: list = None
    
    def __init__(self, key, dataset_cfg, ratio):
        self.key = key
        self.dataset_cfg = dataset_cfg
        self.ratio = ratio
        self.dataset = DATASETS.build(self.dataset_cfg)
        self.inner_len = len(self.dataset)
        self.outer_len = int(self.inner_len * self.ratio)
        self._set_index_conversion()        

    def _set_index_conversion(self):
        self.index_conversion = None
        if self.inner_len != self.outer_len:
            self.index_conversion = list()
            remaining_len = self.outer_len
            while remaining_len >= self.inner_len:
                self.index_conversion += list(range(self.inner_len))
                remaining_len -= self.inner_len
            if remaining_len:
                if remaining_len == 1:
                    self.index_conversion += [0]
                else:
                    step = (self.inner_len - 1) // (remaining_len - 1)
                    self.index_conversion += list(range(0, self.inner_len, step))  #  0, step, 2*step ...
            assert len(self.index_conversion) == self.outer_len

    def _outer_index_to_inner_index(self, outer_index):
        if self.index_conversion:
            return self.index_conversion[outer_index]
        return outer_index

    def __len__(self):
        return self.outer_len

    def __getitem__(self, outer_index):
        inner_idx = self._outer_index_to_inner_index(outer_index)
        return self.dataset[inner_idx]
    

@DATASETS.register_module()
class MergeDataset(Dataset, metaclass=ABCMeta):
    """Dataset that merged several sub-datasets.

    Args:
        datasets: collection of sub-datasets to be merged. It can be either:
        - dict of <key>: <dataset description>
        - list of <dataset description>. In this case <key> is an index in a list

        <key> is a label that marks what dataset the data item it taken for, it is passed in a data["data_samples"].dataset_key property
        <dataset description> is either:
        - dict <dataset cfg>
        - tuple or list (<dataset cfg> [, <ratio>]), where element <ratio> is optionsl, 1.0 by default
        - dict {"dataset": <dataset cfg> [, "ratio":<ratio>] }
        
        where
        <dataset cfg> is dict with "type" key and other keys, that configures Dataset class
        <ratio> is ratio of the dataset that is used in a batch
        If ratio != 1 is specified, ratio*len(sub-dataset) items are randomly sampled from the sub-dataset.
        
    """

    def __init__(self,
                 datasets,
                 **kwargs) -> None:
        if isinstance(datasets, (list, tuple)):
            datasets = {i: ds for i, ds in enumerate(datasets)}
        assert isinstance(datasets, dict)
        assert len(datasets)
        self.datasets = list()  # list of DatasetRec
        self.cum_dataset_lenghts = list()
        for k, ds_desc in datasets.items():
            if isinstance(ds_desc, (list, tuple)):
                ratio = ds_desc[1] if len(ds_desc)>1 else 1.0
                dataset_cfg = ds_desc[0]
            elif isinstance(ds_desc, dict) and "dataset" in ds_desc.keys() and "type" not in ds_desc.keys():
                ratio = ds_desc.get("ratio", 1.0)
                dataset_cfg = ds_desc["dataset"]
            else:
                ratio = 1.0
                dataset_cfg = ds_desc
            ds_rec = DatasetRec(key=k, dataset_cfg=dataset_cfg, ratio=ratio)
            self.datasets.append(ds_rec)
            self.cum_dataset_lenghts.append((self.cum_dataset_lenghts[-1] if self.cum_dataset_lenghts else 0) + len(ds_rec))

    def __len__(self):
        return self.cum_dataset_lenghts[-1]

    def __getitem__(self, index):
        dataset_idx = 0
        start_idx = 0
        while self.cum_dataset_lenghts[dataset_idx] <= index:
            start_idx = self.cum_dataset_lenghts[dataset_idx]
            dataset_idx += 1
        ds_rec = self.datasets[dataset_idx]
        data = ds_rec[index - start_idx]
        data["data_samples"].set_field(self.datasets[dataset_idx].key, '_dataset_key')
        return data
    