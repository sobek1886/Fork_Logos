# Copyright (c) OpenMMLab. All rights reserved.
from collections import Counter
import os.path as osp
from pathlib import Path
from typing import Callable, List, Optional, Union

import h5py

from mmengine.fileio import exists, list_from_file

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset


@DATASETS.register_module()
class VideoDataset(BaseActionDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        data_prefix (dict or ConfigDict): Path to a directory where videos
            are held. Defaults to ``dict(video='')``.
        start_and_end_from_labels (bool): If True, last 2 values in labels are fragment start and end points. Overrides start_index.
            Defaults to False.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Defaults to False.
        num_classes (int, optional): Number of classes of the dataset, used in
            multi-class datasets. Defaults to None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Defaults to 0.
        modality (str): Modality of data. Support ``'RGB'``, ``'Flow'``.
            Defaults to ``'RGB'``.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        delimiter (str): Delimiter for the annotation file.
            Defaults to ``' '`` (whitespace).
        limit_samples_number_to (int): If specified, limits dataset to this number of samples for debuging
        no_event_label (int): The index of no_event gloss. Used to nullify gesture boundaries in videos with the no_event class for gesture boundary regression.
        labels_shifts (int): If specified, labels, stored in dataset are converted as label[0]+=labels_shifts.
            Only label[0] is affected. It is used to fit several datasets to common label space.
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]],
                 data_prefix: ConfigType = dict(video=''),
                 start_and_end_from_labels: bool = False,
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 start_index: int = 0,
                 modality: str = 'RGB',
                 test_mode: bool = False,
                 delimiter: str = ' ',
                 limit_samples_number_to: int = 0,
                 no_event_label = None,
                 labels_shifts: int = None,
                 **kwargs) -> None:
        self.delimiter = delimiter
        self.limit_samples_number_to = limit_samples_number_to
        self.labels_shifts = labels_shifts
        if no_event_label is not None and labels_shifts:
            no_event_label += labels_shifts
        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            start_and_end_from_labels=start_and_end_from_labels,
            multi_class=multi_class,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            test_mode=test_mode,
            no_event_label=no_event_label,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        fin = list_from_file(self.ann_file)
        if self.limit_samples_number_to > 0:
            fin = fin[:self.limit_samples_number_to]
        for line in fin:
            line_split = line.strip().split(self.delimiter)
            if len(line_split) > 1:
                filename, label = line_split[0], line_split[1:]
                label = list(map(int, label))
            # add fake label for inference datalist without label
            else:
                filename, label = line_split[0], [-1]
            if self.labels_shifts and len(label):
                label[0] += self.labels_shifts
            if self.data_prefix['video'] is not None:
                filename = osp.join(self.data_prefix['video'], filename)
            data_list.append(dict(filename=filename, label=label))
        return data_list


@DATASETS.register_module()
class Hdf5VideoDataset(BaseActionDataset):
    """Video dataset for action recognition on hdf5video file

    Dataset file format and preparation see: mmaction/datasets/transforms/hdf5video.py

    Args:
        ann_file (str): Path to the hdf5video dataset file.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        data_prefix (dict or ConfigDict): Path to a directory where videos
            are held. Defaults to ``dict(video='')``.
        start_and_end_from_labels (bool): If True, last 2 values in labels are fragment start and end points. Overrides start_index.
            Defaults to False.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Defaults to False.
        num_classes (int, optional): Number of classes of the dataset, used in
            multi-class datasets. Defaults to None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Defaults to 0.
        modality (str): Modality of data. Support ``'RGB'``, ``'Flow'``.
            Defaults to ``'RGB'``.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        limit_samples_number_to (int): If specified, limits dataset to this number of samples for debuging
        no_event_label (int): The index of no_event gloss. Used to nullify gesture boundaries in videos with the no_event class for gesture boundary regression.
        limit_samples_per_label (int): If set, limit dataset to the specified number of samples per class
        labels_shifts (int): If specified, labels, stored in dataset are converted as label[0]+=labels_shifts.
            Only label[0] is affected. It is used to fit several datasets to common label space.
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]],
                 data_prefix: ConfigType = dict(video=''),
                 start_and_end_from_labels: bool = False,
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 start_index: int = 0,
                 modality: str = 'RGB',
                 test_mode: bool = False,
                 limit_samples_number_to: int = 0,
                 no_event_label = None,
                 limit_samples_per_label: int = 0,
                 labels_shifts: int = None,                 
                 **kwargs) -> None:
        self.limit_samples_number_to = limit_samples_number_to
        self.limit_samples_per_label = limit_samples_per_label
        self.labels_shifts = labels_shifts
        if no_event_label is not None and labels_shifts:
            no_event_label += labels_shifts
        assert Path(ann_file).suffix == '.hdf5video', f'Hdf5VideoDataset requires *.hdf5video as ann_file but got {ann_file}'
        self.hdf5video_f = None
        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            start_and_end_from_labels=start_and_end_from_labels,
            multi_class=multi_class,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            test_mode=test_mode,
            no_event_label=no_event_label,
            **kwargs)

    def _load_label(self, key):
        label = self.hdf5video_f[key]['label']
        label = list(map(int, label))
        if self.labels_shifts and len(label):
            label[0] += self.labels_shifts        
        return label

    def load_data_list(self) -> List[dict]:
        """Load keys from dataset file for iteration."""
        if self.hdf5video_f is None:
            self.hdf5video_f = h5py.File(self.ann_file, 'r')
        keys = self.hdf5video_f.keys()
        if self.limit_samples_number_to > 0:
            keys = list(keys)[:self.limit_samples_number_to]

        data_list = [
            dict(filename=key,
                 label=self._load_label(key),
                 hdf5_file_name=self.ann_file,
                 )
            for key in keys
        ]
        
        if self.limit_samples_per_label:
            shrinked_list = []
            class_counter = Counter()
            for rec in data_list:
                if class_counter[rec['label'][0]] < self.limit_samples_per_label:
                    class_counter[rec['label'][0]] += 1
                    shrinked_list.append(rec)
            data_list = shrinked_list
        
        return data_list