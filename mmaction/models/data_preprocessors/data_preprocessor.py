# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.model import BaseDataPreprocessor

from mmaction.registry import MODELS
from mmaction.utils import SampleList


def stack_batch(tensor_list: List[torch.Tensor],
                pad_size_divisor: int = 1,
                pad_value: Union[int, float] = 0) -> torch.Tensor:
    """
    Copy of mmengine.model.stack_batch, modifyed to handle (Clip*Channel*Frame*H*W) tensors.
    If tensors have different Clip dim, resultimg tensor has ( sum(Clip)*1*Channel*Frame*H*W) shape.
    
    Stack multiple tensors to form a batch and pad the tensor to the max
    shape use the right bottom padding mode in these images. If
    ``pad_size_divisor > 0``, add padding to ensure the shape of each dim is
    divisible by ``pad_size_divisor``.

    Args:
        tensor_list (List[Tensor]): A list of tensors with the same dim.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
            to ensure the shape of each dim is divisible by
            ``pad_size_divisor``. This depends on the model, and many
            models need to be divisible by 32. Defaults to 1
        pad_value (int, float): The padding value. Defaults to 0.

    Returns:
       Tensor: The n dim tensor.
    """
    assert isinstance(
        tensor_list,
        list), (f'Expected input type to be list, but got {type(tensor_list)}')
    assert tensor_list, '`tensor_list` could not be an empty list'
    assert len({
        tensor.ndim
        for tensor in tensor_list
    }) == 1, (f'Expected the dimensions of all tensors must be the same, '
              f'but got {[tensor.ndim for tensor in tensor_list]}')

    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes: torch.Tensor = torch.Tensor(
        [tensor.shape for tensor in tensor_list])
    max_sizes = torch.ceil(
        torch.max(all_sizes, dim=0)[0] / pad_size_divisor) * pad_size_divisor
    padded_sizes = max_sizes - all_sizes
    # The first dim normally means channel,  which should not be padded.
    padded_sizes[:, :2] = 0
    if padded_sizes.sum() != 0:
        # `pad` is the second arguments of `F.pad`. If pad is (1, 2, 3, 4),
        # it means that padding the last dim with 1(left) 2(right), padding the
        # penultimate dim to 3(top) 4(bottom). The order of `pad` is opposite of
        # the `padded_sizes`. Therefore, the `padded_sizes` needs to be reversed,
        # and only odd index of pad should be assigned to keep padding "right" and
        # "bottom".
        pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
        pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
        batch_tensor = []
        for idx, tensor in enumerate(tensor_list):
            batch_tensor.append(
                F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
        tensor_list = batch_tensor
    has_different_shapes = all_sizes[:,0].max() != all_sizes[:,0].min()
    if has_different_shapes:
        return torch.cat(tensor_list, dim=0).unsqueeze(1)
    return torch.stack(tensor_list)


class BaseActionDataPreprocessor(BaseDataPreprocessor):
    
    def forward(self,
                data: Union[dict, Tuple[dict]],
                training: bool = False) -> Union[dict, Tuple[dict]]:
        """Perform normalization, padding, bgr2rgb conversion and batch
        augmentation based on ``BaseDataPreprocessor``.

        Args:
            data (dict or Tuple[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict or Tuple[dict]: Data in the same format as the model input.
        """
        data = self.cast_data(data)
        if isinstance(data, dict):
            return self.forward_onesample(data, training=training)
        elif isinstance(data, (tuple, list)):
            outputs = []
            for data_sample in data:
                output = self.forward_onesample(data_sample, training=training)
                outputs.append(output)
            return tuple(outputs)
        else:
            raise TypeError(f'Unsupported data type: {type(data)}!')

    def forward_onesample(self, data, training: bool = False) -> dict:
        """Perform normalization, padding, bgr2rgb conversion and batch
        augmentation on one data sample.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
                input.
        """
        inputs, data_samples = data['inputs'], data['data_samples']
        inputs, data_samples = self.preprocess(inputs, data_samples, training)
        data['inputs'] = inputs
        data['data_samples'] = data_samples
        return data

    def preprocess(self,
                   inputs: List[torch.Tensor],
                   data_samples: SampleList,
                   training: bool = False) -> Tuple:
        """Perform normalization, padding, bgr2rgb conversion and batch
        augmentation on one data sample"""
        raise NotImplementedError


@MODELS.register_module()
class ActionDataPreprocessor(BaseActionDataPreprocessor):
    """Data pre-processor for action recognition tasks.

    Args:
        mean (Sequence[float or int], optional): The pixel mean of channels
            of images or stacked optical flow. Defaults to None.
        std (Sequence[float or int], optional): The pixel standard deviation
            of channels of images or stacked optical flow. Defaults to None.
        to_rgb (bool): Whether to convert image from BGR to RGB.
            Defaults to False.
        to_float32 (bool): Whether to convert data to float32.
            Defaults to True.
        blending (dict, optional): Config for batch blending.
            Defaults to None.
        format_shape (str): Format shape of input data.
            Defaults to ``'NCHW'``.
    """

    def __init__(self,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 to_rgb: bool = False,
                 to_float32: bool = True,
                 blending: Optional[dict] = None,
                 format_shape: str = 'NCHW') -> None:
        super().__init__()
        self.to_rgb = to_rgb
        self.to_float32 = to_float32
        self.format_shape = format_shape

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            if self.format_shape == 'NCHW':
                normalizer_shape = (-1, 1, 1)
            elif self.format_shape in ['NCTHW', 'MIX2d3d']:
                normalizer_shape = (-1, 1, 1, 1)
            else:
                raise ValueError(f'Invalid format shape: {format_shape}')

            self.register_buffer(
                'mean',
                torch.tensor(mean, dtype=torch.float32).view(normalizer_shape),
                False)
            self.register_buffer(
                'std',
                torch.tensor(std, dtype=torch.float32).view(normalizer_shape),
                False)
        else:
            self._enable_normalize = False

        if blending is not None:
            self.blending = MODELS.build(blending)
        else:
            self.blending = None

    def preprocess(self,
                   inputs: List[torch.Tensor],
                   data_samples: SampleList,
                   training: bool = False) -> Tuple:
        # --- Pad and stack --
        batch_inputs = stack_batch(inputs)

        if self.format_shape == 'MIX2d3d':
            if batch_inputs.ndim == 4:
                format_shape, view_shape = 'NCHW', (-1, 1, 1)
            else:
                format_shape, view_shape = 'NCTHW', None
        else:
            format_shape, view_shape = self.format_shape, None

        # ------ To RGB ------
        if self.to_rgb:
            if format_shape == 'NCHW':
                batch_inputs = batch_inputs[..., [2, 1, 0], :, :]
            elif format_shape == 'NCTHW':
                batch_inputs = batch_inputs[..., [2, 1, 0], :, :, :]
            else:
                raise ValueError(f'Invalid format shape: {format_shape}')

        # -- Normalization ---
        if self._enable_normalize:
            if view_shape is None:
                batch_inputs = (batch_inputs - self.mean) / self.std
            else:
                mean = self.mean.view(view_shape)
                std = self.std.view(view_shape)
                batch_inputs = (batch_inputs - mean) / std
        elif self.to_float32:
            batch_inputs = batch_inputs.to(torch.float32)

        # ----- Blending -----
        if training and self.blending is not None:
            batch_inputs, data_samples = self.blending(batch_inputs,
                                                       data_samples)

        return batch_inputs, data_samples
