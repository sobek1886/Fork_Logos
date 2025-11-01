"""
hdf5video format support.
file format (ver. 1):
- file.attrs['version'] = '1'
- file['<key_1>']
    - file['<key_1>']['video']
        .attrs['mode'] = 'jpg_set'
        .attrs['data_size'] = data size (mp4 file size for raw_video mode, image collection size for jpg and raw modes)
        .attrs['total_frames']
        .attrs['width'] = 
        .attrs['height']
        .attrs['fps']
        data: <image list data> (see below)
    - file['<key_1>']['label']
        data: [int]
- file['<key_2>']
    ...
...

<key_i> is filename from the original .txt file with '/' replaced to '\'. 

<image list data> is byte array:
    <1st image data length (4 bytes as int32)><1st image data (<1st image data length> bytes)> ...
    ... <<total_frames'th image data length (4 bytes)><total_frames'th image data (<total_frames'th image data length> bytes)>
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import decord
import h5py
# import numba
import numpy as np
from turbojpeg import TurboJPEG, TJCS_RGB  # https://github.com/lilohuang/PyTurboJPEG

from mmcv.transforms import BaseTransform
from mmaction.registry import TRANSFORMS

def encode_jpeg(numpy_image, quality):
    numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    success, result = cv2.imencode('.jpg', numpy_image,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("Impossible to encode image in jpeg")
    return result.reshape(-1)

def resizer(image, max_resolution=None, min_resolution=None, interpolation=cv2.INTER_AREA)->np.array:
    if max_resolution is None and min_resolution is None:
        return image
    original_size = np.array([image.shape[1], image.shape[0]])
    if max_resolution is not None:
        ratio = max_resolution / original_size.max()
    elif min_resolution is not None:
        ratio = min_resolution / original_size.min()
    else:
        ratio = 1
    if ratio < 1:
        new_size = (ratio * original_size).astype(int)
        image = cv2.resize(image, tuple(new_size), interpolation=interpolation)
    return image

def extend_path(video_root_path, video_file_path):
    """
    add video_root_path to video_file_path if video_file_path is relative
    """
    if video_root_path and not Path(video_file_path).is_absolute():
        video_file_path = Path(video_root_path) / Path(video_file_path)
    return str(video_file_path)

def encode_video(
                video_full_path,
                write_mode: str = 'jpg_set',
                max_resolution: int = None,
                min_resolution: int = None,
                jpeg_quality: int = 95,
                interpolation = cv2.INTER_AREA,):
    file_obj = io.FileIO(video_full_path)
    if write_mode == 'raw_video':
        data = file_obj.readall()
        file_obj = io.BytesIO(data)
        data = np.frombuffer(data, dtype=np.uint8)
    container = decord.VideoReader(video_full_path)
    container.seek(0)
    if write_mode == 'raw_video':
        frame = container.next()
        frame_count = len(container)
        frame_size = frame.shape[:2]
    else:
        imgs = container.get_batch(list(range(len(container)))).asnumpy()  # THWC shape
        frame_count = imgs.shape[0]
        if max_resolution or min_resolution:
            imgs = [resizer(imgs[frame_idx], max_resolution, min_resolution,
                    interpolation) for frame_idx in range(frame_count)]
        frame_size = imgs[0].shape[:2]

    if write_mode == 'raw':
        if max_resolution or min_resolution:  # imgs is list, see before 
            imgs = np.stack(imgs, axis=0)
        data = imgs.reshape(-1)

    if write_mode == 'jpg_set':
        data = np.zeros( (0, ), dtype=np.uint8)
        for image in imgs:
            frame_data = encode_jpeg(image, jpeg_quality)
            sz = np.array([len(frame_data)], dtype=np.int32)
            data = np.concatenate([data, sz.view(dtype=np.uint8), frame_data])

    meta ={
        'mode': write_mode,
        'data_size': len(data),
        'total_frames': frame_count,
        'height': frame_size[0],
        'width': frame_size[1],
        'fps': container.get_avg_fps(),
    }
    return data, meta


def decode_func():
    """
    Return function to decode image list data.
    Prepared for usse with numba, but numba does not make signifficant profit, so it is disabled 
    """
    # imdecode_c = imdecode  # for FFCV decoder
    jpeg = TurboJPEG()
    my_range = range
    # imdecode_c = numba.jit(imdecode, nopython=True)
    # my_range = numba.prange

    def decode(video_data, frame_indexes, mode, frame_count, height, width):
        """
        params:
           video_data (array(1D) of uint8): image list data
           frame_indexes (array(1D) of int): list of frame indexes for resulting clip
           mode (str): mode from video_data attributes, must be 'jpg_set'
           frame_count (int): total frames count in video_data
           height, width (int, int): size of images, saved in video_data
        return:
            array((len(frame_indexes), height, width, 3)) of RGB images 
        """
        clip_len = len(frame_indexes)
        result = np.empty((clip_len, height, width, 3), dtype=np.uint8)

        if mode == 'jpg_set':
            pos = 0
            data_positions = list()  # (pos, size)
            for frame_ix in range(frame_count):
                data_size = video_data[pos:pos+4].view(np.int32)[0]
                data_positions.append( (pos + 4, data_size,))
                pos += 4 + data_size

            for target_frame_ix in my_range(clip_len):
                result_frame = result[target_frame_ix]
                src_frame_ix = frame_indexes[target_frame_ix]
                pos, data_size = data_positions[src_frame_ix]
                result_frame[:] = jpeg.decode(video_data[pos:pos + data_size], TJCS_RGB)

        else:
            raise Exception("Only 'jpg_set' mode is supported mode .hdf5video files but got {mode}")
        return result
    decode.is_parallel = True
    #decode = numba.jit(decode, nopython=True)
    return decode


class Hdf5VideoWriter:
    """
    Utility to save video in hdf5 file.
    See tools/convert/convert_to_hdf5video.py for usage
    """
    def __init__(self,
                    hdf5_file_name: Union[str, Path],
                    video_root_path: Union[str, Path],
                    max_resolution: int = None,
                    min_resolution: int = None,
                    jpeg_quality: int = 95,
                    interpolation = cv2.INTER_AREA,                 
                 ) -> None:
        self.hdf5video_file = h5py.File(hdf5_file_name, 'w')
        self.hdf5video_file.attrs['version'] = '1'
        self.video_root_path = video_root_path
        self.max_resolution = max_resolution
        self.min_resolution = min_resolution
        self.jpeg_quality = jpeg_quality
        self.interpolation = interpolation

    def save_data(self, video_filename, label, data, meta):
        key = video_filename.replace('/', '\\')  # to avoid creating groups
        group = self.hdf5video_file.create_group(key)
        ds_vid = group.create_dataset('video', data=data)
        ds_vid.attrs.update(meta)
        if not isinstance(label, (list, tuple)):
            label = [label]
        ds_lab = group.create_dataset('label', data=label)

    def save_video(self, video_filename, label, write_mode='jpg_set'): 
        video_full_path = extend_path(self.video_root_path, video_filename)
        data, meta = encode_video(
                video_full_path = video_full_path,
                write_mode=write_mode,
                max_resolution = self.max_resolution,
                min_resolution = self.min_resolution,
                jpeg_quality = self.jpeg_quality,
                interpolation = self.interpolation)        
        self.save_data(video_filename, label, data, meta)


@TRANSFORMS.register_module()
class Hdf5VideoInit(BaseTransform):
    """initialize the video_reader from hdf5video file.

    Required Keys:

        - (optional) hdf5_file_name
        - filename

    Added Keys:

        - hdf5_record
        - total_frames
        - fps

    Args:
        io_backend (str): io backend where frames are store.
            Defaults to ``'disk'``.
        num_threads (int): Number of thread to decode the video. Defaults to 1.
        kwargs (dict): Args for file client.
    """

    def __init__(self,
                 hdf5video_file: str = None,
                 **kwargs) -> None:
        self.hdf5video_file_name = hdf5video_file
        self.hdf5video_f = h5py.File(hdf5video_file, 'r') if hdf5video_file else None

    def transform(self, results: Dict) -> Dict:
        """Perform the Decord initialization.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        if self.hdf5video_f is None:
            assert results.get('hdf5_file_name'), "If you don't provide hdf5video_file as Hdf5VideoInit parameter, you'd' use Hdf5VideoDataset"
            self.hdf5video_f = h5py.File(results['hdf5_file_name'], 'r')
        key = results['filename'].replace('/', '\\')
        hdf5_record = self.hdf5video_f[key]        
        video_meta = hdf5_record['video'].attrs

        if not results.get("end_index"):
            results["end_index"] = video_meta["total_frames"] - 1
        
        results['hdf5_record'] = hdf5_record
        results['total_frames'] = video_meta['total_frames']
        results['fps'] = video_meta['fps']
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'hdf5video_file={str(self.hdf5video_file_name)})')
        return repr_str


@TRANSFORMS.register_module()
class Hdf5VideoDecode(BaseTransform):
    """Using decord to decode the video.

    Required Keys:

        - hdf5_record
        - frame_inds

    Added Keys:

        - imgs
        - original_shape
        - img_shape

    """

    def __init__(self) -> None:
        self.decode_func = decode_func()

    def transform(self, results: Dict) -> Dict:
        """Perform the Decord decoding.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        hdf5_record = results['hdf5_record']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']
        metadata = hdf5_record['video'].attrs 
        imgs = self.decode_func(np.array(hdf5_record['video']), frame_inds,
                                        metadata['mode'], metadata['total_frames'], metadata['height'], metadata['width'])
        imgs = list(imgs)
        results['hdf5_record'] = None
        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

    def __repr__(self) -> str:
        repr_str = f'{self.__class__.__name__}'
        return repr_str
