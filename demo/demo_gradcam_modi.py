# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from typing import List, Tuple, Union

from torch import Tensor
from torch.nn.parallel._functions import Scatter as OrigScatter
from typing import List, Optional, Union

import torch
from torch import Tensor
from torch.nn.parallel._functions import _get_stream

import mmcvold
import numpy as np
import torch
import mmengine
import functools
from typing import Callable, Type, Union

import numpy as np
import torch

from mmcvold import Config, DictAction
from mmcvold.parallel import collate, scatter

from mmaction.apis import init_recognizer
from mmengine.dataset import Compose
from mmaction.utils.gradcam_utils import GradCAM
import pickle
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

dataset='ntu60'
file_name = 'S001C001P001R002A058'

#from .data_container import DataContainer
def assert_tensor_type(func: Callable) -> Callable:

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError(
                f'{args[0].__class__.__name__} has no attribute '
                f'{func.__name__} for type {args[0].datatype}')
        return func(*args, **kwargs)

    return wrapper


class DataContainer:
    """A container for any type of objects.

    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array or Tensor).

    We design `DataContainer` and `MMDataParallel` to overcome these
    limitations. The behavior can be either of the following.

    - copy to GPU, pad all tensors to the same size and stack them
    - copy to GPU without stacking
    - leave the objects as is and pass it to the model
    - pad_dims specifies the number of last few dimensions to do padding
    """

    def __init__(self,
                 data: Union[torch.Tensor, np.ndarray],
                 stack: bool = False,
                 padding_value: int = 0,
                 cpu_only: bool = False,
                 pad_dims: int = 2):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.data)})'

    def __len__(self) -> int:
        return len(self._data)

    @property
    def data(self) -> Union[torch.Tensor, np.ndarray]:
        return self._data

    @property
    def datatype(self) -> Union[Type, str]:
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self) -> bool:
        return self._cpu_only

    @property
    def stack(self) -> bool:
        return self._stack

    @property
    def padding_value(self) -> int:
        return self._padding_value

    @property
    def pad_dims(self) -> int:
        return self._pad_dims

    @assert_tensor_type
    def size(self, *args, **kwargs) -> torch.Size:
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self) -> int:
        return self.data.dim()

def collate(batch: Sequence, samples_per_gpu: int = 1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """

    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], DataContainer):
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.size(-dim))
                    padded_samples = []
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim -
                                1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(
                            F.pad(
                                sample.data, pad, value=sample.padding_value))
                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in batch[i:i + samples_per_gpu]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)


ScatterInputs = Union[Tensor, DataContainer, tuple, list, dict]
def scatter(input: Union[List, Tensor],
            devices: List,
            streams: Optional[List] = None) -> Union[List, Tensor]:
    """Scatters tensor across multiple GPUs."""
    if streams is None:
        streams = [None] * len(devices)

    if isinstance(input, list):
        chunk_size = (len(input) - 1) // len(devices) + 1
        outputs = [
            scatter(input[i], [devices[i // chunk_size]],
                    [streams[i // chunk_size]]) for i in range(len(input))
        ]
        return outputs
    elif isinstance(input, Tensor):
        output = input.contiguous()
        # TODO: copy to a pinned buffer first (if copying from CPU)
        stream = streams[0] if output.numel() > 0 else None
        if devices != [-1]:
            with torch.cuda.device(devices[0]), torch.cuda.stream(stream):
                output = output.cuda(devices[0], non_blocking=True)

        return output
    else:
        raise Exception(f'Unknown type {type(input)}.')


def synchronize_stream(output: Union[List, Tensor], devices: List,
                       streams: List) -> None:
    if isinstance(output, list):
        chunk_size = len(output) // len(devices)
        for i in range(len(devices)):
            for j in range(chunk_size):
                synchronize_stream(output[i * chunk_size + j], [devices[i]],
                                   [streams[i]])
    elif isinstance(output, Tensor):
        if output.numel() != 0:
            with torch.cuda.device(devices[0]):
                main_stream = torch.cuda.current_stream()
                main_stream.wait_stream(streams[0])
                output.record_stream(main_stream)
    else:
        raise Exception(f'Unknown type {type(output)}.')


def get_input_device(input: Union[List, Tensor]) -> int:
    if isinstance(input, list):
        for item in input:
            input_device = get_input_device(item)
            if input_device != -1:
                return input_device
        return -1
    elif isinstance(input, Tensor):
        return input.get_device() if input.is_cuda else -1
    else:
        raise Exception(f'Unknown type {type(input)}.')


class Scatter:

    @staticmethod
    def forward(target_gpus: List[int], input: Union[List, Tensor]) -> tuple:
        input_device = get_input_device(input)
        streams = None
        if input_device == -1 and target_gpus != [-1]:
            # Perform CPU to GPU copies in a background stream
            streams = [_get_stream(device) for device in target_gpus]

        outputs = scatter(input, target_gpus, streams)
        # Synchronize with the copy stream
        if streams is not None:
            synchronize_stream(outputs, target_gpus, streams)

        return tuple(outputs) if isinstance(outputs, list) else (outputs, )


def scatter(inputs: ScatterInputs,
            target_gpus: List[int],
            dim: int = 0) -> list:
    """Scatter inputs to target gpus.

    The only difference from original :func:`scatter` is to add support for
    :type:`~mmcv.parallel.DataContainer`.
    """

    def scatter_map(obj):
        if isinstance(obj, Tensor):
            if target_gpus != [-1]:
                return OrigScatter.apply(target_gpus, None, dim, obj)
            else:
                # for CPU inference we use self-implemented scatter
                return Scatter.forward(target_gpus, obj)
        if isinstance(obj, DataContainer):
            if obj.cpu_only:
                return obj.data
            else:
                return Scatter.forward(target_gpus, obj.data)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            out = list(map(list, zip(*map(scatter_map, obj))))
            return out
        if isinstance(obj, dict) and len(obj) > 0:
            out = list(map(type(obj), zip(*map(scatter_map, obj.items()))))
            return out
        return [obj for _ in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None  # type: ignore


def scatter_kwargs(inputs: ScatterInputs,
                   kwargs: ScatterInputs,
                   target_gpus: List[int],
                   dim: int = 0) -> Tuple[tuple, tuple]:
    """Scatter with support for kwargs dictionary."""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        length = len(kwargs) - len(inputs)
        inputs.extend([() for _ in range(length)])  # type: ignore
    elif len(kwargs) < len(inputs):
        length = len(inputs) - len(kwargs)
        kwargs.extend([{} for _ in range(length)])  # type: ignore
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs




def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 GradCAM demo')

    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    parser.add_argument('video', help='video file/url or rawframes directory')
    parser.add_argument(
        '--use-frames',
        default=False,
        action='store_true',
        help='whether to use rawframes as input')
    parser.add_argument(
        '--device', type=str, default='cuda:1', help='CPU/CUDA device option')
    parser.add_argument(
        '--target-layer-name',
        type=str,
        default='backbone/layer4/1/relu',
        help='GradCAM target layer name')
    parser.add_argument('--out-filename', default=None, help='output filename')
    parser.add_argument('--fps', default=5, type=int)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--target-resolution',
        nargs=2,
        default=None,
        type=int,
        help='Target resolution (w, h) for resizing the frames when using a '
        'video as input. If either dimension is set to -1, the frames are '
        'resized by keeping the existing aspect ratio')
    parser.add_argument(
        '--resize-algorithm',
        default='bilinear',
        help='resize algorithm applied to generate video & gif')

    args = parser.parse_args()
    return args


def build_inputs(model, video_path, use_frames=False):
    """build inputs for GradCAM.

    Note that, building inputs for GradCAM is exactly the same as building
    inputs for Recognizer test stage. Codes from `inference_recognizer`.

    Args:
        model (nn.Module): Recognizer model.
        video_path (str): video file/url or rawframes directory.
        use_frames (bool): whether to use rawframes as input.
    Returns:
        dict: Both GradCAM inputs and Recognizer test stage inputs,
            including two keys, ``imgs`` and ``label``.
    """

    if not (osp.exists(video_path) or video_path.startswith('http')):
        raise RuntimeError(f"'{video_path}' is missing")

    if osp.isfile(video_path) and use_frames:
        raise RuntimeError(
            f"'{video_path}' is a video file, not a rawframe directory")
    if osp.isdir(video_path) and not use_frames:
        raise RuntimeError(
            f"'{video_path}' is a rawframe directory, not a video file")

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # build the data pipeline
    test_pipeline = cfg.test_dataloader.dataset.pipeline
    test_pipeline = Compose(test_pipeline)
    # prepare data
    with open(video_path,"rb") as f:
        f=pickle.load(f)
        if isinstance(f, dict) :
            flag=0
            for v in f['annotations']:
                if v['frame_dir']==file_name:
                    print('find the video!')
                    f = v
                    flag = 1
                    break
            if flag==0:
                print('cannot find the video!')
                exit(-1)

        if use_frames:
            filename_tmpl = cfg.data.test.get('filename_tmpl', 'img_{:05}.jpg')
            modality = cfg.data.test.get('modality', 'RGB')
            start_index = cfg.data.test.get('start_index', 1)
            data = dict(
                frame_dir=video_path,
                total_frames=len(os.listdir(video_path)),
                label=-1,
                start_index=start_index,
                filename_tmpl=filename_tmpl,
                modality=modality)
        else:
            start_index = cfg.test_dataloader.dataset.get('start_index', 0)
            data = dict(
                filename=video_path,
                label=-1,
                start_index=start_index,
                modality='RGB',
                total_frames=f['total_frames'],
                img_shape=f['img_shape'],
                keypoint=f['keypoint'],
                keypoint_score=f['keypoint_score'])
    data = test_pipeline(data)

    #data['img'] = data['inputs']
    #data
    data['label'] = torch.tensor([f['label']], device=device)
    data['data_samples'] = [data['data_samples']]
    data['inputs'] = [data['inputs']]

    #del data['data_samples']

    #data = collate([{'inputs':data['inputs']}], samples_per_gpu=1)
    #if next(model.parameters()).is_cuda:
        # scatter to specified GPU
       # data = scatter(data, [device])[0]

    return data


def _resize_frames(frame_list,
                   scale,
                   keep_ratio=True,
                   interpolation='bilinear'):
    """resize frames according to given scale.

    Codes are modified from `mmaction2/datasets/pipelines/augmentation.py`,
    `Resize` class.

    Args:
        frame_list (list[np.ndarray]): frames to be resized.
        scale (tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size: the image will be rescaled as large
            as possible within the scale. Otherwise, it serves as (w, h)
            of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
    Returns:
        list[np.ndarray]: Both GradCAM and Recognizer test stage inputs,
            including two keys, ``imgs`` and ``label``.
    """
    if scale is None or (scale[0] == -1 and scale[1] == -1):
        return frame_list
    scale = tuple(scale)
    max_long_edge = max(scale)
    max_short_edge = min(scale)
    if max_short_edge == -1:
        scale = (np.inf, max_long_edge)

    img_h, img_w, _ = frame_list[0].shape

    if keep_ratio:
        new_w, new_h = mmcvold.rescale_size((img_w, img_h), scale)
    else:
        new_w, new_h = scale

    frame_list = [
        mmcvold.imresize(img, (new_w, new_h), interpolation=interpolation)
        for img in frame_list
    ]

    return frame_list


def main():
    args = parse_args()

    # assign the desired device.
    device = torch.device(args.device)

    cfg = mmengine.Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    # build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(cfg, args.checkpoint, device=device)

    inputs = build_inputs(model, args.video, use_frames=args.use_frames)
    gradcam = GradCAM(model, args.target_layer_name)
    results = gradcam(inputs)
    args.out_filename = os.path.join(args.out_filename, f'{dataset}_{file_name}_{args.out_filename.split("/")[-1]}.gif')

    if args.out_filename is not None:
        try:
            from moviepy.editor import ImageSequenceClip
        except ImportError:
            raise ImportError('Please install moviepy to enable output file.')

        # frames_batches shape [B, T, H, W, 3], in RGB order
        frames_batches = (results[0] * 255.).numpy().astype(np.uint8)
        frames = frames_batches.reshape(-1, *frames_batches.shape[-3:])

        frame_list = list(frames)
        frame_list = _resize_frames(
            frame_list,
            args.target_resolution,
            interpolation=args.resize_algorithm)

        video_clips = ImageSequenceClip(frame_list, fps=args.fps)
        out_type = osp.splitext(args.out_filename)[1][1:]
        if out_type == 'gif':
            video_clips.write_gif(args.out_filename)
        else:
            video_clips.write_videofile(args.out_filename, remove_temp=True)


if __name__ == '__main__':
    main()
