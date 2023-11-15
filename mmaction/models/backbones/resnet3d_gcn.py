# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import copy
from mmcv.cnn import ConvModule, NonLocal3d, build_activation_layer
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, Sequential
from mmengine.model.weight_init import constant_init, kaiming_init
from mmengine.runner.checkpoint import _load_checkpoint, load_checkpoint
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from torch.nn.modules.utils import _ntuple, _triple

from mmaction.registry import MODELS
import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList

from mmaction.registry import MODELS
from ..utils import Graph, unit_aagcn, unit_tcn

class AAGCNBlock(BaseModule):
    """The basic block of AAGCN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        A (torch.Tensor): The adjacency matrix defined in the graph
            with shape of `(num_subsets, num_nodes, num_nodes)`.
        stride (int): Stride of the temporal convolution. Defaults to 1.
        residual (bool): Whether to use residual connection. Defaults to True.
        init_cfg (dict or list[dict], optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A: torch.Tensor,
                 stride: int = 1,
                 residual: bool = True,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {
            k: v
            for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']
        }
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_aagcn')
        assert gcn_type in ['unit_aagcn']

        self.gcn = unit_aagcn(in_channels, out_channels, A, **gcn_kwargs)

        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(
                out_channels, out_channels, 9, stride=stride, **tcn_kwargs)

        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(
                in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


@MODELS.register_module()
class AAGCN_fusion(BaseModule):
    """AAGCN backbone, the attention-enhanced version of 2s-AGCN.

    Skeleton-Based Action Recognition with Multi-Stream
    Adaptive Graph Convolutional Networks.
    More details can be found in the `paper
    <https://arxiv.org/abs/1912.06971>`__ .

    Two-Stream Adaptive Graph Convolutional Networks for
    Skeleton-Based Action Recognition.
    More details can be found in the `paper
    <https://arxiv.org/abs/1805.07694>`__ .

    Args:
        graph_cfg (dict): Config for building the graph.
        in_channels (int): Number of input channels. Defaults to 3.
        base_channels (int): Number of base channels. Defaults to 64.
        data_bn_type (str): Type of the data bn layer. Defaults to ``'MVC'``.
        num_person (int): Maximum number of people. Only used when
            data_bn_type == 'MVC'. Defaults to 2.
        num_stages (int): Total number of stages. Defaults to 10.
        inflate_stages (list[int]): Stages to inflate the number of channels.
            Defaults to ``[5, 8]``.
        down_stages (list[int]): Stages to perform downsampling in
            the time dimension. Defaults to ``[5, 8]``.
        init_cfg (dict or list[dict], optional): Config to control
            the initialization. Defaults to None.

        Examples:
        >>> import torch
        >>> from mmaction.models import AAGCN
        >>> from mmaction.utils import register_all_modules
        >>>
        >>> register_all_modules()
        >>> mode = 'stgcn_spatial'
        >>> batch_size, num_person, num_frames = 2, 2, 150
        >>>
        >>> # openpose-18 layout
        >>> num_joints = 18
        >>> model = AAGCN(graph_cfg=dict(layout='openpose', mode=mode))
        >>> model.init_weights()
        >>> inputs = torch.randn(batch_size, num_person,
        ...                      num_frames, num_joints, 3)
        >>> output = model(inputs)
        >>> print(output.shape)
        >>>
        >>> # nturgb+d layout
        >>> num_joints = 25
        >>> model = AAGCN(graph_cfg=dict(layout='nturgb+d', mode=mode))
        >>> model.init_weights()
        >>> inputs = torch.randn(batch_size, num_person,
        ...                      num_frames, num_joints, 3)
        >>> output = model(inputs)
        >>> print(output.shape)
        >>>
        >>> # coco layout
        >>> num_joints = 17
        >>> model = AAGCN(graph_cfg=dict(layout='coco', mode=mode))
        >>> model.init_weights()
        >>> inputs = torch.randn(batch_size, num_person,
        ...                      num_frames, num_joints, 3)
        >>> output = model(inputs)
        >>> print(output.shape)
        >>>
        >>> # custom settings
        >>> # disable the attention module to degenerate AAGCN to AGCN
        >>> model = AAGCN(graph_cfg=dict(layout='coco', mode=mode),
        ...               gcn_attention=False)
        >>> model.init_weights()
        >>> output = model(inputs)
        >>> print(output.shape)
        torch.Size([2, 2, 256, 38, 18])
        torch.Size([2, 2, 256, 38, 25])
        torch.Size([2, 2, 256, 38, 17])
        torch.Size([2, 2, 256, 38, 17])
    """

    def __init__(self,
                 graph_cfg: Dict,
                 in_channels: int = 3,
                 base_channels: int = 64,
                 data_bn_type: str = 'MVC',
                 num_person: int = 2,
                 num_stages: int = 10,
                 inflate_stages: List[int] = [5, 8],
                 down_stages: List[int] = [5, 8],
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(
            self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        assert data_bn_type in ['MVC', 'VC', None]
        self.data_bn_type = data_bn_type
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_person = num_person
        self.num_stages = num_stages
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        if self.data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif self.data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [copy.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        modules = []
        if self.in_channels != self.base_channels:
            modules = [
                AAGCNBlock(
                    in_channels,
                    base_channels,
                    A.clone(),
                    1,
                    residual=False,
                    **lw_kwargs[0])
            ]

        for i in range(2, num_stages + 1):
            in_channels = base_channels
            out_channels = base_channels * (1 + (i in inflate_stages))
            stride = 1 + (i in down_stages)
            modules.append(
                AAGCNBlock(
                    base_channels,
                    out_channels,
                    A.clone(),
                    stride=stride,
                    **lw_kwargs[i - 1]))
            base_channels = out_channels

        if self.in_channels == self.base_channels:
            self.num_stages -= 1

        self.gcn = ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))

        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4,
                                          2).contiguous().view(N * M, C, T, V)

        for i in range(self.num_stages):
            x = self.gcn[i](x)

        x = x.reshape((N, M) + x.shape[1:])
        return x

class BasicBlock3d(BaseModule):
    """BasicBlock 3d block for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        spatial_stride (int): Spatial stride in the conv3d layer.
            Defaults to 1.
        temporal_stride (int): Temporal stride in the conv3d layer.
            Defaults to 1.
        dilation (int): Spacing between kernel elements. Defaults to 1.
        downsample (nn.Module or None): Downsample layer. Defaults to None.
        style (str): 'pytorch' or 'caffe'. If set to 'pytorch', the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Defaults to ``'pytorch'``.
        inflate (bool): Whether to inflate kernel. Defaults to True.
        non_local (bool): Determine whether to apply non-local module in this
            block. Defaults to False.
        non_local_cfg (dict): Config for non-local module.
            Defaults to ``dict()``.
        conv_cfg (dict): Config dict for convolution layer.
            Defaults to ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers.
            Required keys are ``type``. Defaults to ``dict(type='BN3d')``.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='ReLU')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """
    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 spatial_stride: int = 1,
                 temporal_stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 style: str = 'pytorch',
                 inflate: bool = True,
                 non_local: bool = False,
                 non_local_cfg: Dict = dict(),
                 conv_cfg: Dict = dict(type='Conv3d'),
                 norm_cfg: Dict = dict(type='BN3d'),
                 act_cfg: Dict = dict(type='ReLU'),
                 with_cp: bool = False,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        assert style in ['pytorch', 'caffe']
        # make sure that only ``inflate_style`` is passed into kwargs
        assert set(kwargs).issubset(['inflate_style'])

        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.style = style
        self.inflate = inflate
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.non_local = non_local
        self.non_local_cfg = non_local_cfg

        self.conv1_stride_s = spatial_stride
        self.conv2_stride_s = 1
        self.conv1_stride_t = temporal_stride
        self.conv2_stride_t = 1

        if self.inflate:
            conv1_kernel_size = (3, 3, 3)
            conv1_padding = (1, dilation, dilation)
            conv2_kernel_size = (3, 3, 3)
            conv2_padding = (1, 1, 1)
        else:
            conv1_kernel_size = (1, 3, 3)
            conv1_padding = (0, dilation, dilation)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, 1, 1)

        self.conv1 = ConvModule(
            inplanes,
            planes,
            conv1_kernel_size,
            stride=(self.conv1_stride_t, self.conv1_stride_s,
                    self.conv1_stride_s),
            padding=conv1_padding,
            dilation=(1, dilation, dilation),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv2 = ConvModule(
            planes,
            planes * self.expansion,
            conv2_kernel_size,
            stride=(self.conv2_stride_t, self.conv2_stride_s,
                    self.conv2_stride_s),
            padding=conv2_padding,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.downsample = downsample
        self.relu = build_activation_layer(self.act_cfg)

        if self.non_local:
            self.non_local_block = NonLocal3d(self.conv2.norm.num_features,
                                              **self.non_local_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x

            out = self.conv1(x)
            out = self.conv2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)

        if self.non_local:
            out = self.non_local_block(out)

        return out


class Bottleneck3d(BaseModule):
    """Bottleneck 3d block for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        spatial_stride (int): Spatial stride in the conv3d layer.
            Defaults to 1.
        temporal_stride (int): Temporal stride in the conv3d layer.
            Defaults to 1.
        dilation (int): Spacing between kernel elements. Defaults to 1.
        downsample (nn.Module, optional): Downsample layer. Defaults to None.
        style (str): 'pytorch' or 'caffe'. If set to 'pytorch', the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Defaults to ``'pytorch'``.
        inflate (bool): Whether to inflate kernel. Defaults to True.
        inflate_style (str): '3x1x1' or '3x3x3'. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Defaults to ``'3x1x1'``.
        non_local (bool): Determine whether to apply non-local module in this
            block. Defaults to False.
        non_local_cfg (dict): Config for non-local module.
            Defaults to ``dict()``.
        conv_cfg (dict): Config dict for convolution layer.
            Defaults to ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers. required
            keys are ``type``. Defaults to ``dict(type='BN3d')``.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='ReLU')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """
    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 spatial_stride: int = 1,
                 temporal_stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 style: str = 'pytorch',
                 inflate: bool = True,
                 inflate_style: str = '3x1x1',
                 non_local: bool = False,
                 non_local_cfg: Dict = dict(),
                 conv_cfg: Dict = dict(type='Conv3d'),
                 norm_cfg: Dict = dict(type='BN3d'),
                 act_cfg: Dict = dict(type='ReLU'),
                 with_cp: bool = False,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert style in ['pytorch', 'caffe']
        assert inflate_style in ['3x1x1', '3x3x3']
        # conv_cfg['type'] = "sp"
        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.style = style
        self.inflate = inflate
        self.inflate_style = inflate_style
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.non_local = non_local
        self.non_local_cfg = non_local_cfg

        if self.style == 'pytorch':
            self.conv1_stride_s = 1
            self.conv2_stride_s = spatial_stride
            self.conv1_stride_t = 1
            self.conv2_stride_t = temporal_stride
        else:
            self.conv1_stride_s = spatial_stride
            self.conv2_stride_s = 1
            self.conv1_stride_t = temporal_stride
            self.conv2_stride_t = 1

        if self.inflate:
            if inflate_style == '3x1x1':
                conv1_kernel_size = (3, 1, 1)
                conv1_padding = (1, 0, 0)
                conv2_kernel_size = (1, 3, 3)
                conv2_padding = (0, dilation, dilation)
            else:
                conv1_kernel_size = (1, 1, 1)
                conv1_padding = (0, 0, 0)
                conv2_kernel_size = (3, 3, 3)
                conv2_padding = (1, dilation, dilation)
        else:
            conv1_kernel_size = (1, 1, 1)
            conv1_padding = (0, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, dilation, dilation)
        self.conv1 = ConvModule(
            inplanes,
            planes,
            conv1_kernel_size,
            stride=(self.conv1_stride_t, self.conv1_stride_s,
                    self.conv1_stride_s),
            padding=conv1_padding,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv2 = ConvModule(
            planes,
            planes,
            conv2_kernel_size,
            stride=(self.conv2_stride_t, self.conv2_stride_s,
                    self.conv2_stride_s),
            padding=conv2_padding,
            dilation=(1, dilation, dilation),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv3 = ConvModule(
            planes,
            planes * self.expansion,
            1,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            # No activation in the third ConvModule for bottleneck
            act_cfg=None)

        self.downsample = downsample
        self.relu = build_activation_layer(self.act_cfg)

        if self.non_local:
            self.non_local_block = NonLocal3d(self.conv3.norm.num_features,
                                              **self.non_local_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x

            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)

        if self.non_local:
            out = self.non_local_block(out)

        return out


@MODELS.register_module()
class ResNet3d_gcn(BaseModule):
    """ResNet 3d backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
            Defaults to 50.
        pretrained (str, optional): Name of pretrained model. Defaults to None.
        stage_blocks (tuple, optional): Set number of stages for each res
            layer. Defaults to None.
        pretrained2d (bool): Whether to load pretrained 2D model.
            Defaults to True.
        in_channels (int): Channel num of input features. Defaults to 3.
        num_stages (int): Resnet stages. Defaults to 4.
        base_channels (int): Channel num of stem output features.
            Defaults to 64.
        out_indices (Sequence[int]): Indices of output feature.
            Defaults to ``(3, )``.
        spatial_strides (Sequence[int]):
            Spatial strides of residual blocks of each stage.
            Defaults to ``(1, 2, 2, 2)``.
        temporal_strides (Sequence[int]):
            Temporal strides of residual blocks of each stage.
            Defaults to ``(1, 1, 1, 1)``.
        dilations (Sequence[int]): Dilation of each stage.
            Defaults to ``(1, 1, 1, 1)``.
        conv1_kernel (Sequence[int]): Kernel size of the first conv layer.
            Defaults to ``(3, 7, 7)``.
        conv1_stride_s (int): Spatial stride of the first conv layer.
            Defaults to 2.
        conv1_stride_t (int): Temporal stride of the first conv layer.
            Defaults to 1.
        pool1_stride_s (int): Spatial stride of the first pooling layer.
            Defaults to 2.
        pool1_stride_t (int): Temporal stride of the first pooling layer.
            Defaults to 1.
        with_pool2 (bool): Whether to use pool2. Defaults to True.
        style (str): 'pytorch' or 'caffe'. If set to 'pytorch', the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Defaults to ``'pytorch'``.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Defaults to -1.
        inflate (Sequence[int]): Inflate Dims of each block.
            Defaults to ``(1, 1, 1, 1)``.
        inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Defaults to ``3x1x1``.
        conv_cfg (dict): Config for conv layers.
            Required keys are ``type``. Defaults to ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers.
            Required keys are ``type`` and ``requires_grad``.
            Defaults to ``dict(type='BN3d', requires_grad=True)``.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='ReLU', inplace=True)``.
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (``mean`` and ``var``). Defaults to False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        non_local (Sequence[int]): Determine whether to apply non-local module
            in the corresponding block of each stages.
            Defaults to ``(0, 0, 0, 0)``.
        non_local_cfg (dict): Config for non-local module.
            Defaults to ``dict()``.
        zero_init_residual (bool):
            Whether to use zero initialization for residual block,
            Defaults to True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    arch_settings = {
        18: (BasicBlock3d, (2, 2, 2, 2)),
        34: (BasicBlock3d, (3, 4, 6, 3)),
        50: (Bottleneck3d, (3, 4, 6, 3)),
        101: (Bottleneck3d, (3, 4, 23, 3)),
        152: (Bottleneck3d, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth: int = 50,
                 pretrained: Optional[str] = None,
                 stage_blocks: Optional[Tuple] = None,
                 pretrained2d: bool = True,
                 in_channels: int = 3,
                 num_stages: int = 4,
                 base_channels: int = 64,
                 out_indices: Sequence[int] = (3, ),
                 spatial_strides: Sequence[int] = (1, 2, 2, 2),
                 temporal_strides: Sequence[int] = (1, 1, 1, 1),
                 dilations: Sequence[int] = (1, 1, 1, 1),
                 conv1_kernel: Sequence[int] = (3, 7, 7),
                 conv1_stride_s: int = 2,
                 conv1_stride_t: int = 1,
                 pool1_stride_s: int = 2,
                 pool1_stride_t: int = 1,
                 with_pool1: bool = True,
                 with_pool2: bool = True,
                 style: str = 'pytorch',
                 frozen_stages: int = -1,
                 inflate: Sequence[int] = (1, 1, 1, 1),
                 inflate_style: str = '3x1x1',
                 conv_cfg: Dict = dict(type='Conv3d'),
                 norm_cfg: Dict = dict(type='BN3d', requires_grad=True),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 norm_eval: bool = False,
                 with_cp: bool = False,
                 non_local: Sequence[int] = (0, 0, 0, 0),
                 non_local_cfg: Dict = dict(),
                 zero_init_residual: bool = True,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 graph_cfg=dict(layout='coco', mode='spatial'),
                 in_channels_gcn=3,
                 base_channels_gcn=64,
                 data_bn_type='MVC',
                 num_person=2,
                 num_stages_gcn=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 gcn_attention=False,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        pretrained = '/work/gyz_Projects/mmaction222/mmaction2/checkpoints/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_nobackbone.pth'
        self.depth = depth
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.stage_blocks = stage_blocks
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(
            dilations) == num_stages
        if self.stage_blocks is not None:
            assert len(self.stage_blocks) == num_stages

        self.conv1_kernel = conv1_kernel
        self.conv1_stride_s = conv1_stride_s
        self.conv1_stride_t = conv1_stride_t
        self.pool1_stride_s = pool1_stride_s
        self.pool1_stride_t = pool1_stride_t
        self.with_pool1 = with_pool1
        self.with_pool2 = with_pool2
        self.style = style
        self.frozen_stages = frozen_stages
        self.stage_inflations = _ntuple(num_stages)(inflate)
        self.non_local_stages = _ntuple(num_stages)(non_local)
        self.inflate_style = inflate_style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        self.block, stage_blocks = self.arch_settings[depth]

        if self.stage_blocks is None:
            self.stage_blocks = stage_blocks[:num_stages]

        self.inplanes = self.base_channels

        self.non_local_cfg = non_local_cfg

        self._make_stem_layer()

        self.res_layers = []
        lateral_inplanes = getattr(self, 'lateral_inplanes', [0, 0, 0, 0])

        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = self.base_channels * 2**i
            res_layer = self.make_res_layer(
                self.block,
                self.inplanes + lateral_inplanes[i],
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                style=self.style,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
                act_cfg=self.act_cfg,
                non_local=self.non_local_stages[i],
                non_local_cfg=self.non_local_cfg,
                inflate=self.stage_inflations[i],
                inflate_style=self.inflate_style,
                with_cp=with_cp,
                **kwargs)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.gcn = AAGCN_fusion(graph_cfg=graph_cfg,
                                                 in_channels = in_channels_gcn,
                                                 base_channels= base_channels_gcn,
                                                 data_bn_type = data_bn_type,
                                                 num_person = num_person,
                                                 num_stages = num_stages_gcn,
                                                 inflate_stages = inflate_stages,
                                                 down_stages= down_stages,
                                                 gcn_attention=gcn_attention)

        self.feat_dim = self.block.expansion * \
            self.base_channels * 2 ** (len(self.stage_blocks) - 1)

    @staticmethod
    def make_res_layer(block: nn.Module,
                       inplanes: int,
                       planes: int,
                       blocks: int,
                       spatial_stride: Union[int, Sequence[int]] = 1,
                       temporal_stride: Union[int, Sequence[int]] = 1,
                       dilation: int = 1,
                       style: str = 'pytorch',
                       inflate: Union[int, Sequence[int]] = 1,
                       inflate_style: str = '3x1x1',
                       non_local: Union[int, Sequence[int]] = 0,
                       non_local_cfg: Dict = dict(),
                       norm_cfg: Optional[Dict] = None,
                       act_cfg: Optional[Dict] = None,
                       conv_cfg: Optional[Dict] = None,
                       with_cp: bool = False,
                       **kwargs) -> nn.Module:
        """Build residual layer for ResNet3D.

        Args:
            block (nn.Module): Residual module to be built.
            inplanes (int): Number of channels for the input feature
                in each block.
            planes (int): Number of channels for the output feature
                in each block.
            blocks (int): Number of residual blocks.
            spatial_stride (int | Sequence[int]): Spatial strides in
                residual and conv layers. Defaults to 1.
            temporal_stride (int | Sequence[int]): Temporal strides in
                residual and conv layers. Defaults to 1.
            dilation (int): Spacing between kernel elements. Defaults to 1.
            style (str): 'pytorch' or 'caffe'. If set to 'pytorch', the
                stride-two layer is the 3x3 conv layer,otherwise the
                stride-two layer is the first 1x1 conv layer.
                Defaults to ``'pytorch'``.
            inflate (int | Sequence[int]): Determine whether to inflate
                for each block. Defaults to 1.
            inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines
                the kernel sizes and padding strides for conv1 and conv2
                in each block. Default: ``'3x1x1'``.
            non_local (int | Sequence[int]): Determine whether to apply
                non-local module in the corresponding block of each stages.
                Defaults to 0.
            non_local_cfg (dict): Config for non-local module.
                Defaults to ``dict()``.
            conv_cfg (dict, optional): Config for conv layers.
                Defaults to None.
            norm_cfg (dict, optional): Config for norm layers.
                Defaults to None.
            act_cfg (dict, optional): Config for activate layers.
                Defaults to None.
            with_cp (bool, optional): Use checkpoint or not. Using checkpoint
                will save some memory while slowing down the training speed.
                Defaults to False.

        Returns:
            nn.Module: A residual layer for the given config.
        """
        inflate = inflate if not isinstance(inflate, int) \
            else (inflate,) * blocks
        non_local = non_local if not isinstance(non_local, int) \
            else (non_local,) * blocks
        assert len(inflate) == blocks and len(non_local) == blocks
        downsample = None
        if spatial_stride != 1 or inplanes != planes * block.expansion:
            downsample = ConvModule(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                downsample=downsample,
                style=style,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                non_local=(non_local[0] == 1),
                non_local_cfg=non_local_cfg,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp,
                **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    spatial_stride=1,
                    temporal_stride=1,
                    dilation=dilation,
                    style=style,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    non_local=(non_local[i] == 1),
                    non_local_cfg=non_local_cfg,
                    norm_cfg=norm_cfg,
                    conv_cfg=conv_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp,
                    **kwargs))

        return Sequential(*layers)

    @staticmethod
    def _inflate_conv_params(conv3d: nn.Module, state_dict_2d: OrderedDict,
                             module_name_2d: str,
                             inflated_param_names: List[str]) -> None:
        """Inflate a conv module from 2d to 3d.

        Args:
            conv3d (nn.Module): The destination conv3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding conv module in the
                2d model.
            inflated_param_names (list[str]): List of parameters that have been
                inflated.
        """
        weight_2d_name = module_name_2d + '.weight'

        conv2d_weight = state_dict_2d[weight_2d_name]
        kernel_t = conv3d.weight.data.shape[2]

        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(
            conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)

        if getattr(conv3d, 'bias') is not None:
            bias_2d_name = module_name_2d + '.bias'
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)

    @staticmethod
    def _inflate_bn_params(bn3d: nn.Module, state_dict_2d: OrderedDict,
                           module_name_2d: str,
                           inflated_param_names: List[str]) -> None:
        """Inflate a norm module from 2d to 3d.

        Args:
            bn3d (nn.Module): The destination bn3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding bn module in the
                2d model.
            inflated_param_names (list[str]): List of parameters that have been
                inflated.
        """
        for param_name, param in bn3d.named_parameters():
            param_2d_name = f'{module_name_2d}.{param_name}'
            param_2d = state_dict_2d[param_2d_name]
            if param.data.shape != param_2d.shape:
                warnings.warn(f'The parameter of {module_name_2d} is not'
                              'loaded due to incompatible shapes. ')
                return

            param.data.copy_(param_2d)
            inflated_param_names.append(param_2d_name)

        for param_name, param in bn3d.named_buffers():
            param_2d_name = f'{module_name_2d}.{param_name}'
            # some buffers like num_batches_tracked may not exist in old
            # checkpoints
            if param_2d_name in state_dict_2d:
                param_2d = state_dict_2d[param_2d_name]
                param.data.copy_(param_2d)
                inflated_param_names.append(param_2d_name)

    @staticmethod
    def _inflate_weights(self, logger: MMLogger) -> None:
        """Inflate the resnet2d parameters to resnet3d.

        The differences between resnet3d and resnet2d mainly lie in an extra
        axis of conv kernel. To utilize the pretrained parameters in 2d model,
        the weight of conv2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (MMLogger): The logger used to print
                debugging information.
        """

        state_dict_r2d = _load_checkpoint(self.pretrained, map_location='cpu')
        if 'state_dict' in state_dict_r2d:
            state_dict_r2d = state_dict_r2d['state_dict']

        inflated_param_names = []
        for name, module in self.named_modules():
            if isinstance(module, ConvModule):
                # we use a ConvModule to wrap conv+bn+relu layers, thus the
                # name mapping is needed
                if 'downsample' in name:
                    # layer{X}.{Y}.downsample.conv->layer{X}.{Y}.downsample.0
                    original_conv_name = name + '.0'
                    # layer{X}.{Y}.downsample.bn->layer{X}.{Y}.downsample.1
                    original_bn_name = name + '.1'
                else:
                    # layer{X}.{Y}.conv{n}.conv->layer{X}.{Y}.conv{n}
                    original_conv_name = name
                    # layer{X}.{Y}.conv{n}.bn->layer{X}.{Y}.bn{n}
                    original_bn_name = name.replace('conv', 'bn')
                if original_conv_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d'
                                   f': {original_conv_name}')
                else:
                    shape_2d = state_dict_r2d[original_conv_name +
                                              '.weight'].shape
                    shape_3d = module.conv.weight.data.shape
                    if shape_2d != shape_3d[:2] + shape_3d[3:]:
                        logger.warning(f'Weight shape mismatch for '
                                       f': {original_conv_name} : '
                                       f'3d weight shape: {shape_3d}; '
                                       f'2d weight shape: {shape_2d}. ')
                    else:
                        self._inflate_conv_params(module.conv, state_dict_r2d,
                                                  original_conv_name,
                                                  inflated_param_names)

                if original_bn_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d'
                                   f': {original_bn_name}')
                else:
                    self._inflate_bn_params(module.bn, state_dict_r2d,
                                            original_bn_name,
                                            inflated_param_names)

        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(
            state_dict_r2d.keys()) - set(inflated_param_names)
        if remaining_names:
            logger.info(f'These parameters in the 2d checkpoint are not loaded'
                        f': {remaining_names}')

    def inflate_weights(self, logger: MMLogger) -> None:
        """Inflate weights."""
        self._inflate_weights(self, logger)

    def _make_stem_layer(self) -> None:
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        self.conv1 = ConvModule(
            self.in_channels,
            self.base_channels,
            kernel_size=self.conv1_kernel,
            stride=(self.conv1_stride_t, self.conv1_stride_s,
                    self.conv1_stride_s),
            padding=tuple([(k - 1) // 2 for k in _triple(self.conv1_kernel)]),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(self.pool1_stride_t, self.pool1_stride_s,
                    self.pool1_stride_s),
            padding=(0, 1, 1))

        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def _freeze_stages(self) -> None:
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    @staticmethod
    def _init_weights(self, pretrained: Optional[str] = None) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch.

        Args:
            pretrained (str | None): The path of the pretrained weight. Will
                override the original `pretrained` if set. The arg is added to
                be compatible with mmdet. Defaults to None.
        """
        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights(logger)
            else:
                # Directly load 3D model.
                load_checkpoint(
                    self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck3d):
                        constant_init(m.conv3.bn, 0)
                    elif isinstance(m, BasicBlock3d):
                        constant_init(m.conv2.bn, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def init_weights(self, pretrained: Optional[str] = None) -> None:
        """Initialize weights."""
        self._init_weights(self, pretrained)

    def forward(self, x: tuple) \
            -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Defines the computation performed at every call.

        Args:
            x_pose (torch.Tensor): The input data.

        Returns:
            torch.Tensor or tuple[torch.Tensor]: The feature of the input
            samples extracted by the backbone.
        """
        x_pose,x_gcn = x
        outs_gcn = self.gcn(x_gcn)
        x_pose = self.conv1(x_pose)
        if self.with_pool1:
            x_pose = self.maxpool(x_pose)
        outs_pose = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x_pose = res_layer(x_pose)
            if i == 0 and self.with_pool2:
                x_pose = self.pool2(x_pose)
            if i in self.out_indices:
                outs_pose.append(x_pose)
        if len(outs_pose) == 1:
            return tuple([outs_pose[0],outs_gcn])

        return tuple(outs_pose)

    def train(self, mode: bool = True) -> None:
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


