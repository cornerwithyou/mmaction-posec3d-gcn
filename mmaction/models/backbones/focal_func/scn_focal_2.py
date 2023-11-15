import numpy as np
try:
    import spconv.pytorch as spconv
    from spconv.pytorch import ops
    from spconv.pytorch import SparseConv3d, SubMConv3d
except:
    import spconv
    from spconv import ops
    from spconv import SparseConv3d, SubMConv3d

from torch import nn
from functools import partial

from .focal_sparse_conv import FocalSparseConv
from .norm import build_norm_layer



class SparseSequentialBatchdict(spconv.SparseSequential):
    def __init__(self, *args, **kwargs):
        super(SparseSequentialBatchdict, self).__init__(*args, **kwargs)

    def forward(self, input, gt_boxes=None):
        loss_box_of_pts = 0
        for k, module in self._modules.items():
            if module is None:
                continue
            if isinstance(module, (FocalSparseConv,)):
                input, _loss = module(input, gt_boxes)
                loss_box_of_pts += _loss
            else:
                input = module(input)
        return input, loss_box_of_pts


def conv3x3(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


def conv1x1(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """1x1 convolution"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )

class SparseBlock_input(spconv.SparseModule):
    def __init__(
        self,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
    ):
        super(SparseBlock_input, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-5, momentum=0.1)

        self.conv1 = spconv.SubMConv3d(17,32,kernel_size=(1,7,7),stride=(1,1,1),padding=(0,3,3),bias=False,indice_key=indice_key,)
        self.bn1 = build_norm_layer(norm_cfg, 32)[1]
        self.relu = nn.ReLU(inplace=True)
        # self.pool = nn.MaxPool1d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=1, ceil_mode=False)
        # self.pool2 = nn.MaxPool1d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False)

    def forward(self, x):
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))
        # out = self.pool(out)
        # out = self.pool2(out)
        return out


class SparseBlock(spconv.SparseModule):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        kernel_size = [[1,1,1],[1,3,3],[1,1,1]],
        stride = [[1,1,1],[1,1,1],[1,1,1]],
        padding1 = None,
        padding2 = None,
        norm_cfg=None,
        downsample=False,
        indice_key=None,
    ):
        super(SparseBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-5, momentum=0.1)

        bias = norm_cfg is not None
        if padding1:
            self.conv1 = spconv.SubMConv3d(in_channels, mid_channels, kernel_size=kernel_size[0], stride=stride[0], padding=padding1,bias=False,)
        else:
            self.conv1 = spconv.SubMConv3d(in_channels, mid_channels, kernel_size=kernel_size[0], stride=stride[0], bias=False,)
        self.bn1 = build_norm_layer(norm_cfg, mid_channels)[1]
        self.relu = nn.ReLU(inplace=True)
        if padding2:
            self.conv2 = spconv.SubMConv3d(mid_channels, mid_channels, kernel_size=kernel_size[1], stride=stride[1], padding=padding2, bias=False,)
        else:
            self.conv2 = spconv.SubMConv3d(mid_channels, mid_channels, kernel_size=kernel_size[1], stride=stride[1], bias=False,)
        self.bn2 = build_norm_layer(norm_cfg, mid_channels)[1]
        self.conv3 = spconv.SubMConv3d(mid_channels, out_channels, kernel_size=kernel_size[2], stride=stride[2], bias=False,)
        self.bn3 = build_norm_layer(norm_cfg, out_channels)[1]
        if downsample:
            self.downsample = spconv.SubMConv3d(in_channels, out_channels, kernel_size=kernel_size[3], stride=stride[3], bias=False,)
            self.bn4 = build_norm_layer(norm_cfg, out_channels)[1]
        else:
            self.downsample = None
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        out = self.conv3(out)
        out = out.replace_feature(self.bn3(out.features))
        if self.downsample is not None:
            identity = self.downsample(x)
            identity = identity.replace_feature(self.bn4(identity.features))
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out


class SpMiddleResNetFHDFocal(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpMiddleResNetFHDFocal", **kwargs
    ):
        super(SpMiddleResNetFHDFocal, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        # use_img = kwargs.get('USE_IMG', False)
        use_img = False
        topk = kwargs.get('TOPK', True)
        threshold = kwargs.get('THRESHOLD', 0.5)
        skip_loss = kwargs.get('SKIP_LOSS', False)
        mask_multi = kwargs.get('MASK_MULTI', True)
        enlarge_voxel_channels = kwargs.get('ENLARGE_VOXEL_CHANNELS', -1)
        self.use_img = use_img

        if use_img:
            self.conv_focal_multimodal = FocalSparseConv(16, 16, voxel_stride=1, norm_cfg=norm_cfg, padding=1,
                                                    indice_key='spconv_focal_multimodal', skip_loss=skip_loss,
                                                    mask_multi=mask_multi, topk=topk, threshold=threshold, use_img=True)

        special_spconv_fn = partial(FocalSparseConv, skip_loss=skip_loss, enlarge_voxel_channels=enlarge_voxel_channels,
                                                    mask_multi=mask_multi, topk=topk, threshold=threshold)
        special_conv_list = kwargs.get('SPECIAL_CONV_LIST', [])

        # input: # [1600, 1200, 41]
        self.conv_input = SparseBlock_input(indice_key="input")

        self.conv1 = SparseSequentialBatchdict(
            SparseBlock(in_channels=32,mid_channels=32,out_channels=128,
                        kernel_size=[[1,1,1],[1,3,3],[1,1,1],[1,1,1]],
                        stride=[[1,1,1],[1,2,2],[1,1,1],[1,2,2]],
                        padding1=None,
                        padding2=[0,1,1],
                        downsample=True,norm_cfg=norm_cfg, indice_key=["res1","res2","res1","res1"]),
            SparseBlock(in_channels=128,mid_channels=32,out_channels=128,
                        kernel_size=[[1,1,1],[1,3,3],[1,1,1]],
                        stride=[[1,1,1],[1,1,1],[1,1,1]],
                        padding1=None,
                        padding2=[0,1,1],
                        downsample=False,norm_cfg=norm_cfg, indice_key=["res1","res2","res1"]),
            SparseBlock(in_channels=128,mid_channels=32,out_channels=128,
                        kernel_size=[[1,1,1],[1,3,3],[1,1,1]],
                        stride=[[1,1,1],[1,2,2],[1,1,1]],
                        padding1=None,
                        padding2=[0,1,1],
                        downsample=False,norm_cfg=norm_cfg, indice_key=["res1","res2","res1"]),
            SparseBlock(in_channels=128,mid_channels=32,out_channels=128,
                        kernel_size=[[1,1,1],[1,3,3],[1,1,1]],
                        stride=[[1,1,1],[1,1,1],[1,1,1]],
                        padding1=None,
                        padding2=[0,1,1],
                        downsample=False,norm_cfg=norm_cfg, indice_key=["res1","res2","res1"]),

            # special_spconv_fn(16, 16, kernel_size=3, voxel_stride=1, norm_cfg=norm_cfg, padding=1, indice_key='focal0')
            # if 1 in special_conv_list else None,
        )

        self.conv2 = SparseSequentialBatchdict(
            SparseBlock(in_channels=128,mid_channels=64,out_channels=256,
                        kernel_size=[[3,1,1],[1,3,3],[1,1,1],[1,1,1]],
                        stride=[[1,1,1],[1,2,2],[1,1,1],[1,2,2]],
                        padding1=[1,0,0],
                        padding2=[0,1,1],
                        downsample=True,norm_cfg=norm_cfg, indice_key=["res3","res4","res5","res5"]),
            SparseBlock(in_channels=256,mid_channels=64,out_channels=256,
                        kernel_size=[[3,1,1],[1,3,3],[1,1,1]],
                        stride=[[1,1,1],[1,1,1],[1,1,1]],
                        padding1=[1,0,0],
                        padding2=[0,1,1],
                        downsample=False,norm_cfg=norm_cfg, indice_key=["res3","res4","res5"]),
            SparseBlock(in_channels=256,mid_channels=64,out_channels=256,
                        kernel_size=[[3,1,1],[1,3,3],[1,1,1]],
                        stride=[[1,1,1],[1,1,1],[1,1,1]],
                        padding1=[1,0,0],
                        padding2=[0,1,1],
                        downsample=False,norm_cfg=norm_cfg, indice_key=["res3","res4","res5"]),
            SparseBlock(in_channels=256,mid_channels=64,out_channels=256,
                        kernel_size=[[3,1,1],[1,3,3],[1,1,1]],
                        stride=[[1,1,1],[1,1,1],[1,1,1]],
                        padding1=[1,0,0],
                        padding2=[0,1,1],
                        downsample=False,norm_cfg=norm_cfg, indice_key=["res3","res4","res5"]),
            SparseBlock(in_channels=256,mid_channels=64,out_channels=256,
                        kernel_size=[[3,1,1],[1,3,3],[1,1,1]],
                        stride=[[1,1,1],[1,1,1],[1,1,1]],
                        padding1=[1,0,0],
                        padding2=[0,1,1],
                        downsample=False,norm_cfg=norm_cfg, indice_key=["res3","res4","res5"]),
            SparseBlock(in_channels=256,mid_channels=64,out_channels=256,
                        kernel_size=[[3,1,1],[1,3,3],[1,1,1]],
                        stride=[[1,1,1],[1,1,1],[1,1,1]],
                        padding1=[1,0,0],
                        padding2=[0,1,1],
                        downsample=False,norm_cfg=norm_cfg, indice_key=["res3","res4","res5"]),

            # special_spconv_fn(16, 16, kernel_size=3, voxel_stride=1, norm_cfg=norm_cfg, padding=1, indice_key='focal0')
            # if 1 in special_conv_list else None,
        )

        self.conv3 = SparseSequentialBatchdict(
            SparseBlock(in_channels=256,mid_channels=128,out_channels=512,
                        kernel_size=[[3,1,1],[1,3,3],[1,1,1],[1,1,1]],
                        stride=[[1,1,1],[1,2,2],[1,1,1],[1,2,2]],
                        padding1=[1,0,0],
                        padding2=[0,1,1],
                        downsample=True,norm_cfg=norm_cfg, indice_key=["res6","res7","res8","res8"]),
            SparseBlock(in_channels=512,mid_channels=128,out_channels=512,
                        kernel_size=[[3,1,1],[1,3,3],[1,1,1]],
                        stride=[[1,1,1],[1,1,1],[1,1,1]],
                        padding1=[1,0,0],
                        padding2=[0,1,1],
                        downsample=False,norm_cfg=norm_cfg, indice_key=["res6","res7","res8"]),
            SparseBlock(in_channels=512,mid_channels=128,out_channels=512,
                        kernel_size=[[3,1,1],[1,3,3],[1,1,1]],
                        stride=[[1,1,1],[1,1,1],[1,1,1]],
                        padding1=[1,0,0],
                        padding2=[0,1,1],
                        downsample=False,norm_cfg=norm_cfg, indice_key=["res6","res7","res8"]),
        )


    def forward(self, x, batch_dict = {},fuse_func=None):

        x = spconv.SparseConvTensor.from_dense(x.permute(0,2,3,4,1))
        ret = x
        loss_box_of_pts = 0

        x = self.conv_input(ret)
        x_conv1, _loss = self.conv1(x, batch_dict)
        # print(x_conv1.dense().shape)
        loss_box_of_pts += _loss

        if self.use_img:
            x_conv1, _loss = self.conv_focal_multimodal(x_conv1, batch_dict, fuse_func)
            loss_box_of_pts = loss_box_of_pts + _loss

        x_conv2, _loss = self.conv2(x_conv1, batch_dict)
        loss_box_of_pts += _loss
        # print(x_conv2.dense().shape)
        x_conv3, _loss = self.conv3(x_conv2, batch_dict)
        loss_box_of_pts += _loss
        # print(x_conv3.dense().shape)
        x_conv4, _loss = self.conv4(x_conv3, batch_dict)
        loss_box_of_pts += _loss

        ret = x_conv4.dense()
        return ret
