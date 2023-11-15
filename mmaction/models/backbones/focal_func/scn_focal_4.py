from tkinter import NO
import numpy as np

from mmaction.registry import TRANSFORMS
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
from spconv.pytorch import functional as Fsp
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


# def conv3x3(in_planes, out_planes, stride=1, indice_key=None, bias=True,**kwargs):
#     """3x3 convolution with padding"""
#     return spconv.SubMConv3d(
#         in_planes,
#         out_planes,
#         kernel_size=3,
#         stride=stride,
#         padding=1,
#         bias=bias,
#         indice_key=indice_key,
#     )
def conv3x3(in_planes, out_planes, stride=1,kernel_size=3, voxel_stride=1, norm_cfg=None, padding=1, indice_key=None,**kwargs):
    """3x3 convolution with padding"""
    """topk = kwargs.get('TOPK', True)
        threshold = kwargs.get('THRESHOLD', 0.5)
        skip_loss = kwargs.get('SKIP_LOSS', False)
        mask_multi = kwargs.get('MASK_MULTI', True)
        enlarge_voxel_channels = kwargs.get('ENLARGE_VOXEL_CHANNELS', -1)"""
    return FocalSparseConv(in_planes,out_planes,kernel_size=kernel_size,voxel_stride=1,norm_cfg =norm_cfg ,indice_key=indice_key,
                           skip_loss=False, enlarge_voxel_channels=-1,
                            mask_multi=True, topk=True, threshold=0.5)


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


class SparseBasicBlock(spconv.SparseModule):
    def __init__(
        self,
        inplanes,
        midplanes,
        planes,
        stride=1,
        norm_cfg=None,
        downsample=False,
        indice_key=None,
    ):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = conv3x3(inplanes, midplanes, stride, indice_key=indice_key + "0",norm_cfg=norm_cfg,  bias=bias)
        # self.bn1 = build_norm_layer(norm_cfg, midplanes)[1]
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(midplanes, midplanes, indice_key=indice_key + "1",norm_cfg=norm_cfg, bias=bias)
        # self.bn2 = build_norm_layer(norm_cfg, midplanes)[1]
        self.conv3 = conv3x3(midplanes, planes, indice_key=indice_key + "2",norm_cfg=norm_cfg, bias=bias)
        # self.bn3 = build_norm_layer(norm_cfg, planes)[1]
        if downsample:
            self.downsample = conv3x3(inplanes, planes, indice_key=indice_key + "3",norm_cfg=norm_cfg, bias=bias)
            # self.bn4 = build_norm_layer(norm_cfg, planes)[1]
        else:
            self.downsample = None


    def forward(self, x,batch_dict = {}):
        identity = x
        out,_ = self.conv1(x,batch_dict)
        # out = self.conv1(x,batch_dict)
        # out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out,_ = self.conv2(out,batch_dict)
        # out = self.conv2(out,batch_dict)
        # out = out.replace_feature(self.bn2(out.features))

        out,_ = self.conv3(out,batch_dict)
        # out = self.conv3(out,batch_dict)
        # out = out.replace_feature(self.bn3(out.features))
        if self.downsample is not None:
            identity,_ = self.downsample(x,batch_dict)
            # identity = self.downsample(x,batch_dict)
            # identity = identity.replace_feature(self.bn4(identity.features))
        print(out)
        print(identity)
        out = Fsp.sparse_add(out, identity)
        # out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))
        print(out)
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
        # special_conv_list = kwargs.get('SPECIAL_CONV_LIST', [])
        special_conv_list = [1,2,3]
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(num_input_features, 32, 3, bias=False, indice_key="res0"),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(inplace=True)
        )

        self.conv1 = SparseSequentialBatchdict(
            SparseBasicBlock(32, 32,128, norm_cfg=norm_cfg,downsample=True, indice_key="res0"),
            # SparseBasicBlock(128, 32,128, norm_cfg=norm_cfg, indice_key="res1"),
            # SparseBasicBlock(128, 32,128, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(128, 32,512, norm_cfg=norm_cfg,downsample=True,  indice_key="res3"),
        )

        self.conv2 = SparseSequentialBatchdict(
            # SparseBasicBlock(128, 64,256, norm_cfg=norm_cfg,downsample=True, indice_key="res1"),
            # SparseBasicBlock(256, 64,256 ,norm_cfg=norm_cfg, indice_key="res1"),
            # SparseBasicBlock(256, 64,256 ,norm_cfg=norm_cfg, indice_key="res1"),
            # SparseBasicBlock(256, 64,256 ,norm_cfg=norm_cfg, indice_key="res1"),
            # SparseBasicBlock(256, 64,256 ,norm_cfg=norm_cfg, indice_key="res1"),
            # SparseBasicBlock(256, 64,256 ,norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv3 = SparseSequentialBatchdict(
            # SparseBasicBlock(256, 128,512, norm_cfg=norm_cfg,downsample=True, indice_key="res2"),
            # SparseBasicBlock(512,128, 512, norm_cfg=norm_cfg, indice_key="res2"),
            # SparseBasicBlock(512,128, 512, norm_cfg=norm_cfg, indice_key="res2"),
        )

        # self.conv4 = SparseSequentialBatchdict(
        #     SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res1"),
        #     SparseBasicBlock(512, 512, norm_cfg=norm_cfg, indice_key="res3"),
        #     SparseBasicBlock(512, 512, norm_cfg=norm_cfg, indice_key="res3"),
        # )


    def forward(self, x, batch_dict = {},fuse_func=None):

        x = spconv.SparseConvTensor.from_dense(x.permute(0,2,3,4,1))
        ret = x
        loss_box_of_pts = 0

        x = self.conv_input(ret)
        x, _loss = self.conv1(x, batch_dict)
        # print(x_conv1.dense().shape)
        loss_box_of_pts += _loss

        x, _loss = self.conv2(x, batch_dict)
        loss_box_of_pts += _loss
        # print(x_conv2.dense().shape)
        x, _loss = self.conv3(x, batch_dict)
        loss_box_of_pts += _loss
        # print(x_conv3.dense().shape)
        # x, _loss = self.conv4(x, batch_dict)
        # loss_box_of_pts += _loss

        ret = x.dense()
        return ret
