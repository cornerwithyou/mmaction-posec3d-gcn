import numpy as np

from mmaction.models.backbones.uniformer import conv_1x1x1
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


def conv3x3(in_planes, out_planes, stride=1,kernel_size=(1,3,3), voxel_stride=1, norm_cfg=None, padding=1, indice_key=None,**kwargs):
    return FocalSparseConv(in_planes,out_planes,kernel_size=kernel_size,voxel_stride=1,norm_cfg =norm_cfg ,indice_key=indice_key,
                           skip_loss=False, enlarge_voxel_channels=-1,
                            mask_multi=True, topk=True, threshold=0.5)

def conv311(in_planes, out_planes, stride=1,kernel_size=(3,1,1), voxel_stride=1, norm_cfg=None, padding=1, indice_key=None,**kwargs):
    return FocalSparseConv(in_planes,out_planes,kernel_size=kernel_size,voxel_stride=1,norm_cfg =norm_cfg ,indice_key=indice_key,
                           skip_loss=False, enlarge_voxel_channels=-1,
                            mask_multi=True, topk=True, threshold=0.5)

def conv1x1(in_planes, out_planes, stride=1,kernel_size=(1,1,1), voxel_stride=1, norm_cfg=None, padding=1, indice_key=None,**kwargs):
    return FocalSparseConv(in_planes,out_planes,kernel_size=kernel_size,voxel_stride=1,norm_cfg =norm_cfg ,indice_key=indice_key,
                           skip_loss=False, enlarge_voxel_channels=-1,
                            mask_multi=True, topk=True, threshold=0.5)


class SparseBasicBlock1(spconv.SparseModule):
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
        super(SparseBasicBlock1, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = conv1x1(inplanes, midplanes, stride, indice_key=indice_key + "1",norm_cfg=norm_cfg,  bias=bias)
        # self.conv1 = conv1x1(inplanes, midplanes, stride, indice_key=indice_key,norm_cfg=norm_cfg,  bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, midplanes)[1]
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(midplanes, midplanes, indice_key=indice_key + "1",norm_cfg=norm_cfg, bias=bias)
        # self.conv2 = conv3x3(midplanes, midplanes, indice_key=indice_key,norm_cfg=norm_cfg, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, midplanes)[1]
        self.conv3 = conv1x1(midplanes, planes, indice_key=indice_key + "2",norm_cfg=norm_cfg, bias=bias)
        # self.conv3 = conv1x1(midplanes, planes, indice_key=indice_key,norm_cfg=norm_cfg, bias=bias)
        self.bn3 = build_norm_layer(norm_cfg, planes)[1]
        if downsample or inplanes != planes:
            self.downsample = conv1x1(inplanes, planes, indice_key=indice_key + "1",norm_cfg=norm_cfg, bias=bias)
            # self.downsample = conv1x1(inplanes, planes, indice_key=indice_key,norm_cfg=norm_cfg, bias=bias)
            self.bn4 = build_norm_layer(norm_cfg, planes)[1]
        else:
            self.downsample = None


    def forward(self, x):
        identity = x
        # out,_ = self.conv1(x,batch_dict)
        out = self.conv1(x)
        print(out)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        # out,_ = self.conv2(out)
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        # out,_ = self.conv3(out)
        out = self.conv3(out)
        out = out.replace_feature(self.bn3(out.features))
        if self.downsample is not None:
            # identity,_ = self.downsample(x)
            identity = self.downsample(x)
            identity = identity.replace_feature(self.bn4(identity.features))
        # out = Fsp.sparse_add(out, identity)
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))
        print(out)
        return out

class SparseBasicBlock2(spconv.SparseModule):
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
        super(SparseBasicBlock2, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        # self.conv1 = conv3x3(inplanes, midplanes, stride, indice_key=indice_key + "0",norm_cfg=norm_cfg,  bias=bias)
        self.conv1 = conv311(inplanes, midplanes, stride, indice_key=indice_key+"1",norm_cfg=norm_cfg,  bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, midplanes)[1]
        self.relu = nn.ReLU()
        # self.conv2 = conv3x3(midplanes, midplanes, indice_key=indice_key + "1",norm_cfg=norm_cfg, bias=bias)
        self.conv2 = conv3x3(midplanes, midplanes, indice_key=indice_key+"2",norm_cfg=norm_cfg, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, midplanes)[1]
        # self.conv3 = conv3x3(midplanes, planes, indice_key=indice_key + "2",norm_cfg=norm_cfg, bias=bias)
        self.conv3 = conv1x1(midplanes, planes, indice_key=indice_key,norm_cfg=norm_cfg, bias=bias)
        self.bn3 = build_norm_layer(norm_cfg, planes)[1]
        if downsample or inplanes != planes:
            # self.downsample = conv3x3(inplanes, planes, indice_key=indice_key + "3",norm_cfg=norm_cfg, bias=bias)
            self.downsample = conv1x1(inplanes, planes, indice_key=indice_key,norm_cfg=norm_cfg, bias=bias)
            self.bn4 = build_norm_layer(norm_cfg, planes)[1]
        else:
            self.downsample = None


    def forward(self, x):
        identity = x
        # out,_ = self.conv1(x,batch_dict)
        out = self.conv1(x)
        print(out)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        # out,_ = self.conv2(out)
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        # out,_ = self.conv3(out)
        out = self.conv3(out)
        out = out.replace_feature(self.bn3(out.features))
        if self.downsample is not None:
            # identity,_ = self.downsample(x)
            identity = self.downsample(x)
            identity = identity.replace_feature(self.bn4(identity.features))
        # out = Fsp.sparse_add(out, identity)
        out = out.replace_feature(out.features + identity.features)
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
            norm_cfg = dict(type="BN1d", eps=1e-5, momentum=0.1)

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
        special_conv_list = [1,2]
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(num_input_features, 32, kernel_size=(1,7,7),bias=False, indice_key="res0"),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(inplace=True)
        )

        self.conv1 = SparseSequentialBatchdict(
            SparseBasicBlock1(32, 32,128 ,norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock1(128, 32,128 ,norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock1(128, 32,128 ,norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock1(128, 32,128 ,norm_cfg=norm_cfg, indice_key="res0"),

            # special_spconv_fn(128, 128, kernel_size=3, voxel_stride=1, norm_cfg=norm_cfg, padding=1, indice_key='focal0')
            # if 1 in special_conv_list else None,
        )

        self.conv2 = SparseSequentialBatchdict(
            SparseBasicBlock2(128,64, 256, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock2(256,64, 256, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock2(256,64, 256, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock2(256,64, 256, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock2(256,64, 256, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock2(256,64, 256, norm_cfg=norm_cfg, indice_key="res1"),

            # special_spconv_fn(256, 256, kernel_size=3, voxel_stride=2, norm_cfg=norm_cfg, padding=1, indice_key='focal1')
            # if 2 in special_conv_list else None,

        )

        self.conv3 = SparseSequentialBatchdict(
            SparseBasicBlock2(256,128, 512, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock2(512, 128,512, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock2(512, 128,512, norm_cfg=norm_cfg, indice_key="res3"),

            # special_spconv_fn(512, 512, kernel_size=3, voxel_stride=4, norm_cfg=norm_cfg, padding=1, indice_key='focal2')
            # if 3 in special_conv_list else None,
        )

        # self.conv4 = SparseSequentialBatchdict(
        #     spconv.SparseSequential(SparseConv3d(256, 512, 3, 2, padding=[0, 1, 1], bias=False),  # [400, 300, 11] -> [200, 150, 5]
        #     build_norm_layer(norm_cfg, 512)[1],
        #     nn.ReLU(inplace=True)),
        #     SparseBasicBlock(512, 512, norm_cfg=norm_cfg, indice_key="res3"),
        #     SparseBasicBlock(512, 512, norm_cfg=norm_cfg, indice_key="res3"),
        # )


    def forward(self, x, batch_dict = {},fuse_func=None):

        x = spconv.SparseConvTensor.from_dense(x.permute(0,2,3,4,1))
        ret = x
        loss_box_of_pts = 0

        ret = self.conv_input(ret)
        print(ret)
        ret, _loss = self.conv1(ret, batch_dict)
        # print(x_conv1.dense().shape)
        loss_box_of_pts += _loss

        if self.use_img:
            x_conv1, _loss = self.conv_focal_multimodal(x_conv1, batch_dict, fuse_func)
            loss_box_of_pts = loss_box_of_pts + _loss

        ret, _loss = self.conv2(ret, batch_dict)
        loss_box_of_pts += _loss
        # print(x_conv2.dense().shape)
        ret, _loss = self.conv3(ret, batch_dict)
        loss_box_of_pts += _loss
        # print(x_conv3.dense().shape)
        # ret, _loss = self.conv4(ret, batch_dict)
        # loss_box_of_pts += _loss

        ret = ret.dense()
        return ret
