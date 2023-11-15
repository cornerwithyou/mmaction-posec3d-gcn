from multiprocessing import pool
import numpy as np

from mmaction.models.backbones.uniformer import conv_1x1x1
try:
    import spconv.pytorch as spconv
    from spconv.pytorch import ops
    from spconv.pytorch import SparseConv3d, SubMConv3d, SparseMaxPool3d
    from spconv.pytorch import functional as Fsp
except:
    import spconv
    from spconv import ops
    from spconv import SparseConv3d, SubMConv3d

from torch import nn
from functools import partial

from .focal_sparse_conv_all import FocalSparseConv
from .norm import build_norm_layer
import time


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

def conv3x3(in_planes, out_planes,stride=(1,1,1), indice_key=None,norm_cfg=None ,  bias=True,**kwargs):
    """3x3 convolution with padding"""
    # return nn.Conv3d(in_planes,out_planes,kernel_size=(1,3,3),stride=stride,padding=(0, 1, 1),bias=bias,
    #                        )
    # return spconv.SparseConv3d(in_planes,out_planes,kernel_size=(1,3,3),stride=stride,padding=1,bias=bias,indice_key=indice_key,
    #                        )
    return spconv.SubMConv3d(in_planes,out_planes,kernel_size=(1,3,3),stride=stride,padding=1,bias=bias,indice_key=indice_key,
                           )

def conv311(in_planes, out_planes,stride=(1,1,1), indice_key=None,norm_cfg=None ,  bias=True,**kwargs):

    return nn.Conv3d(in_planes,out_planes,kernel_size=(3,1,1),stride=stride,padding=(1, 0, 0),bias=bias,
                           )
    # return spconv.SubMConv3d(in_planes,out_planes,kernel_size=(3,1,1),stride=stride,padding=1,bias=bias,indice_key=indice_key,
    #                        )

def conv1x1(in_planes, out_planes,stride=(1,1,1), indice_key=None,norm_cfg=None ,  bias=True,**kwargs):
    """1x1 convolution"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1,1,1),
        stride=stride,
        # padding=1,
        bias=bias,
    )


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
        visualize=False,
    ):
        super(SparseBasicBlock1, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None
        self.visualize = visualize
        self.conv1 = conv1x1(inplanes, midplanes, stride, indice_key=indice_key + "1",norm_cfg=norm_cfg,  bias=bias)
        # self.conv1 = conv1x1(inplanes, midplanes, stride, indice_key=indice_key,norm_cfg=norm_cfg,  bias=bias)
        self.bn1 = nn.BatchNorm3d(midplanes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.conv2 = conv3x3(midplanes, midplanes, indice_key=indice_key + "3",norm_cfg=norm_cfg, bias=bias)
        # self.conv2 = conv3x3(midplanes, midplanes, indice_key=indice_key,norm_cfg=norm_cfg, bias=bias)
        self.bn2 = nn.BatchNorm3d(midplanes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = conv1x1(midplanes, planes, indice_key=indice_key + "1",norm_cfg=norm_cfg, bias=bias)
        # self.conv3 = conv1x1(midplanes, planes, indice_key=indice_key,norm_cfg=norm_cfg, bias=bias)
        self.bn3 = nn.BatchNorm3d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if inplanes != planes:
            self.downsample = conv1x1(inplanes, planes, indice_key=indice_key + "1",norm_cfg=norm_cfg, bias=bias)
            # self.downsample = SubMConv3d(inplanes, planes,(1,1,1),stride=(1, 2, 2), indice_key=indice_key,bias=bias)
            self.bn4 = nn.BatchNorm3d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        # out,_ = self.conv1(x,batch_dict)
        out = self.conv1(x)
        # print(f"conv1={out.shape}")
        out = self.bn1(out)
        out = self.relu1(out)

        # out,_ = self.conv2(out)

        out = spconv.SparseConvTensor.from_dense(out.permute(0,2,3,4,1))

        out = self.conv2(out)
        out = out.dense()
        # print(f"conv2={out.shape}")
        # print(f"conv2={out.dense().shape}")
        out = self.bn2(out)

        # out,_ = self.conv3(out)
        out = self.conv3(out)
        # print(f"conv3={out.shape}")
        out = self.bn3(out)
        if self.downsample is not None:
            # identity,_ = self.downsample(x)
            identity = self.downsample(x)
            # print(f"downsample={identity.shape}")
            identity = self.bn4(identity)
        # out = Fsp.sparse_add(out, identity)
        out = out + identity
        if self.visualize:
            import pickle
            with open('bn3_out.pkl', 'wb') as f:
                pickle.dump(out, f)
        out = self.relu2(out)

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
        visualize=False,
    ):
        super(SparseBasicBlock2, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None
        self.visualize = visualize
        # self.conv1 = conv3x3(inplanes, midplanes, stride, indice_key=indice_key + "0",norm_cfg=norm_cfg,  bias=bias)
        self.conv1 = conv311(inplanes, midplanes, stride, indice_key=indice_key+"1",norm_cfg=norm_cfg,  bias=bias)
        self.bn1 = nn.BatchNorm3d(midplanes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        # self.conv2 = conv3x3(midplanes, midplanes, indice_key=indice_key + "1",norm_cfg=norm_cfg, bias=bias)
        self.conv2 = conv3x3(midplanes, midplanes, indice_key=indice_key+"2",norm_cfg=norm_cfg, bias=bias)
        self.bn2 = nn.BatchNorm3d(midplanes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.conv3 = conv3x3(midplanes, planes, indice_key=indice_key + "2",norm_cfg=norm_cfg, bias=bias)
        self.conv3 = conv1x1(midplanes, planes, indice_key=indice_key + "3",norm_cfg=norm_cfg, bias=bias)
        self.bn3 = nn.BatchNorm3d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if inplanes != planes:
            self.downsample = conv1x1(inplanes, planes, indice_key=indice_key + "3",norm_cfg=norm_cfg, bias=bias)
            # self.downsample = SubMConv3d(inplanes, planes,(1,1,1), indice_key=indice_key,bias=bias)
            self.bn4 = nn.BatchNorm3d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        # out,_ = self.conv1(x,batch_dict)
        # out = spconv.SparseConvTensor.from_dense(out.permute(0,2,3,4,1))
        out = self.conv1(x)
        # out = out.dense()
        # print(f"conv1={out.shape}")
        out = self.bn1(out)
        out = self.relu1(out)
        out = spconv.SparseConvTensor.from_dense(out.permute(0,2,3,4,1))
        out = self.conv2(out)
        out = out.dense()
        # print(f"conv2={out.shape}")
        out = self.bn2(out)

        # out,_ = self.conv3(out)
        out = self.conv3(out)
        # print(f"conv3={out.shape}")
        out = self.bn3(out)
        if self.downsample is not None:
            # identity,_ = self.downsample(x)
            identity = self.downsample(identity)
            # print(f"downsample={identity.shape}")
            identity = self.bn4(identity)
        # out = Fsp.sparse_add(out, identity)
        out = out + identity
        if self.visualize:
            import pickle
            with open('bn3_out.pkl', 'wb') as f:
                pickle.dump(out, f)
        out = self.relu2(out)
        return out

class ConvWithSub(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpMiddleResNetFHDFocal", **kwargs
    ):
        super(ConvWithSub, self).__init__()
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

        self.conv_input = nn.Sequential(
            nn.Conv3d(17, 32, kernel_size=(1, 7, 7), stride=(1, 1, 1), padding=(0, 3, 3), bias=False),
            # special_spconv_fn(num_input_features, 32, kernel_size=(1,7,7),voxel_stride=1, indice_key="res0"),
            nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=1, ceil_mode=False)
        )

        self.layer1 = SparseSequentialBatchdict(
            SparseBasicBlock1(32, 32,128 ,norm_cfg=norm_cfg, indice_key="layer0"),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0, dilation=1, ceil_mode=False),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=1, ceil_mode=False),
            SparseBasicBlock1(128, 32,128 ,norm_cfg=norm_cfg, indice_key="layer01"),
            SparseBasicBlock1(128, 32,128 ,norm_cfg=norm_cfg, indice_key="layer01"),
            SparseBasicBlock1(128, 32,128 ,norm_cfg=norm_cfg, indice_key="layer01"),
        )

        self.layer2 = SparseSequentialBatchdict(
            SparseBasicBlock2(128,64, 256, norm_cfg=norm_cfg, indice_key="layer1"),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0, dilation=1, ceil_mode=False),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=1, ceil_mode=False),
            SparseBasicBlock2(256,64, 256, norm_cfg=norm_cfg, indice_key="layer11"),
            SparseBasicBlock2(256,64, 256, norm_cfg=norm_cfg, indice_key="layer11"),
            SparseBasicBlock2(256,64, 256, norm_cfg=norm_cfg, indice_key="layer11"),
            SparseBasicBlock2(256,64, 256, norm_cfg=norm_cfg, indice_key="layer11"),
            SparseBasicBlock2(256,64, 256, norm_cfg=norm_cfg, indice_key="layer11"),
        )

        self.layer3 = SparseSequentialBatchdict(
            SparseBasicBlock2(256,128, 512, norm_cfg=norm_cfg, indice_key="layer2"),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0, dilation=1, ceil_mode=False),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=1, ceil_mode=False),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
            SparseBasicBlock2(512, 128,512, norm_cfg=norm_cfg, indice_key="layer21"),
            SparseBasicBlock2(512, 128,512, norm_cfg=norm_cfg, indice_key="layer21"),
        )

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc_cls = nn.Linear(512, 60)

    def forward(self, x, batch_dict = {},fuse_func=None):
        # torch.Size([20, 128, 48, 32, 32])
        # torch.Size([20, 256, 48, 16, 16])
        # torch.Size([20, 512, 24, 8, 8])
        print("start")

        start_time = time.time()
        ret = x
        # print(ret.dense().shape)
        ret = self.conv_input(ret)
        ret, _loss = self.layer1(ret, batch_dict)
        # print(ret.dense().shape)
        ret, _loss = self.layer2(ret, batch_dict)
        # print(ret.dense().shape)
        ret, _loss = self.layer3(ret, batch_dict)
        # print(ret.dense().shape)

        # [N, in_channels, 4, 7, 7]
        ret = self.avg_pool(ret)
        # [N, in_channels, 1, 1, 1]
        ret = self.dropout(ret)
        # [N, in_channels, 1, 1, 1]
        ret = ret.view(ret.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(ret)
        end_time = time.time()
        print(f"runtime={end_time - start_time},fps={72/(end_time - start_time)}")
        # [N, num_classes]
        return cls_score