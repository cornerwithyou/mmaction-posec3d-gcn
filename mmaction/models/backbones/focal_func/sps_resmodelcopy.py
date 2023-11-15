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
from sps_cp import call_Sps

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

def conv3x3(in_planes, out_planes,stride=1, indice_key=None,norm_cfg=None ,  bias=True,**kwargs):
    """3x3 convolution with padding"""
    return call_Sps(in_planes,
                    out_planes, 
                    kernel_size=(1,3,3),
                    stride=stride, 
                    padding=1,
                    indice_key=indice_key + "sps",
                )
        
def conv311(in_planes, out_planes,stride=1, indice_key=None,norm_cfg=None ,  bias=True,**kwargs):
    # return call_Sps(in_planes,
    #                 out_planes, 
    #                 kernel_size=(3,1,1),
    #                 stride=stride, 
    #                 padding=1,
    #                 indice_key=indice_key + "sps",
    #             )
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=(3,1,1),
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )
def conv1x1(in_planes, out_planes,stride=1, indice_key=None,norm_cfg=None ,  bias=True,**kwargs):
    """1x1 convolution"""
    # return call_Sps(in_planes,
    #                 out_planes, 
    #                 kernel_size=(1,1,1),
    #                 stride=stride, 
    #                 padding=1,
    #                 indice_key=indice_key + "sps",
    #             )
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=(1,1,1),
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
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
        visualize_file = None,
    ):
        super(SparseBasicBlock1, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None
        self.visualize = visualize
        self.visualizefile = visualize_file
        self.conv1 = conv1x1(inplanes, midplanes, stride, indice_key=indice_key + "1",norm_cfg=norm_cfg,  bias=bias)
        # self.conv1 = conv1x1(inplanes, midplanes, stride, indice_key=indice_key,norm_cfg=norm_cfg,  bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, midplanes)[1]
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.conv2 = conv3x3(midplanes, midplanes, indice_key=indice_key + "3",norm_cfg=norm_cfg, bias=bias)
        # self.conv2 = conv3x3(midplanes, midplanes, indice_key=indice_key,norm_cfg=norm_cfg, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, midplanes)[1]
        self.conv3 = conv1x1(midplanes, planes, indice_key=indice_key + "1",norm_cfg=norm_cfg, bias=bias)
        # self.conv3 = conv1x1(midplanes, planes, indice_key=indice_key,norm_cfg=norm_cfg, bias=bias)
        self.bn3 = build_norm_layer(norm_cfg, planes)[1]
        if downsample or inplanes != planes:
            self.downsample = conv1x1(inplanes, planes, indice_key=indice_key + "1",norm_cfg=norm_cfg, bias=bias)
            # self.downsample = SubMConv3d(inplanes, planes,(1,1,1),stride=(1, 2, 2), indice_key=indice_key,bias=bias)
            self.bn4 = build_norm_layer(norm_cfg, planes)[1]
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        # out,_ = self.conv1(x,batch_dict)
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu1(out.features))

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
        if self.visualize:
            import pickle
            with open(self.visualize_file, 'wb') as f:
                pickle.dump(out, f)
        out = out.replace_feature(self.relu2(out.features))

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
        visualize_file=None,
    ):
        super(SparseBasicBlock2, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None
        self.visualize = visualize
        self.visualizefile = visualize_file
        # self.conv1 = conv3x3(inplanes, midplanes, stride, indice_key=indice_key + "0",norm_cfg=norm_cfg,  bias=bias)
        self.conv1 = conv311(inplanes, midplanes, stride, indice_key=indice_key+"1",norm_cfg=norm_cfg,  bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, midplanes)[1]
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        # self.conv2 = conv3x3(midplanes, midplanes, indice_key=indice_key + "1",norm_cfg=norm_cfg, bias=bias)
        self.conv2 = conv3x3(midplanes, midplanes, indice_key=indice_key+"2",norm_cfg=norm_cfg, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, midplanes)[1]
        # self.conv3 = conv3x3(midplanes, planes, indice_key=indice_key + "2",norm_cfg=norm_cfg, bias=bias)
        self.conv3 = conv1x1(midplanes, planes, indice_key=indice_key + "3",norm_cfg=norm_cfg, bias=bias)
        self.bn3 = build_norm_layer(norm_cfg, planes)[1]
        if downsample or inplanes != planes:
            self.downsample = conv1x1(inplanes, planes, indice_key=indice_key + "3",norm_cfg=norm_cfg, bias=bias)
            # self.downsample = SubMConv3d(inplanes, planes,(1,1,1), indice_key=indice_key,bias=bias)
            self.bn4 = build_norm_layer(norm_cfg, planes)[1]
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        # out,_ = self.conv1(x,batch_dict)
        out = self.conv1(x)

        out = out.replace_feature(self.bn1(out.features))
        # if self.visualize:
        #     import pickle
        #     with open('bn3_out.pkl', 'wb') as f:
        #         pickle.dump(out, f)
        out = out.replace_feature(self.relu1(out.features))

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
        if self.visualize:
            import pickle
            with open(self.visualizefile, 'wb') as f:
                pickle.dump(out, f)
        out = out.replace_feature(self.relu2(out.features))
        return out

class spsresnet(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpMiddleResNetFHDFocal", **kwargs
    ):
        super(spsresnet, self).__init__()
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
        maxpool = SparseMaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=1)
        pool1 = SparseMaxPool3d(kernel_size=(2,1,1), stride=(2,1,1), padding=0, dilation=1)
        pool2 = SparseMaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=0, dilation=1)
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(num_input_features, 32, kernel_size=(1,7,7),stride=(1, 1, 1), padding=(0, 3, 3), bias=False, indice_key="res0"),
            # special_spconv_fn(num_input_features, 32, kernel_size=(1,7,7),voxel_stride=1, indice_key="res0"),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(inplace=True),
            maxpool,
        )

        self.layer1 = spconv.SparseSequential(
            SparseBasicBlock1(32, 32,128 ,norm_cfg=norm_cfg, indice_key="layer00",visualize_file="backbone_model_layer1_0_relu2"),
            pool2,
            SparseBasicBlock1(128, 32,128 ,norm_cfg=norm_cfg, indice_key="layer0",visualize_file="backbone_model_layer1_2_relu2"),
            SparseBasicBlock1(128, 32,128 ,norm_cfg=norm_cfg, indice_key="layer0",visualize_file="backbone_model_layer1_3_relu2"),
            SparseBasicBlock1(128, 32,128 ,norm_cfg=norm_cfg, indice_key="layer0",visualize_file="backbone_model_layer1_4_relu2"),
        )

        self.layer2 = spconv.SparseSequential(
            SparseBasicBlock2(128,64, 256, norm_cfg=norm_cfg, indice_key="layer11",visualize_file="backbone_model_layer2_0_relu2"),
            pool2,
            SparseBasicBlock2(256,64, 256, norm_cfg=norm_cfg, indice_key="layer1",visualize_file="backbone_model_layer2_2_relu2"),
            SparseBasicBlock2(256,64, 256, norm_cfg=norm_cfg, indice_key="layer1",visualize_file="backbone_model_layer2_3_relu2"),
            SparseBasicBlock2(256,64, 256, norm_cfg=norm_cfg, indice_key="layer1",visualize_file="backbone_model_layer2_4_relu2"),
            SparseBasicBlock2(256,64, 256, norm_cfg=norm_cfg, indice_key="layer1",visualize_file="backbone_model_layer2_5_relu2"),
            SparseBasicBlock2(256,64, 256, norm_cfg=norm_cfg, indice_key="layer1",visualize_file="backbone_model_layer2_6_relu2"),
        )

        self.layer3 = spconv.SparseSequential(
            SparseBasicBlock2(256,128, 512, norm_cfg=norm_cfg, indice_key="layer22",visualize_file="backbone_model_layer3/0/relu2"),
            pool1,
            pool2,
            SparseBasicBlock2(512, 128,512, norm_cfg=norm_cfg, indice_key="layer2",visualize_file="backbone_model_layer3_3_relu2"),
            SparseBasicBlock2(512, 128,512, norm_cfg=norm_cfg, indice_key="layer2",visualize_file="backbone_model_layer3_4_relu2"),
        )

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc_cls = nn.Linear(512, 60)

    def forward(self, x,fuse_func=None):
        # torch.Size([20, 128, 48, 32, 32])
        # torch.Size([20, 256, 48, 16, 16])
        # torch.Size([20, 512, 24, 8, 8])
        x = spconv.SparseConvTensor.from_dense(x.permute(0,2,3,4,1))
        ret = x
        # print(ret.dense().shape)
        ret = self.conv_input(ret)
        # print(ret.dense().shape)
        ret = self.layer1(ret)
        # print(ret.dense().shape)
        ret = self.layer2(ret)
        # print(ret.dense().shape)
        ret = self.layer3(ret)
        # print(ret.dense().shape)
        ret = ret.dense()

        # avg_pool
        ret = self.avg_pool(ret)
        # [N, in_channels, 1, 1, 1]
        ret = self.dropout(ret)
        # [N, in_channels, 1, 1, 1]
        ret = ret.view(ret.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(ret)
        # [N, num_classes]
        return cls_score

