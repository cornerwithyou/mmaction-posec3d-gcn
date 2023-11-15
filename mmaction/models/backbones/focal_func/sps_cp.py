import os.path as osp

import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.core import ConvAlgo
import numpy as np

import os
visual=False
num=0

class call_Sps(spconv.SparseModule):
    def __init__(self, in_channels,out_channels, kernel_size,stride, padding=1,norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01), indice_key=None,
                skip_loss=False, mask_multi=False, topk=False, threshold=0.5, enlarge_voxel_channels=-1, use_img=False,
                point_cloud_range=[-5.0, -54, -54, 3.0, 54, 54], voxel_size=[0.2, 0.075, 0.075], **kwargs):
        super(call_Sps, self).__init__()
        if in_channels == 3:
            self.in_channels = 17
        else:
            self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = kwargs["dilation"] if "dilation" in kwargs else 1
        self.transposed = kwargs["transposed"] if "transposed" in kwargs else False
        self.output_padding = kwargs["output_padding"] if "output_padding" in kwargs else 0
        self.groups = kwargs["groups"] if "groups" in kwargs else 1
        # indice_key = str(in_channels) + str(out_channels) + str(kernel_size)
        if in_channels != out_channels:
            self.conv = spconv.SubMConv3d(
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride=stride,
                            padding=1,
                            bias=False,
                            indice_key=indice_key,
                        )
        else:
            self.conv = SpatialPrunedSubmConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    voxel_stride=1,
                    indice_key=indice_key,
                    stride=1,
                    padding=0,
                    bias=False,
                    pruning_ratio=0.56,
                    # pred_mode="learnable",
                    # pred_kernel_size=(1,3,3),
                    pred_mode="attn_pred",
                    pred_kernel_size=None,
                    pruning_mode="thre")

    def forward(self, x):
        y = self.conv(x)
        return y


class SpatialPrunedSubmConvBlock(spconv.SparseModule):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 voxel_stride,
                 indice_key=None, 
                 stride=1, 
                 padding=0, 
                 bias=False, 
                 pruning_ratio=0.5,
                 pred_mode="attn_pred",
                 pred_kernel_size=None,
                 point_cloud_range=[-3, -40, 0, 1, 40, 70.4],
                 voxel_size = [0.1, 0.05, 0.05],
                 algo=ConvAlgo.Native,
                 pruning_mode="thre"):
        super().__init__()
        self.indice_key = indice_key
        self.pred_mode =  pred_mode
        self.pred_kernel_size = pred_kernel_size
        self.ori_pruning_ratio= pruning_ratio
        self.pruning_ratio = pruning_ratio
        self.kernel_size = kernel_size
        self.inv_idx =  torch.Tensor([2, 1, 0]).long().cuda()
        
        self.pruning_mode = pruning_mode
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.voxel_stride = voxel_stride
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()
    
        if pred_mode=="learnable":
            assert pred_kernel_size is not None
            self.pred_conv = spconv.SubMConv3d(
                    in_channels,
                    1,
                    kernel_size=pred_kernel_size,
                    stride=1,
                    padding=padding,
                    bias=False,
                    indice_key=indice_key + "_pred_conv",
                    algo=algo
                )

        self.conv_block = spconv.SubMConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        bias=bias,
                                        indice_key=indice_key,
                                        # subm_torch=False,
                                        algo=algo
                                    )
        
        self.sigmoid = nn.Sigmoid()

    def _combine_feature(self, x_im, x_nim, mask_position):
        assert x_im.features.shape[0] == x_nim.features.shape[0] == mask_position.shape[0]
        new_features = x_im.features
        new_features[mask_position] = x_nim.features[mask_position]
        x_im = x_im.replace_feature(new_features)
        return x_im 

    def get_importance_mask(self, x, voxel_importance):
        batch_size = x.batch_size
        mask_position = torch.zeros(x.features.shape[0],).cuda()
        index = x.indices[:, 0]
        for b in range(batch_size):
            batch_index = index==b
            batch_voxel_importance = voxel_importance[batch_index]
            batch_mask_position = mask_position[batch_index]
            if self.pruning_mode == "topk":
                batch_mask_position_idx = torch.argsort(batch_voxel_importance.view(-1,))[:int(batch_voxel_importance.shape[0]*self.pruning_ratio)]
                batch_mask_position[batch_mask_position_idx] = 1
                mask_position[batch_index] =  batch_mask_position
            elif self.pruning_mode == "thre":
                batch_mask_position_idx = (batch_voxel_importance.view(-1,) <= self.pruning_ratio)
                batch_mask_position[batch_mask_position_idx] = 1
                mask_position[batch_index] =  batch_mask_position
        return mask_position.bool()


    def forward(self, x):
        # pred importance
        if self.pred_mode=="learnable":
            x_ = x
            x_conv_predict = self.pred_conv(x_)
            voxel_importance = self.sigmoid(x_conv_predict.features) # [N, 1]
        elif self.pred_mode=="attn_pred":
            x_features = x.features
            x_attn_predict = torch.abs(x_features).sum(1) / x_features.shape[1]
            voxel_importance = self.sigmoid(x_attn_predict.view(-1, 1))
        else:
             raise Exception('pred_mode is not define')

        # get mask
        mask_position = self.get_importance_mask(x, voxel_importance)
        global num
        # conv
        if visual:
            import seaborn as sns
            import matplotlib.pyplot as plt
            xx = x.dense().detach().cpu().numpy()
            xx_v = x
            voxel_importance_vv = xx_v.replace_feature(voxel_importance)
            voxel_importance_vv = voxel_importance_vv.dense().detach().cpu().numpy()
            if not os.path.exists(f'/work/gyz_Projects/mmaction222/mmaction2/demo/heatmap_geadcam/feature_map/S001C001P001R001A016/{num}'):
                os.makedirs(f'/work/gyz_Projects/mmaction222/mmaction2/demo/heatmap_geadcam/feature_map/S001C001P001R001A016/{num}')
            for T in range(1):
                plt.figure(num=num)
                ax = sns.heatmap(np.mean(np.abs(xx), axis=1)[0,T], cmap='viridis')

                plt.savefig(f'/work/gyz_Projects/mmaction222/mmaction2/demo/heatmap_geadcam/feature_map/S001C001P001R001A016/{num}/xx_{num}_{T}.png', format='png',
                            dpi=1000)
                print(f'save xx_{num}_{T}.png!')
            for T in range(1):
                plt.figure(f'{num}')
                ax = sns.heatmap(np.mean(np.abs(voxel_importance_vv), axis=1)[0,T], cmap='viridis')

                plt.savefig(f'/work/gyz_Projects/mmaction222/mmaction2/demo/heatmap_geadcam/feature_map/S001C001P001R001A016/{num}/voxel_importance_vv_{num}_{T}.png', format='png',
                            dpi=1000)
                print(f'save voxel_importance_vv_{num}_{T}.png!')

        x = x.replace_feature(x.features * voxel_importance)
        x_nim = x
        x_im = self.conv_block(x)

        if visual:
            import seaborn as sns
            import matplotlib.pyplot as plt
            xx_nim = x_nim
            xx_nim = xx_nim.replace_feature(x_nim.features[mask_position])
            xx_nim.indices = x_nim.indices[mask_position]
            xx_nim = xx_nim.dense().detach().cpu().numpy()
            for T in range(1):
                plt.figure(num=num+1)
                ax = sns.heatmap(np.mean(np.abs(xx_nim), axis=1)[0,T], cmap='viridis')

                plt.savefig(f'/work/gyz_Projects/mmaction222/mmaction2/demo/heatmap_geadcam/feature_map/S001C001P001R001A016/{num}/xx_nim_{num}_{T}.png', format='png',
                            dpi=1000)
                print(f'save xx_nim_{num}_{T}.png!')
            xx_im = x_im
            xx_im = xx_im.replace_feature(x_im.features[~mask_position])
            xx_im.indices = x_im.indices[~mask_position]
            xx_im = xx_im.dense().detach().cpu().numpy()
            for T in range(1):
                plt.figure(num=num+2)
                ax = sns.heatmap(np.mean(np.abs(xx_im), axis=1)[0,T], cmap='viridis')

                plt.savefig(f'/work/gyz_Projects/mmaction222/mmaction2/demo/heatmap_geadcam/feature_map/S001C001P001R001A016/{num}/xx_im_{num}_{T}.png', format='png',
                            dpi=1000)
                print(f'save xx_im_{num}_{T}.png!')
        num += 3

        # mask feature
        out = self._combine_feature(x_im, x_nim, mask_position)
        return out