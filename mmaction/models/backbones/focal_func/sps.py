import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.core import ConvAlgo

class SparseSequentialBatchdict(spconv.SparseSequential):
    def __init__(self, *args, **kwargs):
        super(SparseSequentialBatchdict, self).__init__(*args, **kwargs)

    def forward(self, input, batch_dict=None):
        for k, module in self._modules.items():
            if module is None:
                continue
            input, batch_dict = module(input, batch_dict)
        return input, batch_dict


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
        indice_key = str(in_channels) + str(out_channels) + str(kernel_size)
        if in_channels != out_channels:
            # self.conv = SpatialPrunedConvDownsample(
            #         in_channels,
            #         out_channels,
            #         kernel_size,
            #         voxel_stride=1,
            #         indice_key=indice_key,
            #         stride=1,
            #         padding=0,
            #         bias=False,
            #         pruning_ratio=0.5,
            #         pred_mode="attn_pred",
            #         pred_kernel_size=None,
            #         pruning_mode="topk")
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
                    # voxel_stride=1,
                    indice_key=indice_key,
                    stride=1,
                    padding=0,
                    bias=False,
                    pruning_ratio=0.5,
                    pred_mode="attn_pred",
                    pred_kernel_size=None,
                    pruning_mode="thre")

    def forward(self, x):
        y = self.conv(x)
        return y


def sort_by_indices(features_foreground_cat, indices_foreground_coords, additional_features=None):
    a = indices_foreground_coords[:, 1:]
    # print("a shape:", a.shape)
    augmented_a = a.select(1, 0) * a[:, 1].max() * a[:, 2].max() + a.select(1, 1) * a[:, 2].max() + a.select(1, 2)
    augmented_a_sorted, ind = augmented_a.sort()
    features_foreground_cat = features_foreground_cat[ind]
    indices_foreground_coords = indices_foreground_coords[ind]
    if not additional_features is None:
        additional_features = additional_features[ind]
    return features_foreground_cat, indices_foreground_coords, additional_features

def check_repeat(x_foreground_features, x_foreground_indices, additional_features=None, sort_first=True, flip_first=True):
    if sort_first:
        x_foreground_features, x_foreground_indices, additional_features = sort_by_indices(x_foreground_features, x_foreground_indices, additional_features)

    if flip_first:
        x_foreground_features, x_foreground_indices = x_foreground_features.flip([0]), x_foreground_indices.flip([0])

    if not additional_features is None:
        additional_features=additional_features.flip([0])

    a = x_foreground_indices[:, 1:].int()
    augmented_a = torch.add(torch.add(a.select(1, 0) * a[:, 1].max() * a[:, 2].max(), a.select(1, 1) * a[:, 2].max()), a.select(1, 2))
    _unique, inverse, counts = torch.unique_consecutive(augmented_a, return_inverse=True, return_counts=True, dim=0)

    if _unique.shape[0] < x_foreground_indices.shape[0]:
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        x_foreground_features_new = torch.zeros((_unique.shape[0], x_foreground_features.shape[-1]), device=x_foreground_features.device)
        x_foreground_features_new.index_add_(0, inverse.long(), x_foreground_features)
        x_foreground_features = x_foreground_features_new
        perm_ = inverse.new_empty(_unique.size(0)).scatter_(0, inverse, perm)
        x_foreground_indices = x_foreground_indices[perm_].int()

        if not additional_features is None:
            additional_features_new = torch.zeros((_unique.shape[0],), device=additional_features.device)
            additional_features_new.index_add(0, inverse.long(), additional_features)
            additional_features = additional_features_new / counts
    return x_foreground_features, x_foreground_indices, additional_features

def split_voxels_v2(x, b, voxel_importance, kernel_offsets, mask_multi=True, pruning_mode="topk", pruning_ratio=0.5):
    index = x.indices[:, 0]
    batch_index = index==b
    indices_ori = x.indices[batch_index]
    features_ori = x.features[batch_index]
    voxel_importance = voxel_importance[batch_index]

    if mask_multi:
        features_ori *= voxel_importance

    # get mask
    # print("pruning_mode-----------------------:", pruning_mode)
    if pruning_mode == "topk":
        _, indices = voxel_importance.view(-1,).sort()
        indices_im = indices[int(voxel_importance.shape[0]*pruning_ratio):]
        indices_nim = indices[:int(voxel_importance.shape[0]*pruning_ratio)]
        # print("indices_im num:", indices_im.sum(), "indices_nim num:",indices_nim.sum(), "pruning_ratio:", pruning_ratio, "x shape:", x.features.shape)
        # print("indices_im num:", indices_im.shape, "indices_nim num:",indices_nim.shape, "pruning_ratio:", pruning_ratio, "x shape:", x.features.shape)
    elif pruning_mode == "thre":
        indices_im = (voxel_importance.view(-1,) > pruning_ratio)
        indices_nim = (voxel_importance.view(-1,) <= pruning_ratio)
        # print("indices_im num:", indices_im.sum(), "indices_nim num:",indices_nim.sum(), "pruning_ratio:", pruning_ratio, "x shape:", x.features.shape)

    features_im = features_ori[indices_im]
    coords_im = indices_ori[indices_im]
    voxel_kerels_offset = kernel_offsets.unsqueeze(0).repeat(features_im.shape[0],1, 1) # [features_im.shape[0], 26, 3]
    indices_im_kernels = coords_im[:, 1:].unsqueeze(1).repeat(1, kernel_offsets.shape[0], 1) # [coords_im.shape[0], 26, 3]
    # print("kernel_offsets:", kernel_offsets.dtype, "indices_im_kernels:", indices_im_kernels.dtype, "voxel_kerels_offset:", voxel_kerels_offset.dtype)
    indices_with_imp = (indices_im_kernels + voxel_kerels_offset).view(-1, 3)
    spatial_indices = (indices_with_imp[:, 0] >0) * (indices_with_imp[:, 1] >0) * (indices_with_imp[:, 2] >0)  * \
                        (indices_with_imp[:, 0] < x.spatial_shape[0]) * (indices_with_imp[:, 1] < x.spatial_shape[1]) * (indices_with_imp[:, 2] < x.spatial_shape[2])

    selected_indices = indices_with_imp[spatial_indices]
    selected_indices = torch.cat([torch.ones((selected_indices.shape[0], 1), device=features_im.device, dtype=torch.int)*b, selected_indices], dim=1)
    selected_features = torch.zeros((selected_indices.shape[0], features_ori.shape[1]), device=features_im.device)

    features_im = torch.cat([features_im, selected_features], dim=0) # [N', C]
    coords_im = torch.cat([coords_im, selected_indices], dim=0) # [N', 3]
    # mask_kernel_im = voxel_importance[indices_im][spatial_indices]
    # mask_kernel_im = mask_kernel_im.unsqueeze(1).repeat(1, kernel_offsets.shape[0], 1)
    # mask_kernel_im = torch.cat([torch.ones(features_im_cat.shape[0], device=features_im.device), mask_kernel_im], dim=0)
    # print("before:", features_im.shape)
    assert features_im.shape[0] == coords_im.shape[0]
    if indices_im.sum()>0:
        features_im, coords_im, _ = check_repeat(features_im, coords_im)
        # print("after:", features_im.shape)
    # print("coords_im after:", coords_im.dtype)
    features_nim = features_ori[indices_nim]
    coords_nim = indices_ori[indices_nim]

    return features_im, coords_im, features_nim, coords_nim



class SpatialPrunedSubmConvBlock(spconv.SparseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 #voxel_stride,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 indice_key=None,
                 algo=ConvAlgo.Native,
                 pruning_ratio=0.6,
                 pred_mode="attn_pred",
                 pred_kernel_size=None,
                 #point_cloud_range=[-3, -40, 0, 1, 40, 70.4],
                 #voxel_size = [0.1, 0.05, 0.05],
                 pruning_mode="thre",
                 output_padding=1,
                 transposed = True,
                 inverse = True):
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
        #self.bias = bias

        self.dilation = dilation#, dilation,dilation)
        self.output_padding= output_padding,
        self.transposed= transposed,
        self.inverse= inverse,
        self.groups = groups,

        #self.voxel_stride = voxel_stride
        #self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        #self.voxel_size = torch.Tensor(voxel_size).cuda()

        if pred_mode=="learnable":
            assert pred_kernel_size is not None
            self.pred_conv = spconv.SubMConv3d(
                    in_channels,
                    1,
                    kernel_size=pred_kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=self.dilation,
                    #groups=self.groups,
                    bias=bias,
                    indice_key=indice_key + "_pred_conv",
                    algo=algo
                )

        self.conv_block = spconv.SubMConv3d(
                                        self.in_channels,
                                        self.out_channels,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        #groups=self.groups,
                                        bias=bias,
                                        indice_key=indice_key,
                                        # subm_torch=False,
                                        algo=algo
                                    )
        if self.in_channels != self.out_channels :
            self.conv_block_2 = spconv.SubMConv3d(
                self.in_channels,
                self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                # groups=self.groups,
                bias=bias,
                indice_key=indice_key,
                # subm_torch=False,
                algo=algo
            )

        self.sigmoid = nn.Sigmoid()

    def _combine_feature(self, x_im, x_nim, mask_position):
        assert x_im.features.shape[0] == x_nim.features.shape[0] == mask_position.shape[0]
        new_features = x_im.features
        #new_features[mask_position] = x_nim.features[mask_position]
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

        # conv
        x = x.replace_feature(x.features * voxel_importance)
        if self.in_channels != self.out_channels:
            x_nim = self.conv_block_2(x)
        else:
            x_nim = x
        x_im = self.conv_block(x)

        # mask feature
        out = self._combine_feature(x_im, x_nim, mask_position)

        return out

class SpatialPrunedConvDownsample(spconv.SparseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 indice_key=None,
                 stride=1,
                 padding=0,
                 bias=False,
                 pruning_ratio=0.5,
                 dilation=1,
                 voxel_stride=1,
                 point_cloud_range=[-3, -40, 0, 1, 40, 70.4],
                 voxel_size=[0.1, 0.05, 0.05],
                 pred_mode="attn_pred",
                 pred_kernel_size=None,
                 algo=ConvAlgo.Native,
                 pruning_mode="topk"):
        super().__init__()
        if isinstance(padding, int):
            self.padding = [padding] * 3
        else:
            self.padding = padding
        self.indice_key = indice_key
        self.stride = stride
        self.dilation = dilation
        self.pred_mode =  pred_mode
        self.pred_kernel_size = pred_kernel_size

        self.pruning_ratio = pruning_ratio
        self.origin_pruning_ratio = pruning_ratio
        self.kernel_size = kernel_size
        self.inv_idx =  torch.Tensor([2, 1, 0]).long().cuda()

        self.pruning_mode = pruning_mode

        self.in_channels = in_channels
        self.out_channels = out_channels

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
                    dilation=1,
                    groups=1,
                    bias=True,
                    indice_key=indice_key + "_pred_conv",
                    algo=algo
                )


        self.conv_block = spconv.SubMConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=padding,
                                        bias=bias,
                                        indice_key=indice_key,
                                        # subm_torch=False,
                                        algo=algo
                                    )

        _stepx = int(kernel_size[0]//2)
        _stepy = int(kernel_size[1]//2)
        _stepz = int(kernel_size[2]//2)
        kernel_offsets = [[i, j, k] for i in range(-_stepx, _stepx+1) for j in range(-_stepy, _stepy+1) for k in range(-_stepz, _stepz+1)]
        kernel_offsets.remove([0, 0, 0])
        self.kernel_offsets = torch.Tensor(kernel_offsets).cuda().int()

        self.sigmoid = nn.Sigmoid()

    def gemerate_sparse_tensor(self, x, voxel_importance):
        batch_size = x.batch_size
        voxel_features_im = []
        voxel_indices_im = []
        voxel_features_nim = []
        voxel_indices_nim = []
        for b in range(batch_size):
            features_im, indices_im, features_nim, indices_nim = split_voxels_v2(x, b, voxel_importance, self.kernel_offsets, pruning_mode=self.pruning_mode, pruning_ratio=self.pruning_ratio)
            voxel_features_im.append(features_im)
            voxel_indices_im.append(indices_im)
            voxel_features_nim.append(features_nim)
            voxel_indices_nim.append(indices_nim)

        voxel_features_im = torch.cat(voxel_features_im, dim=0)
        voxel_indices_im = torch.cat(voxel_indices_im, dim=0)
        voxel_features_nim = torch.cat(voxel_features_nim, dim=0)
        voxel_indices_nim = torch.cat(voxel_indices_nim, dim=0)
        x_im = spconv.SparseConvTensor(voxel_features_im, voxel_indices_im, x.spatial_shape, x.batch_size)
        x_nim = spconv.SparseConvTensor(voxel_features_nim, voxel_indices_nim, x.spatial_shape, x.batch_size)

        return x_im, x_nim

    def combine_feature(self, x_im, x_nim, remove_repeat=True):
        x_features = torch.cat([x_im.features, x_nim.features], dim=0)
        x_indices = torch.cat([x_im.indices, x_nim.indices], dim=0)
        if remove_repeat:
            index = x_indices[:, 0]
            features_out_list = []
            indices_coords_out_list = []
            for b in range(x_im.batch_size):
                batch_index = index==b
                features_out, indices_coords_out, _ = check_repeat(x_features[batch_index], x_indices[batch_index], flip_first=False)
                features_out_list.append(features_out)
                indices_coords_out_list.append(indices_coords_out)
            x_features = torch.cat(features_out_list, dim=0)
            x_indices = torch.cat(indices_coords_out_list, dim=0)

        x_im = x_im.replace_feature(x_features)
        x_im.indices = x_indices
        return x_im

    def reset_spatial_shape(self, x):
        indices = x.indices
        features = x.features
        conv_valid_mask = ((indices[:,1:] % 2).sum(1)==0)

        pre_spatial_shape = x.spatial_shape
        new_spatial_shape = []
        for i in range(3):
            size = (pre_spatial_shape[i] + 2 * self.padding[i] - self.dilation *
                    (self.kernel_size - 1) - 1) // self.stride + 1
            if self.kernel_size == -1:
                new_spatial_shape.append(1)
            else:
                new_spatial_shape.append(size)
        indices[:,1:] = indices[:,1:] // 2
        coords = indices[:,1:][conv_valid_mask]
        spatial_indices = (coords[:, 0] >0) * (coords[:, 1] >0) * (coords[:, 2] >0)  * \
            (coords[:, 0] < new_spatial_shape[0]) * (coords[:, 1] < new_spatial_shape[1]) * (coords[:, 2] < new_spatial_shape[2])

        x = spconv.SparseConvTensor(features[conv_valid_mask][spatial_indices], indices[conv_valid_mask][spatial_indices].contiguous(), new_spatial_shape, x.batch_size)

        return x

    def forward(self, x):

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

        x_im, x_nim = self.gemerate_sparse_tensor(x, voxel_importance)
        out = self.combine_feature(x_im, x_nim, remove_repeat=True)
        out = self.conv_block(out)
        out = self.reset_spatial_shape(out)

        return out

