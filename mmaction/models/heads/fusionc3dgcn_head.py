# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model.weight_init import normal_init
from torch import Tensor, nn
import torch
import torch.nn.functional as F
from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from .base import BaseHead
from einops import rearrange
import math


class FeatureFusion(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super().__init__()

        # Initialize multihead attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

    def forward(self, image_features, skeleton_features):
        # image_features and skeleton_features should have shape (seq_len, batch, feature_dim)
        # If your features are (batch, seq_len, feature_dim), use .permute(1, 0, 2)

        # Attention from image to skeleton
        skeleton_output, _ = self.multihead_attn(skeleton_features, image_features, image_features)

        # Attention from skeleton to image
        image_output, _ = self.multihead_attn(image_features, skeleton_features, skeleton_features)

        # Fusion: this can be concatenation, addition, or any other operation
        fused_output = image_output + skeleton_output

        return fused_output

# Usage
'''
feature_dim = 512
num_heads = 8
fusion_model = FeatureFusion(feature_dim, num_heads)

# Assume we have 10 images and 10 corresponding skeletons in a batch, each with a sequence length of 100
image_features = torch.rand(100, 10, feature_dim)  # (seq_len, batch, feature_dim)
skeleton_features = torch.rand(100, 10, feature_dim)  # (seq_len, batch, feature_dim)

fused_output = fusion_model(image_features, skeleton_features)
'''


def generate_a_heatmap(x_pose,x_gcn, inputs_gcn, #centers: Tensor,
                       #max_values: Tensor,
                       sigma = 0.3):

    x_pose = rearrange(x_pose, 'n c t h w  -> n t h w c')
    img_h, img_w = x_pose.shape[-3], x_pose.shape[-2]
    #inputs_gcn = rearrange(inputs_gcn, 'n m c v -> n m v c')
    x_gcn_out = x_pose#torch.zeros_like(x_pose)
    #x_gcn_mask = torch.zeros_like(x_pose)
    for b,b_inputs_gcn in enumerate(inputs_gcn):
        for m, m_inputs_gcn in enumerate(b_inputs_gcn):
            for k, key_inputs_gcn in enumerate(m_inputs_gcn):
                if key_inputs_gcn[2] != 0:
                    x_coordi = math.floor(key_inputs_gcn[0] * (img_w / 2) + (img_w / 2))
                    y_coordi = math.floor(key_inputs_gcn[1] * (img_h / 2) + (img_h / 2))
                    conf = key_inputs_gcn[2]
                    if x_coordi > img_w-1 : x_coordi = img_w-1
                    if y_coordi > img_h-1 : y_coordi = img_h-1
                    x_gcn_out[b,:,x_coordi,y_coordi,:] += x_gcn[b,m,:,k,:]*conf
                else:
                    continue
    x_pose = rearrange(x_pose, 'n t h w c -> n c t h w')
    x_gcn_out = rearrange(x_gcn_out, 'n t h w c -> n c t h w')
    return x_gcn_out

class MLP(nn.Module):
    def __init__(self, input_size,hidden_size,output_size, init_std: float = 0.01):
        super().__init__()
        self.init_std = init_std
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size )

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.layer1, std=self.init_std)
        normal_init(self.layer2, std=self.init_std)
        normal_init(self.layer3, std=self.init_std)

    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x



@MODELS.register_module()
class FusionC3DGCN_Head(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 num_heads :int =8,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 spatial_type: str = 'avg',
                 dropout_ratio: float = 0.5,
                 init_std: float = 0.01,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes, bias=False)
        self.linear_256_512 = nn.Linear(int(self.in_channels/2),self.in_channels,bias=False)
        self.linear_512_512 = nn.Linear(self.in_channels,self.in_channels,bias=False)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.avg_pool_T = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            self.avg_pool = None
        self.multihead_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)
        self.mlp = MLP(input_size=self.in_channels, hidden_size=self.in_channels, output_size=self.in_channels, init_std=self.init_std)
        #self.activate = nn.GELU()
        self.norm = nn.BatchNorm1d(self.in_channels)

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x: tuple ,**kwargs) -> Tensor:

        x_pose, x_gcn, inputs_gcn = x
        #对x_pose和x_gcn作线性映射
        x_pose = rearrange(x_pose, 'n c t h w -> n t h w c')
        x_gcn = rearrange(x_gcn, 'n m c t v -> n m t v c')
        x_pose = self.linear_512_512(x_pose)
        #x_gcn = self.linear_256_512(x_gcn)
        #x_gcn = rearrange(x_gcn, ' -> n m c t v ')
        #pool掉时序维
        #x_gcn = rearrange(x_gcn, 'n m t v c -> n m t c v')
        x_pose = rearrange(x_pose, 'n t h w c -> n c t h w')
        x_pose = self.avg_pool_T(x_pose)
        x_gcn  = self.avg_pool_T(x_gcn)
        #x_gcn = rearrange(x_gcn, 'n m t c v -> (n m) t c v')
        inputs_gcn = torch.mean(inputs_gcn, dim = 2)
        #x_gcn = generate_a_heatmap(x_pose, x_gcn, inputs_gcn)
        x_pose = rearrange(x_pose, 'n c t h w -> n c (t h w)')
        #x_gcn  = rearrange(x_gcn, 'n c t h w -> n c (t h w)')
        x_pose = rearrange(x_pose, 'n c s ->s n c')
        #x_gcn = rearrange(x_gcn, 'n c s ->s n c')
        pose_out = x_pose#self.multihead_attn(x_pose, x_gcn, x_gcn)
        #gcn_out = x_gcn#self.multihead_attn(x_gcn, x_pose, x_pose)
        #pose_out=self.mlp(pose_out)
        #gcn_out =self.mlp(gcn_out)
        pose_out = rearrange(pose_out, 's n c -> n c s')
        #gcn_out = rearrange(gcn_out, 's n c -> n c s')
        if self.dropout is not None:
            pose_out = self.dropout(pose_out)
            #gcn_out = self.dropout(gcn_out)

        x = self.norm(pose_out)#+gcn_out)#torch.cat((self.norm(pose_out),self.norm(gcn_out)),dim=1)#[seq, N, feature]
        #x = torch.mean(x, dim = 0)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.avg_pool(x)
        x = rearrange(x,'b c s -> b (c s)')


        cls_score = self.fc_cls(x)
        #cls_score = self.activate(cls_score)
        # [N, num_classes]
        return cls_score

@MODELS.register_module()
class FusionC3DGCN_Head_cp(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 num_heads :int =8,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 spatial_type: str = 'avg',
                 dropout_ratio: float = 0.5,
                 init_std: float = 0.01,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes, bias=False)
        self.linear_256_512 = nn.Linear(int(self.in_channels/2),self.in_channels,bias=False)
        self.linear_512_512 = nn.Linear(self.in_channels,self.in_channels,bias=False)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.avg_pool_T = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            self.avg_pool = None
        self.multihead_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)
        self.mlp = MLP(input_size=self.in_channels, hidden_size=self.in_channels, output_size=self.in_channels, init_std=self.init_std)
        #self.activate = nn.GELU()
        self.norm = nn.BatchNorm1d(self.in_channels)

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x: tuple ,**kwargs) -> Tensor:

        x_pose, x_gcn, inputs_gcn = x
        #对x_pose和x_gcn作线性映射
        x_pose = rearrange(x_pose, 'n c t h w -> n t h w c')
        x_gcn = rearrange(x_gcn, 'n m c t v -> n m t v c')
        x_pose = self.linear_512_512(x_pose)
        #x_gcn = self.linear_256_512(x_gcn)
        #x_gcn = rearrange(x_gcn, ' -> n m c t v ')
        #pool掉时序维
        #x_gcn = rearrange(x_gcn, 'n m t v c -> n m t c v')
        x_pose = rearrange(x_pose, 'n t h w c -> n c t h w')
        #x_pose = self.avg_pool_T(x_pose)
        x_gcn  = self.avg_pool_T(x_gcn)
        #x_gcn = rearrange(x_gcn, 'n m t c v -> (n m) t c v')
        inputs_gcn = torch.mean(inputs_gcn, dim = 2)
        #x_gcn = generate_a_heatmap(x_pose, x_gcn, inputs_gcn)
        #x_pose = rearrange(x_pose, 'n c t h w -> n c (t h w)')
        #x_gcn  = rearrange(x_gcn, 'n c t h w -> n c (t h w)')
        #x_pose = rearrange(x_pose, 'n c s ->s n c')
        #x_gcn = rearrange(x_gcn, 'n c s ->s n c')
        pose_out = x_pose#self.multihead_attn(x_pose, x_gcn, x_gcn)
        #gcn_out = x_gcn#self.multihead_attn(x_gcn, x_pose, x_pose)
        #pose_out=self.mlp(pose_out)
        #gcn_out =self.mlp(gcn_out)
        #pose_out = rearrange(pose_out, 's n c -> n c s')
        #gcn_out = rearrange(gcn_out, 's n c -> n c s')
        if self.dropout is not None:
            pose_out = self.dropout(pose_out)
            #gcn_out = self.dropout(gcn_out)

        x = self.norm(pose_out)#+gcn_out)#torch.cat((self.norm(pose_out),self.norm(gcn_out)),dim=1)#[seq, N, feature]
        #x = torch.mean(x, dim = 0)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)


        cls_score = self.fc_cls(x)
        #cls_score = self.activate(cls_score)
        # [N, num_classes]
        return cls_score