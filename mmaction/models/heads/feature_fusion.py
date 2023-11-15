import torch
from torch import nn

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
feature_dim = 512
num_heads = 8
fusion_model = FeatureFusion(feature_dim, num_heads)

# Assume we have 10 images and 10 corresponding skeletons in a batch, each with a sequence length of 100
image_features = torch.rand(100, 10, feature_dim)  # (seq_len, batch, feature_dim)
skeleton_features = torch.rand(100, 10, feature_dim)  # (seq_len, batch, feature_dim)

fused_output = fusion_model(image_features, skeleton_features)