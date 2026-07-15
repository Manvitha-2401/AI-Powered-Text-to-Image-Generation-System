"""
--------------------------------------------------------
Task 5: Cross Attention Module
AI-Powered Text-to-Image Generation System

Author: Donthi Reddy Manvitha Reddy
Internship Project

Objective:
Implement a Cross-Attention mechanism to improve the
quality of generated images by aligning text embeddings
with image features.
--------------------------------------------------------
"""

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, image_dim=512, text_dim=512, num_heads=8):
        super().__init__()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, image_features, text_features):

        attended_features, attention_weights = self.cross_attention(
            query=image_features,
            key=text_features,
            value=text_features
        )

        return attended_features, attention_weights


if __name__ == "__main__":

    image_features = torch.randn(1, 6, 512)
    text_features = torch.randn(1, 6, 512)

    attention = CrossAttention()

    output, weights = attention(
        image_features,
        text_features
    )

    print("Output Shape :", output.shape)
    print("Weights Shape:", weights.shape)