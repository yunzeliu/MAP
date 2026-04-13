import random

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'modules'))
# sys.path.append(os.path.join(ROOT_DIR, 'models'))
from transformer import *


class PointNet4D(nn.Module):
    def __init__(self, dim=1024, depth=1, heads=4, mlp_dim=4.0, dim_head=1024, length=150):                                                 # output
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=0.))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=0.)))
            ]))
        self.mask = get_decoder_mask(length)

        self.mamba = nn.ModuleList([
         Mamba3DBlock(
             dim=dim,
             bimamba_type="v2"
             )
        for i in range(depth)])

    def forward(self, features):

        # 4d BACKBONE
        # features: [B, L, C]
        # return: [B, L, C]

        device = features.get_device()
        self.mask = self.mask.to(device)

        for (n, block), (attn, ff) in zip(enumerate(self.mamba), self.layers):
            features = block(features)
            features = attn(features, mask=self.mask)
            features = ff(features)

        return features

