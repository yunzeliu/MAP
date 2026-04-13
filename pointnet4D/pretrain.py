import random

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys
import os
from timm.models.layers import DropPath, trunc_normal_
import scipy.stats as stats
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from point_4d_convolution import *
from transformer import *
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

class P4_Pretrain(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,  # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,  # P4DConv: temporal
                 emb_relu,  # embedding: relu
                 dim, depth, heads, dim_head,  # transformer
                 mlp_dim, num_classes):  # output
        super().__init__()
        # MAP encoder
        # need to change to pn++ here!!!!
        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                      spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                      temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride,
                                      temporal_padding=[1, 1],
                                      operator='+', spatial_pooling='max', temporal_pooling='max')

        self.emb_relu1 = nn.ReLU()

        # transformer
        self.seq_len = 150
        self.mask = get_decoder_mask(self.seq_len)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=0.))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=0.)))
            ]))
        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)


        # mamba
        self.mamba = nn.ModuleList([
            Mamba3DBlock(
                dim=dim,
                bimamba_type="v2"
            )
            for i in range(depth)])

        # MAP decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        trunc_normal_(self.mask_token, std=.02)
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.decoder_depth = 4
        self.drop_path_rate = 0.1
        self.decoder_num_heads = 4
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]

        self.MAE_decoder = TransformerDecoder(
            embed_dim=dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )
        self.increase_dim = nn.Sequential(
            nn.Conv1d(dim, 3 * dim, 1)
        )

        # mask ratio
        mask_ratio_max = 0.7

        mask_ratio_min = 0.3
        loc = (mask_ratio_max + mask_ratio_min) / 2
        scale = 0.25

        a = (mask_ratio_min - loc) / scale
        b = (mask_ratio_max - loc) / scale

        self.mask_ratio_generator = stats.truncnorm(a, b, loc=loc, scale=scale)

        # loss
        self.criterion = ChamferDistanceL2()
    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def forward(self, input):

        target = input.clone()  # B*150*2048*3

        # 4d BACKBONE
        # input: [B, L, N, 3]

        xyzs, features = self.tube_embedding(input)  # [B, L, n, 3], [B, L, C, n]

        xyzs = xyzs[:, 1:-1]
        features = features[:, 1:-1]

        features = features.transpose(2, 3)  # B ,L , n, C
        B, L, N, C = features.size()

        raw_feat = features

        device = raw_feat.get_device()
        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t + 1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts,
                              shape=(xyzts.shape[0], xyzts.shape[1] * xyzts.shape[2], xyzts.shape[3]))  # [B, L*n, 4]
        # [B, L,   n, C]
        features = torch.reshape(input=raw_feat, shape=(
        raw_feat.shape[0], raw_feat.shape[1] * raw_feat.shape[2], raw_feat.shape[3]))  # [B, L*n, C]

        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts + features

        embedding = embedding.reshape(B, L, N, C)
        embedding = embedding.reshape(B * L, N, C)
        embedding = embedding.permute(0, 2, 1)
        embedding = F.adaptive_max_pool1d(embedding, (1)).reshape(B, L, C)

        features = self.emb_relu1(embedding)
        self.mask = self.mask.to(device)


        orders = self.sample_orders(bsz=features.size(0))
        random_mask = self.random_masking(features, orders)

        bsz, seq_len, embed_dim = features.shape
        features = features[(1 - random_mask).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)
        mask_seq = features.shape[1]
        for (n, block), (attn, ff) in zip(enumerate(self.mamba), self.layers):
            features = block(features)

            features = attn(features, mask=self.mask[:mask_seq,:mask_seq])
            features = ff(features)


        # start decode

        x = self.decoder_pos_embed(features)

        # pad mask tokens
        mask_tokens = self.mask_token.repeat(random_mask.shape[0], random_mask.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - random_mask).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        x = x_after_pad

        x_rec = self.MAE_decoder(x)

        B, M, C = x_rec.shape
        rebuild_points = (self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3))  # B M 1024


        target = target.reshape(B * M, -1, 3)
        loss = self.criterion(rebuild_points, target)
        return loss

