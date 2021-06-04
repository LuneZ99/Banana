import torch
import numpy as np
import torch.nn as nn

from functools import partial
from einops.layers.torch import Rearrange, Reduce


# 继承式实现


class TokenMixer(nn.Module):
    def __init__(self, dim_norm, dim, expansion_factor=4, dropout=0.):
        super(TokenMixer, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim * expansion_factor, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim * expansion_factor, dim, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim_norm)

    def forward(self, x):
        return self.block(self.norm(x)) + x


class ChannelMixer(nn.Module):
    def __init__(self, dim_norm, dim, expansion_factor=4, dropout=0.):
        super(ChannelMixer, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim_norm)

    def forward(self, x):
        return self.block(self.norm(x)) + x


class MLPMixer(nn.Module):
    def __init__(self, image_size, channels, patch_size, hidden_dim, num_block, num_classes, expansion_factor=4, dropout=0.):
        super(MLPMixer, self).__init__()
        num_patches = (image_size // patch_size) ** 2   # tokens_mlp_dim
        self.num_blocks = num_block
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        self.linear = nn.Linear((patch_size ** 2) * channels, hidden_dim)

        self.mixer_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    TokenMixer(hidden_dim, num_patches, expansion_factor, dropout),
                    ChannelMixer(hidden_dim, hidden_dim, expansion_factor, dropout)
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.pre_head_layer_norm = nn.LayerNorm(hidden_dim)
        self.reduce = Reduce('b n c -> b c', 'mean')
        self.linear_output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.rearrange(x)
        x = self.linear(x)
        for i in range(self.num_blocks):
            x = self.mixer_blocks[i](x)
        x = self.pre_head_layer_norm(x)
        x = self.reduce(x)
        x = self.linear_output(x)
        return x


# from https://github.com/lucidrains/mlp-mixer-pytorch ↓

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def feed_forward_layer(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )


def get_mlp_mixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor=4, dropout=0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, feed_forward_layer(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, feed_forward_layer(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )


if __name__ == '__main__':
    mlp_mixer = MLPMixer(
        image_size=256,
        channels=3,
        patch_size=16,
        hidden_dim=512,
        num_block=12,
        num_classes=1000
    )

    mlp_mixer2 = get_mlp_mixer(
        image_size=256,
        channels=3,
        patch_size=16,
        dim=512,
        depth=12,
        num_classes=1000
    )

    img = torch.randn(1, 3, 256, 256)

    print(mlp_mixer(img).argmax().item())
    print(mlp_mixer2(img).argmax().item())



