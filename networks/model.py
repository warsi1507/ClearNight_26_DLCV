# ClearNight network. Backbone (DehazeFormer stages, MPE, SKFusion) is unchanged from the
# official release (BSD-3-Clause), see baseblocks.py.
# Changes:
#   1. specificity_init='fixed': the official code zero-initialises every WDS decoder conv and unit,
#      so the branch gets no gradients and only learns a constant. 'fixed' zeroes only the last conv.
#   2. flare=True: Haar DWT flare suppression on the stage-1 features (networks/flare.py).
#   3. input and both priors are padded together to a multiple of 16.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseblocks import RLN, PatchEmbed, PatchUnEmbed, BasicLayer, SKFusion, MPE
from .dsm import MixtureOfUnits
from .flare import FlareSuppression

ALL_WEATHER_LABELS = ['haze', 'snow', 'rain', 'drop']


class DWS(nn.Module):
    """Weather-aware dynamic selection block: encoder -> mixture of units -> decoder."""

    def __init__(self, dim, num_labels, num_units=25, top_k=10,
                 router_noise_std=0.01, router_l2_lambda=0.01, specificity_init='fixed'):
        super().__init__()
        if specificity_init not in ('fixed', 'official'):
            raise ValueError("specificity_init must be 'fixed' or 'official'")

        self.encoder = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(dim, dim // 2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(dim // 2, dim // 2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(dim // 2, dim // 2, 3, padding=2, dilation=2, stride=2), nn.ReLU(),
            nn.Conv2d(dim // 2, 12, 3, padding=2, dilation=2, stride=2),
        )
        self.moe = MixtureOfUnits(12, num_units, top_k, num_labels,
                                  noise_std=router_noise_std, l2_lambda=router_l2_lambda,
                                  zero_init_units=(specificity_init == 'official'))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(12, dim // 2, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(dim // 2, dim // 2, 2, stride=2), nn.ReLU(),
            nn.Conv2d(dim // 2, dim // 2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(dim // 2, dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        self._init_weights(specificity_init)

    def _init_weights(self, mode):
        convs = (nn.Conv2d, nn.ConvTranspose2d)
        for layer in self.encoder:
            if isinstance(layer, convs):
                if mode == 'official':
                    nn.init.normal_(layer.weight, std=0.02)
                else:
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)
        for layer in self.decoder:
            if isinstance(layer, convs):
                if mode == 'official':
                    nn.init.zeros_(layer.weight)
                else:
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)
        # Both modes: the block outputs exactly zero at init.
        nn.init.zeros_(self.decoder[-1].weight)

    def forward(self, x, rmap):
        z = self.encoder(torch.cat([x, rmap], dim=1))
        z, class_logits, l2_reg, unit_idx, lb_loss = self.moe(z)
        return self.decoder(z), class_logits, l2_reg, unit_idx, lb_loss


class ClearNight(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, window_size=8,
                 embed_dims=(24, 48, 96, 48, 24),
                 mlp_ratios=(2., 4., 4., 2., 2.),
                 depths=(8, 8, 8, 4, 4),
                 num_heads=(2, 4, 6, 1, 1),
                 attn_ratio=(1 / 4, 1 / 2, 3 / 4, 0, 0),
                 conv_type=('DWConv',) * 5,
                 norm_layer=(RLN,) * 5,
                 num_labels=4, moe_num_units=25, moe_top_k=10,
                 router_noise_std=0.01, router_l2_lambda=0.01, specificity_init='fixed', flare=False):
        super().__init__()
        self.patch_size = 16
        net_depth = sum(depths)

        def stage(i):
            return BasicLayer(network_depth=net_depth, dim=embed_dims[i], depth=depths[i],
                              num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                              norm_layer=norm_layer[i], window_size=window_size,
                              attn_ratio=attn_ratio[i], attn_loc='last', conv_type=conv_type[i])

        self.patch_embed = PatchEmbed(patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)
        self.layer1 = stage(0)
        self.patch_merge1 = PatchEmbed(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)
        self.layer2 = stage(1)
        self.patch_merge2 = PatchEmbed(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)
        self.layer3 = stage(2)
        self.patch_split1 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])
        self.fusion1 = SKFusion(embed_dims[3])
        self.layer4 = stage(3)
        self.patch_split2 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])
        self.fusion2 = SKFusion(embed_dims[4])
        self.layer5 = stage(4)
        self.patch_unembed = PatchUnEmbed(patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

        self.mpe = MPE(channel=48)

        self.flare = FlareSuppression(embed_dims[0]) if flare else None

        wds = dict(num_units=moe_num_units, top_k=moe_top_k,
                   router_noise_std=router_noise_std, router_l2_lambda=router_l2_lambda,
                   specificity_init=specificity_init)
        self.moe1 = DWS(embed_dims[0], num_labels, **wds)
        self.moe2 = DWS(embed_dims[1], num_labels, **wds)
        self.moe3 = DWS(embed_dims[2], num_labels, **wds)
        self.moe4 = DWS(embed_dims[3], num_labels, **wds)
        self.moe5 = DWS(embed_dims[4], num_labels, **wds)

    def pad(self, x):
        h, w = x.shape[-2:]
        ph = (self.patch_size - h % self.patch_size) % self.patch_size
        pw = (self.patch_size - w % self.patch_size) % self.patch_size
        return F.pad(x, (0, pw, 0, ph), 'reflect')

    def forward_features(self, x, imap, rmap):
        imap1, imap2, imap3 = self.mpe(imap)
        rmap1, rmap2, rmap3 = self.mpe(rmap)

        x = self.patch_embed(x)
        if self.flare is not None:
            x = self.flare(x, imap.mean(dim=1, keepdim=True))  # before the first WDS / TFB stage
        y1, c1, r1, _, b1 = self.moe1(x, rmap1)
        x = self.layer1(x, imap=imap1) + y1
        skip1 = x

        x = self.patch_merge1(x)
        y2, c2, r2, _, b2 = self.moe2(x, rmap2)
        x = self.layer2(x, imap=imap2) + y2
        skip2 = x

        x = self.patch_merge2(x)
        y3, c3, r3, _, b3 = self.moe3(x, rmap3)
        x = self.layer3(x, imap=imap3) + y3
        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        y4, c4, r4, _, b4 = self.moe4(x, rmap2)
        x = self.layer4(x, imap=None) + y4
        x = self.patch_split2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        y5, c5, r5, _, b5 = self.moe5(x, rmap1)
        x = self.layer5(x, imap=None) + y5
        x = self.patch_unembed(x)

        return x, c1 + c2 + c3 + c4 + c5, r1 + r2 + r3 + r4 + r5, b1 + b2 + b3 + b4 + b5

    def forward(self, x, imap, rmap):
        if x.shape != imap.shape or x.shape != rmap.shape:
            raise ValueError('input, imap and rmap must have the same shape')
        h, w = x.shape[-2:]
        x = self.pad(x)
        feat, class_logits, l2_reg, lb_loss = self.forward_features(x, self.pad(imap), self.pad(rmap))
        k, b = torch.split(feat, (1, 3), dim=1)
        out = k * x - b + x
        return out[:, :, :h, :w], class_logits, l2_reg, lb_loss


def build_clearnight(**kwargs):
    """ClearNight with the paper's settings; kwargs override the WDS options."""
    return ClearNight(num_labels=len(ALL_WEATHER_LABELS), **kwargs)


if __name__ == '__main__':
    net = build_clearnight(flare=True)
    t = torch.rand(1, 3, 256, 256)
    print(net(t, t, t)[0].shape, sum(p.numel() for p in net.parameters()))
