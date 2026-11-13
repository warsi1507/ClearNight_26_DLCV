# Frequency-decoupled flare suppression with a one-level Haar DWT.
# Glow is smooth, so it sits in the LL band; edges sit in LH/HL/HH and are left untouched.

import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_kernels():
    ll = torch.tensor([[1., 1.], [1., 1.]])
    lh = torch.tensor([[1., 1.], [-1., -1.]])
    hl = torch.tensor([[1., -1.], [1., -1.]])
    hh = torch.tensor([[1., -1.], [-1., 1.]])
    return 0.5 * torch.stack([ll, lh, hl, hh]).unsqueeze(1)  # (4, 1, 2, 2), orthonormal


class HaarDWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('kernels', haar_kernels(), persistent=False)

    def forward(self, x):
        b, c, h, w = x.shape
        y = F.conv2d(x, self.kernels.repeat(c, 1, 1, 1), stride=2, groups=c)
        return y.view(b, c, 4, h // 2, w // 2)  # bands: LL, LH, HL, HH

    def inverse(self, bands):
        b, c, _, h, w = bands.shape
        # orthonormal transform, so the inverse is the transposed convolution
        return F.conv_transpose2d(bands.reshape(b, 4 * c, h, w), self.kernels.repeat(c, 1, 1, 1),
                                  stride=2, groups=c)


class FlareSuppression(nn.Module):
    """LL' = LL - m * G(LL), with gate m from LL and the illumination prior."""

    def __init__(self, dim, gate_hidden=16):
        super().__init__()
        self.dwt = HaarDWT()
        self.gate = nn.Sequential(
            nn.Conv2d(dim + 1, gate_hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(gate_hidden, 1, 1),
        )
        self.glow = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        # zero last layer: the module starts as an exact identity but still gets gradients
        nn.init.zeros_(self.glow[-1].weight)
        nn.init.zeros_(self.glow[-1].bias)
        self.last_gate = None  # kept in eval mode for visualize.py

    def forward(self, x, light):
        bands = self.dwt(x)
        ll = bands[:, :, 0]
        light = F.adaptive_avg_pool2d(light, ll.shape[-2:])
        m = torch.sigmoid(self.gate(torch.cat([ll, light], dim=1)))
        ll = ll - m * self.glow(ll)
        if not self.training:
            self.last_gate = m.detach()
        return self.dwt.inverse(torch.cat([ll.unsqueeze(2), bands[:, :, 1:]], dim=2))
