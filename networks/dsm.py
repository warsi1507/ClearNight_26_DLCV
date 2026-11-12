# Dynamic selection module, based on networks/DSM.py from the official ClearNight code (BSD-3-Clause).
# Changes: hyperparameters are arguments, routing noise only in training, units not zero-initialised.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, zero_init=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        if zero_init:
            nn.init.zeros_(self.conv.weight)
        else:
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity='linear')
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class TopKRouter(nn.Module):
    """Shared MLP on the pooled feature with two heads: unit scores and weather logits."""

    def __init__(self, in_channels, num_units, top_k, num_labels, hidden_size=128,
                 noise_std=0.01, l2_lambda=0.01):
        super().__init__()
        if not 1 <= top_k <= num_units:
            raise ValueError('need 1 <= top_k <= num_units')
        self.num_units = num_units
        self.top_k = top_k
        self.noise_std = noise_std
        self.l2_lambda = l2_lambda

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp_shared = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
        )
        self.unit_head = nn.Linear(hidden_size // 2, num_units)
        self.classification_head = nn.Linear(hidden_size // 2, num_labels)

        for layer in list(self.mlp_shared) + [self.unit_head, self.classification_head]:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        shared = self.mlp_shared(self.pool(x).flatten(1))
        unit_logits = self.unit_head(shared)
        class_logits = self.classification_head(shared)
        # official code: 0.01 * sum of router parameter norms is added to the loss
        l2_reg = self.l2_lambda * sum(p.norm(2) for p in self.parameters())
        return unit_logits, class_logits, l2_reg


def top_k_softmax(logits, k):
    top_idx = logits.topk(k, dim=1).indices
    mask = torch.full_like(logits, float('-inf')).scatter(1, top_idx, 0.0)
    return F.softmax(logits + mask, dim=1), top_idx


def load_balance_loss(weights):
    # variance of the average unit usage (official compute_load_balancing_loss)
    usage = weights.mean(dim=0)
    return ((usage - usage.mean()) ** 2).mean()


class MixtureOfUnits(nn.Module):
    """M(z) = sum_k w_k * U_k(z) over the top-k units chosen per image."""

    def __init__(self, channels, num_units, top_k, num_labels, noise_std=0.01, l2_lambda=0.01,
                 zero_init_units=False):
        super().__init__()
        self.top_k = top_k
        self.noise_std = noise_std
        self.router = TopKRouter(channels, num_units, top_k, num_labels,
                                 noise_std=noise_std, l2_lambda=l2_lambda)
        self.units = nn.ModuleList([Unit(channels, channels, zero_init=zero_init_units)
                                    for _ in range(num_units)])

    def forward(self, z):
        logits, class_logits, l2_reg = self.router(z)  # (B, N)
        if self.training and self.noise_std > 0:
            logits = logits + self.noise_std * torch.randn_like(logits)

        weights, top_idx = top_k_softmax(logits, self.top_k)
        unit_out = torch.stack([unit(z) for unit in self.units], dim=1)  # (B, N, C, h, w)
        out = (weights[:, :, None, None, None] * unit_out).sum(dim=1)
        return out, class_logits, l2_reg, top_idx, load_balance_loss(weights)
