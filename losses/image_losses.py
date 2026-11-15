# Depth consistency loss (paper: L_d, weight 0.02), from the official loss/losses.py (BSD-3-Clause).
# A frozen ADDS-DepthNet predicts disparity for the output and for the ground truth; we take the L1
# difference. The official code uses the 'day' branch of the depth encoder; we keep that.

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import depth as depth_networks


class DepthConsistencyLoss(nn.Module):
    def __init__(self, encoder_path='losses/weights/encoder.pth', decoder_path='losses/weights/depth.pth'):
        super().__init__()
        self.criterion = nn.L1Loss()

        self.encoder = depth_networks.ResnetEncoder(18, False)
        loaded_enc = torch.load(encoder_path, map_location='cpu', weights_only=True)
        self.feed_height = loaded_enc['height']
        self.feed_width = loaded_enc['width']
        filtered_enc = {k: v for k, v in loaded_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_enc)
        self.encoder.eval()

        self.decoder = depth_networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        loaded_dec = torch.load(decoder_path, map_location='cpu', weights_only=True)
        self.decoder.load_state_dict(loaded_dec)
        self.decoder.eval()
        self.requires_grad_(False)

    def train(self, mode=True):
        # keep BatchNorm frozen even when the parent calls .train()
        super().train(False)
        return self

    def forward(self, x, y):
        x_in = F.interpolate(x, size=(self.feed_height, self.feed_width), mode='bilinear')
        y_in = F.interpolate(y, size=(self.feed_height, self.feed_width), mode='bilinear')

        x_disp = self.decoder(self.encoder(x_in, 'day', 'val'))[('disp', 0)]
        with torch.no_grad():
            y_disp = self.decoder(self.encoder(y_in, 'day', 'val'))[('disp', 0)]

        return self.criterion(x_disp, y_disp)
