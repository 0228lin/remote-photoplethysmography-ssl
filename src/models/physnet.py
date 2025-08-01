"""
PhysNet model implementation for remote photoplethysmography.
Based on: "Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks"
Demonstration code - no confidential information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysNet(nn.Module):
    def __init__(self, S=2, in_ch=3):
        """
        PhysNet model for rPPG signal extraction.
        
        Args:
            S (int): Spatial dimension of ST-rPPG block
            in_ch (int): Number of input channels (3 for RGB)
        """
        super().__init__()
        self.S = S

        self.start = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=32, kernel_size=(1, 5, 5), 
                     stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU()
        )

        # Encoder layers
        self.loop1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), 
                     stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), 
                     stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        self.encoder1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), 
                     stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), 
                     stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.encoder2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), 
                     stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), 
                     stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        self.loop4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), 
                     stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), 
                     stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # Decoder layers
        self.decoder1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), 
                     stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.decoder2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), 
                     stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        self.end = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, S, S)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), 
                     stride=1, padding=(0, 0, 0))
        )

    def forward(self, x):
        """Forward pass of PhysNet."""
        # Normalize input
        means = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        stds = torch.std(x, dim=(2, 3, 4), keepdim=True)
        x = (x - means) / stds

        parity = []
        x = self.start(x)
        x = self.loop1(x)
        parity.append(x.size(2) % 2)
        x = self.encoder1(x)
        parity.append(x.size(2) % 2)
        x = self.encoder2(x)
        x = self.loop4(x)

        # Decoder with upsampling
        x = F.interpolate(x, scale_factor=(2, 1, 1))
        x = self.decoder1(x)
        x = F.pad(x, (0, 0, 0, 0, 0, parity[-1]), mode='replicate')
        x = F.interpolate(x, scale_factor=(2, 1, 1))
        x = self.decoder2(x)
        x = F.pad(x, (0, 0, 0, 0, 0, parity[-2]), mode='replicate')
        x = self.end(x)

        # Extract spatial signals
        x_list = []
        for a in range(self.S):
            for b in range(self.S):
                x_list.append(x[:, :, :, a, b])

        x = sum(x_list) / (self.S * self.S)
        X = torch.cat(x_list + [x], 1)
        return X
