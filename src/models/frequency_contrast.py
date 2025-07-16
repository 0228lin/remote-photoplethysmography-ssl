"""
Frequency contrast learning module for self-supervised rPPG training.
Implements novel frequency domain augmentation and consistency learning.

Demonstration code - no confidential information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CalculateMultiView(nn.Module):
    """Calculate multiple temporal views for contrastive learning."""
    
    def __init__(self, sub_length, num_views):
        super().__init__()
        self.num_views = num_views
        self.sub_length = sub_length
        
    def forward(self, input, zero_pad=0):
        """Generate multiple random temporal views."""
        if input.shape[-1] < self.sub_length:
            input = F.pad(input, (0, self.sub_length - input.shape[-1]))
        
        views = []
        for i in range(self.num_views):
            offset = torch.randint(
                0, input.shape[-1] - self.sub_length + 1, 
                (1,), device=input.device
            )
            x = input[..., offset:offset + self.sub_length]
            views.append(x)
        return views


class FrequencyContrast(nn.Module):
    """Frequency contrast learning wrapper around backbone model."""
    
    def __init__(self, backbone, window_size=150, num_views=2):
        super().__init__()
        self.backbone = backbone
        self.get_temp_views = CalculateMultiView(window_size, num_views)
        
    def forward(self, x_a):
        """Forward pass with frequency domain augmentation."""
        B = x_a.shape[0]
        D = x_a.shape[2]
        branches = {}

        # Random frequency resampling
        freq_factor = 1.25 + (torch.rand(1, device=x_a.device) / 4)
        target_size = int(D / freq_factor)
        
        resampler = nn.Upsample(
            size=(target_size, x_a.shape[3], x_a.shape[4]),
            mode='trilinear',
            align_corners=False
        )
        
        x_n = resampler(x_a)
        x_n = F.pad(x_n, (0, 0, 0, 0, 0, D - target_size))

        # Pass through backbone
        y_a = self.backbone(x_a).squeeze(4).squeeze(3)
        y_n = self.backbone(x_n).squeeze(4).squeeze(3)

        # Remove padding and upsample
        y_n = y_n[:, :, :target_size]
        upsampler = nn.Upsample(size=(D,), mode='linear', align_corners=False)
        y_p = upsampler(y_n)

        # Store branches
        branches['anc'] = y_a.squeeze(1)
        branches['neg'] = y_n.squeeze(1)  
        branches['pos'] = y_p.squeeze(1)

        backbone_out = branches['anc']

        # Sample random views for each branch
        for key, branch in branches.items():
            branches[key] = self.get_temp_views(branch)

        return backbone_out, branches


class FrequencyRankLoss(nn.Module):
    """Frequency ranking loss for heart rate estimation."""
    
    def __init__(self, T, model, freq_factor=5):
        super().__init__()
        self.T = T
        self.backbone = model
        self.freq_factor = freq_factor

    def forward(self, x_a):
        """Apply frequency augmentation and compute ranking loss."""
        B = x_a.shape[0]
        
        # Generate frequency factors
        freq_factor_up = 1 + (torch.rand(1, device=x_a.device) / self.freq_factor)
        freq_factor_down = 1 - (torch.rand(1, device=x_a.device) / self.freq_factor)

        # Resample inputs
        target_size_up = int(self.T * freq_factor_up)
        target_size_down = int(self.T * freq_factor_down)
        
        resampler_up = nn.Upsample(
            size=(target_size_up, x_a.shape[3], x_a.shape[4]),
            mode='trilinear', align_corners=False
        )
        resampler_down = nn.Upsample(
            size=(target_size_down, x_a.shape[3], x_a.shape[4]),
            mode='trilinear', align_corners=False
        )

        x_up = resampler_up(x_a)
        x_down = resampler_down(x_a)
        
        # Pad to common length
        sub_length = int(self.T * (1 + 1/self.freq_factor))
        x_a = F.pad(x_a, (0, 0, 0, 0, 0, sub_length - x_a.shape[2]))
        x_down = F.pad(x_down, (0, 0, 0, 0, 0, sub_length - x_down.shape[2]))
        x_up = F.pad(x_up, (0, 0, 0, 0, 0, sub_length - x_up.shape[2]))
        
        # Concatenate and process
        x_all = torch.cat((x_a, x_up, x_down), 0)
        y_all = self.backbone(x_all)[:, -1]

        # Split outputs
        x_a_len = x_a.shape[2] - (sub_length - self.T)
        x_up_len = x_up.shape[2] - (sub_length - target_size_up) 
        x_down_len = x_down.shape[2] - (sub_length - target_size_down)
        
        y_a = y_all[:B, :x_a_len]
        y_up = y_all[B:2*B, :x_up_len]
        y_down = y_all[2*B:, :x_down_len]
        
        return y_a, y_up, y_down, freq_factor_up, freq_factor_down


def hr_by_gumbel(rppg, fps, N, BP_LOW=0.6, BP_HIGH=3):
    """Estimate heart rate using Gumbel softmax attention."""
    from ..utils.signal_processing import torch_power_spectral_density
    
    freqs, psd = torch_power_spectral_density(
        x=rppg, nfft=N, fps=fps, low_hz=BP_LOW, high_hz=BP_HIGH,
        normalize=True, bandpass=True
    )
    
    psd = psd * 10000
    attention = F.gumbel_softmax(psd)
    hr = torch.sum(attention * freqs.to(rppg.device), dim=-1) * 60
    return hr
