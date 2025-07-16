"""
Loss functions for self-supervised rPPG learning.
Includes contrastive, ranking, SNR, and EMD losses.

Demonstration code - no confidential information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.signal_processing import torch_power_spectral_density


class ContrastLoss(nn.Module):
    """Contrastive loss for frequency domain consistency learning."""
    
    def __init__(self, delta_t, K, Fs, high_pass=40, low_pass=180):
        super(ContrastLoss, self).__init__()
        self.ST_sampling = STSampling(delta_t, K, Fs, high_pass, low_pass)
        self.distance_func = nn.MSELoss(reduction='mean')

    def compare_samples(self, list_a, list_b, exclude_same=False):
        """Compare samples between two lists."""
        total_distance = 0.
        M = 0
        for i in range(len(list_a)):
            for j in range(len(list_b)):
                if exclude_same and i == j:
                    continue
                total_distance += self.distance_func(list_a[i], list_b[j])
                M += 1
        return total_distance / M if M > 0 else 0.

    def forward(self, model_output):
        """Compute contrastive loss."""
        samples = self.ST_sampling(model_output)
        
        # Positive loss (same video samples)
        pos_loss = (
            self.compare_samples(samples[0], samples[0], exclude_same=True) + 
            self.compare_samples(samples[1], samples[1], exclude_same=True)
        ) / 2
        
        # Negative loss (different video samples)
        neg_loss = -self.compare_samples(samples[0], samples[1])
        
        loss = pos_loss + neg_loss
        return loss, pos_loss, neg_loss


class STSampling(nn.Module):
    """Spatiotemporal sampling for contrastive learning."""
    
    def __init__(self, delta_t, K, Fs, high_pass, low_pass):
        super().__init__()
        self.delta_t = delta_t
        self.K = K
        self.norm_psd = CalculateNormPSD(Fs, high_pass, low_pass)

    def forward(self, input):
        """Sample temporal windows and compute normalized PSD."""
        samples = []
        for b in range(input.shape[0]):
            samples_per_video = []
            for c in range(input.shape[1]):
                for i in range(self.K):
                    offset = torch.randint(
                        0, input.shape[-1] - self.delta_t + 1, 
                        (1,), device=input.device
                    )
                    segment = input[b, c, offset:offset + self.delta_t]
                    psd = self.norm_psd(segment)
                    samples_per_video.append(psd)
            samples.append(samples_per_video)
        return samples


class CalculateNormPSD(nn.Module):
    """Calculate normalized power spectral density."""
    
    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x, zero_pad=0):
        """Compute normalized PSD in frequency band."""
        x = x - torch.mean(x, dim=-1, keepdim=True)
        
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant', 0)

        # Get PSD using FFT
        x_fft = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
        x_psd = torch.add(x_fft[:, 0] ** 2, x_fft[:, 1] ** 2)

        # Filter for relevant frequencies
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, x_psd.shape[0])
        use_freqs = torch.logical_and(
            freqs >= self.high_pass / 60, 
            freqs <= self.low_pass / 60
        )
        x_filtered = x_psd[use_freqs]

        # Normalize
        x_norm = x_filtered / torch.sum(x_filtered, dim=-1, keepdim=True)
        return x_norm


class RankLoss(nn.Module):
    """Ranking loss for heart rate estimation."""
    
    def __init__(self, N=1800):
        super(RankLoss, self).__init__()
        self.N = N

    def forward(self, fs, model_output):
        """Compute ranking loss for frequency-augmented outputs."""
        y_a, y_up, y_down, freq_factor_up, freq_factor_down = model_output
        fs = fs.item()
        
        # Estimate heart rates using Gumbel softmax
        from ..models.frequency_contrast import hr_by_gumbel
        hr_a = hr_by_gumbel(y_a, fs, self.N)
        hr_up = hr_by_gumbel(y_up, fs, self.N)
        hr_down = hr_by_gumbel(y_down, fs, self.N)

        # Ranking loss
        ranking_loss = torch.nn.MarginRankingLoss()
        y = -1 * torch.ones_like(hr_a)
        
        # Expected: hr_down > hr_a > hr_up
        judge = (hr_a >= 40/freq_factor_down) & (hr_a <= 180/freq_factor_up)
        if judge.any():
            loss1 = ranking_loss(hr_up[judge], hr_a[judge], y[judge])
            loss2 = ranking_loss(hr_a[judge], hr_down[judge], y[judge])
            loss = loss1 + loss2
        else:
            loss = torch.tensor(0.0, device=hr_a.device)
        
        return loss


class SNRLoss(nn.Module):
    """Signal-to-Noise Ratio loss for signal quality."""
    
    def __init__(self, low_hz=2/3, high_hz=3.0, freq_delta=0.1):
        super().__init__()
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.freq_delta = freq_delta
        self.epsilon = 1e-10

    def forward(self, freqs, psd, device='cuda'):
        """Compute SNR loss."""
        signal_freq_idx = torch.argmax(psd, dim=1)
        signal_freq = freqs[signal_freq_idx].view(-1, 1)
        freqs_expanded = freqs.repeat(psd.shape[0], 1)
        
        low_cut = signal_freq - self.freq_delta
        high_cut = signal_freq + self.freq_delta
        band_idcs = torch.logical_and(
            freqs_expanded >= low_cut, 
            freqs_expanded <= high_cut
        ).to(device)
        
        signal_band = torch.sum(psd * band_idcs, dim=1)
        noise_band = torch.sum(psd * torch.logical_not(band_idcs), dim=1)
        denom = signal_band + noise_band + self.epsilon
        snr_loss = torch.mean(noise_band / denom)
        
        return snr_loss


class EMDLoss(nn.Module):
    """Earth Mover's Distance loss for distribution alignment."""
    
    def __init__(self, low_hz=2/3, high_hz=3.0):
        super().__init__()
        self.low_hz = low_hz
        self.high_hz = high_hz

    def forward(self, freqs, psd, device='cuda'):
        """Compute EMD loss to uniform distribution."""
        B, T = psd.shape
        psd_mean = torch.sum(psd, dim=0) / B
        expected = (torch.ones(T) / T).to(device)
        
        emd_loss = torch.mean(torch.square(
            torch.cumsum(psd_mean, dim=0) - torch.cumsum(expected, dim=0)
        ))
        
        return emd_loss
