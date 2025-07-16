"""
Self-supervised trainer for remote photoplethysmography.
Implements distributed training with multiple loss functions.

Demonstration code - no confidential information.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from tqdm import tqdm

from .losses import ContrastLoss, SNRLoss, EMDLoss, RankLoss
from .distributed_utils import is_main_process, get_rank
from ..utils.signal_processing import torch_power_spectral_density


class SelfSupervisedTrainer:
    """Self-supervised trainer for rPPG models."""
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.rank = get_rank()
        
        # Initialize losses
        self.setup_losses()
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['training']['learning_rate']
        )
        
        # Metrics tracking
        self.train_metrics = {
            'snr_loss': 0.0,
            'emd_loss': 0.0,
            'contrast_loss': 0.0,
            'total_loss': 0.0
        }

    def setup_losses(self):
        """Initialize loss functions."""
        # Contrast loss for multi-view learning
        delta_t = self.config['data']['temporal_length'] // 2
        K = 4
        self.contrast_loss = ContrastLoss(
            delta_t=delta_t, 
            K=K, 
            Fs=30, 
            high_pass=40, 
            low_pass=180
        )
        
        # SNR and EMD losses for signal quality
        self.snr_loss = SNRLoss()
        self.emd_loss = EMDLoss()
        
        # Loss weights
        self.weights = {
            'snr': self.config['training']['snr_weight'],
            'emd': self.config['training']['emd_weight'],
            'contrast': self.config['training']['contrast_weight']
        }

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {key: 0.0 for key in self.train_metrics.keys()}
        num_batches = 0
        
        if is_main_process():
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        else:
            pbar = dataloader
            
        for batch_idx, batch_data in enumerate(pbar):
            if batch_data is None:
                continue
                
            # Unpack batch
            if len(batch_data) == 4:
                train_imgs, train_bvp, train_hr, fps = batch_data
            else:
                continue
                
            train_imgs = train_imgs.to(self.device)
            batch_size = train_imgs.shape[0]
            
            # Forward pass
            model_output = self.model(train_imgs)
            rppg_pred = model_output[:, -1]
            
            # Compute losses
            total_loss, losses = self.compute_losses(rppg_pred, model_output)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            for key, value in losses.items():
                epoch_metrics[key] += value.item()
            epoch_metrics['total_loss'] += total_loss.item()
            num_batches += 1
            
            # Log batch metrics
            if is_main_process() and batch_idx % self.config['logging']['log_interval'] == 0:
                self.log_batch_metrics(losses, total_loss, epoch, batch_idx)
                
        # Average metrics over epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)
            
        return epoch_metrics

    def compute_losses(self, rppg_pred, model_output):
        """Compute all loss components."""
        losses = {}
        
        # Frequency domain analysis
        freqs, psd = torch_power_spectral_density(
            rppg_pred, 
            nfft=5400, 
            fps=30, 
            low_hz=2/3, 
            high_hz=3.0,
            normalize=True, 
            bandpass=True
        )
        freqs = freqs.to(self.device)
        
        # SNR loss
        losses['snr_loss'] = self.snr_loss(freqs, psd, self.device)
        
        # EMD loss  
        losses['emd_loss'] = self.emd_loss(freqs, psd, self.device)
        
        # Contrastive loss
        contrast_loss, _, _ = self.contrast_loss(model_output)
        losses['contrast_loss'] = contrast_loss
        
        # Total loss
        total_loss = (
            self.weights['snr'] * losses['snr_loss'] +
            self.weights['emd'] * losses['emd_loss'] +
            self.weights['contrast'] * losses['contrast_loss']
        )
        
        return total_loss, losses

    def log_batch_metrics(self, losses, total_loss, epoch, batch_idx):
        """Log metrics for current batch."""
        if self.config['logging']['use_wandb']:
            wandb.log({
                'batch/snr_loss': losses['snr_loss'].item(),
                'batch/emd_loss': losses['emd_loss'].item(),
                'batch/contrast_loss': losses['contrast_loss'].item(),
                'batch/total_loss': total_loss.item(),
                'epoch': epoch,
                'batch': batch_idx
            })

    def save_checkpoint(self, epoch, metrics, save_path):
        """Save model checkpoint."""
        if is_main_process():
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics,
                'config': self.config
            }
            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']


class AvgrageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
