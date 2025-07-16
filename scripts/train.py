"""
Training script for self-supervised rPPG learning.
Supports single-GPU and distributed multi-GPU training.

Usage:
    python scripts/train.py --config configs/training_config.yaml
    torchrun --nproc_per_node=4 scripts/train.py --config configs/training_config.yaml

Demonstration code - no confidential information.
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.physnet import PhysNet
from src.training.distributed_utils import init_distributed_mode, cleanup, is_main_process
from src.training.trainer import SelfSupervisedTrainer
from src.data.datasets import UBFCDataset, PUREDataset, collate_fn


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Self-supervised rPPG training')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_datasets(config):
    """Setup training and validation datasets."""
    # Training dataset
    if config['data']['dataset'] == 'UBFC':
        train_dataset = UBFCDataset(
            img_root_path=config['data']['img_root_path'],
            meta_root_path=config['data']['meta_root_path'],
            T=config['data']['temporal_length'],
            resize=config['data']['resize'],
            N=config['data']['nfft']
        )
    elif config['data']['dataset'] == 'PURE':
        train_subjects = config['data']['train_subjects']
        train_dataset = PUREDataset(
            img_root_path=config['data']['img_root_path'],
            meta_root_path=config['data']['meta_root_path'],
            subjects_id=train_subjects,
            T=config['data']['temporal_length'],
            resize=config['data']['resize'],
            N=config['data']['nfft']
        )
    else:
        raise ValueError(f"Unknown dataset: {config['data']['dataset']}")
    
    # Validation dataset
    if 'val_subjects' in config['data']:
        val_subjects = config['data']['val_subjects']
        val_dataset = PUREDataset(
            img_root_path=config['data']['img_root_path'],
            meta_root_path=config['data']['meta_root_path'],
            subjects_id=val_subjects,
            T=config['data']['temporal_length'],
            resize=config['data']['resize'],
            N=config['data']['nfft']
        )
    else:
        val_dataset = None
    
    return train_dataset, val_dataset


def setup_dataloaders(train_dataset, val_dataset, config, rank, world_size):
    """Setup data loaders for training and validation."""
    # Training data loader
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    # Validation data loader
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=2,
           pin_memory=True,
           drop_last=False,
           collate_fn=collate_fn
       )
   
   return train_loader, val_loader, train_sampler


def setup_model(config, device, rank, world_size):
   """Setup model for training."""
   model = PhysNet(
       S=config['model']['spatial_dim'],
       in_ch=config['model']['input_channels']
   )
   
   model = model.to(device)
   
   # Wrap model for distributed training
   if world_size > 1:
       model = DDP(model, device_ids=[rank])
   
   return model


def setup_logging(config, rank):
   """Setup logging with Weights & Biases."""
   if config['logging']['use_wandb'] and rank == 0:
       wandb.init(
           project="rppg-self-supervised",
           name=f"experiment_{config['model']['name']}",
           config=config,
           dir=config['paths']['log_dir']
       )


def main():
   """Main training function."""
   args = parse_args()
   config = load_config(args.config)
   
   # Initialize distributed training
   rank, world_size, gpu = init_distributed_mode()
   device = torch.device(f'cuda:{gpu}')
   
   # Create output directories
   os.makedirs(args.output_dir, exist_ok=True)
   os.makedirs(config['paths']['log_dir'], exist_ok=True)
   
   # Setup logging
   setup_logging(config, rank)
   
   if is_main_process():
       print(f"Starting training with {world_size} GPUs")
       print(f"Configuration: {config}")
   
   # Setup datasets and data loaders
   train_dataset, val_dataset = setup_datasets(config)
   train_loader, val_loader, train_sampler = setup_dataloaders(
       train_dataset, val_dataset, config, rank, world_size
   )
   
   if is_main_process():
       print(f"Training dataset size: {len(train_dataset)}")
       if val_dataset:
           print(f"Validation dataset size: {len(val_dataset)}")
   
   # Setup model
   model = setup_model(config, device, rank, world_size)
   
   # Setup trainer
   trainer = SelfSupervisedTrainer(model, config, device)
   
   # Resume from checkpoint if specified
   start_epoch = 0
   if args.resume:
       start_epoch, _ = trainer.load_checkpoint(args.resume)
       if is_main_process():
           print(f"Resumed training from epoch {start_epoch}")
   
   # Training loop
   best_val_loss = float('inf')
   
   for epoch in range(start_epoch, config['training']['epochs']):
       if train_sampler and world_size > 1:
           train_sampler.set_epoch(epoch)
       
       # Training
       train_metrics = trainer.train_epoch(train_loader, epoch)
       
       # Validation
       val_metrics = None
       if val_loader:
           val_metrics = validate(model, val_loader, device, config)
       
       # Logging
       if is_main_process():
           print(f"Epoch {epoch}")
           print(f"Train - SNR: {train_metrics['snr_loss']:.4f}, "
                 f"EMD: {train_metrics['emd_loss']:.4f}, "
                 f"Contrast: {train_metrics['contrast_loss']:.4f}, "
                 f"Total: {train_metrics['total_loss']:.4f}")
           
           if val_metrics:
               print(f"Val - Loss: {val_metrics['loss']:.4f}, "
                     f"MAE: {val_metrics['mae']:.2f}")
           
           if config['logging']['use_wandb']:
               log_dict = {
                   f'train/{k}': v for k, v in train_metrics.items()
               }
               if val_metrics:
                   log_dict.update({
                       f'val/{k}': v for k, v in val_metrics.items()
                   })
               log_dict['epoch'] = epoch
               wandb.log(log_dict)
       
       # Save checkpoint
       if epoch % config['logging']['save_interval'] == 0:
           checkpoint_path = os.path.join(args.output_dir, f'epoch_{epoch}.pt')
           all_metrics = {'train': train_metrics}
           if val_metrics:
               all_metrics['val'] = val_metrics
           trainer.save_checkpoint(epoch, all_metrics, checkpoint_path)
           
           # Save best model
           if val_metrics and val_metrics['loss'] < best_val_loss:
               best_val_loss = val_metrics['loss']
               best_path = os.path.join(args.output_dir, 'best_model.pt')
               trainer.save_checkpoint(epoch, all_metrics, best_path)
   
   # Cleanup
   if world_size > 1:
       cleanup()
   
   if is_main_process() and config['logging']['use_wandb']:
       wandb.finish()


def validate(model, val_loader, device, config):
   """Validation function."""
   model.eval()
   total_loss = 0
   all_pred_hr = []
   all_true_hr = []
   num_batches = 0
   
   with torch.no_grad():
       for batch_data in val_loader:
           if batch_data is None:
               continue
               
           val_imgs, val_bvp, val_hr, fps = batch_data
           val_imgs = val_imgs.to(device)
           val_hr = val_hr.to(device)
           
           # Forward pass
           model_output = model(val_imgs)
           rppg_pred = model_output[:, -1]
           
           # Compute simple MSE loss for validation
           loss = nn.MSELoss()(rppg_pred, val_bvp.to(device))
           total_loss += loss.item()
           
           # Estimate heart rates for evaluation
           from src.utils.signal_processing import hr_fft_batch_small_Fs_internal_my
           pred_hr = hr_fft_batch_small_Fs_internal_my(
               rppg_pred.detach().cpu().numpy(),
               fps.detach().cpu().numpy(),
               config['data']['nfft']
           )
           
           all_pred_hr.extend(pred_hr)
           all_true_hr.extend(val_hr.detach().cpu().numpy())
           num_batches += 1
   
   # Calculate metrics
   avg_loss = total_loss / max(num_batches, 1)
   mae = np.mean(np.abs(np.array(all_pred_hr) - np.array(all_true_hr)))
   
   model.train()
   return {'loss': avg_loss, 'mae': mae}


if __name__ == '__main__':
   main()
