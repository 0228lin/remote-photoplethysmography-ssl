"""
Dataset classes for remote photoplethysmography.
Supports UBFC, PURE, and custom datasets.

Demonstration code - no confidential information.
"""

import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch.nn.functional as F
from ..utils.signal_processing import hr_fft_small_Fs_internal


class RPPGDataset(Dataset):
    """Base dataset class for rPPG data."""
    
    def __init__(self, img_root_path, meta_root_path, T=300, resize=128, N=1800):
        self.T = T  # Temporal length
        self.img_root_path = img_root_path
        self.meta_root_path = meta_root_path
        self.resize = resize
        self.N = N  # FFT resolution
        self.file_len = len(os.listdir(meta_root_path))

    def __len__(self):
        return self.file_len

    def normalize_frames(self, frames):
        """Normalize frame values to [-1, 1]."""
        return (frames - 127.5) / 128

    def load_and_resize_frames(self, img_folder, img_len):
        """Load and resize frames from image folder."""
        imgs = os.listdir(img_folder)
        imgs.sort()
        frames = np.zeros((img_len, self.resize, self.resize, 3)).astype('float32')
        
        for j in range(img_len):
            frame_name = imgs[j]
            frame_path = os.path.join(img_folder, frame_name)
            frame = cv2.resize(
                cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB),
                (self.resize, self.resize)
            )
            frames[j, :] = self.normalize_frames(frame)
            
        return frames


class UBFCDataset(RPPGDataset):
    """UBFC-rPPG dataset implementation."""
    
    def __init__(self, img_root_path, meta_root_path, T=300, resize=128, N=1800):
        super().__init__(img_root_path, meta_root_path, T, resize, N)

    def __getitem__(self, idx):
        files = os.listdir(self.meta_root_path)
        files.sort()
        file = files[idx]
        meta_path = os.path.join(self.meta_root_path, file)
       
       # Extract subject and video info
       subject = file.split('.')[0].replace('subject', '')
       img_folder = os.path.join(self.img_root_path, f'subject{subject}')
       
       if not os.path.exists(img_folder):
           return None
           
       # Load metadata
       data = np.load(meta_path)
       bvp = data['wave'].astype('float32')
       fps = data['fps_cal'].astype('float32')
       
       # Load frames
       img_len = len(os.listdir(img_folder))
       frames = self.load_and_resize_frames(img_folder, min(img_len, len(bvp)))
       
       # Temporal resampling to 30fps
       img_seq = torch.tensor(frames).permute(1, 2, 3, 0)
       if fps != 30:
           resample = F.interpolate(img_seq, scale_factor=(1, 30/fps))
       else:
           resample = img_seq
           
       if resample.shape[3] >= self.T:
           # Estimate heart rate
           hr_cal, _, _ = hr_fft_small_Fs_internal(bvp[:self.T], 30, self.N)
           
           if 40 <= hr_cal < 180:
               new_frames = resample.permute(2, 3, 0, 1)[:, :self.T]
               bvp_segment = bvp[:self.T]
               return new_frames, bvp_segment, hr_cal, fps
               
       return None


class PUREDataset(RPPGDataset):
   """PURE dataset implementation."""
   
   def __init__(self, img_root_path, meta_root_path, subjects_id, T=300, resize=128, N=1800):
       self.subjects_id = subjects_id
       self.all_files = [f for f in os.listdir(meta_root_path) if f.endswith('.npz')]
       self.files = [f for f in self.all_files 
                    if int(f.split('-')[0]) in subjects_id]
       self.file_len = len(self.files)
       super().__init__(img_root_path, meta_root_path, T, resize, N)

   def __len__(self):
       return self.file_len

   def __getitem__(self, idx):
       self.files.sort()
       file = self.files[idx]
       meta_path = os.path.join(self.meta_root_path, file)
       
       # Extract subject and video info
       sub = file.split('-')[0]
       vid = file.split('-')[1].split('.')[0]
       img_folder = os.path.join(self.img_root_path, f'{sub}-{vid}')
       
       if not os.path.exists(img_folder):
           return None
           
       # Load metadata
       data = np.load(meta_path)
       bvp = data['wave'].astype('float32')
       fps = data['fps_cal'].astype('float32')
       hr_gt = np.mean(data['hr'].astype('float32')[:self.T])
       
       # Load frames
       img_len = len(os.listdir(img_folder))
       frames = self.load_and_resize_frames(img_folder, img_len)
       
       # Process frames
       img_seq = torch.tensor(frames).permute(1, 2, 3, 0)
       
       if fps != 30:
           resample = F.interpolate(img_seq, scale_factor=(1, 30/fps))
           if resample.shape[3] >= self.T:
               new_frames = resample.permute(2, 3, 0, 1)[:, :self.T]
               bvp_segment = bvp[:self.T]
               return new_frames, bvp_segment, hr_gt, fps
       else:
           if img_seq.shape[3] >= self.T:
               new_frames = img_seq.permute(2, 3, 0, 1)[:, :self.T]
               bvp_segment = bvp[:self.T]
               return new_frames, bvp_segment, hr_gt, fps
               
       return None


class CustomDataset(RPPGDataset):
   """Custom dataset for healthcare data (anonymized)."""
   
   def __init__(self, img_root_path, meta_root_path, T=300, resize=128, N=1800):
       super().__init__(img_root_path, meta_root_path, T, resize, N)

   def __getitem__(self, idx):
       """
       Load anonymized healthcare data.
       Note: This is demonstration code - no real data included.
       """
       files = os.listdir(self.meta_root_path)
       files.sort()
       file = files[idx]
       
       # Anonymized processing would go here
       # All sensitive information removed
       
       return None  # Placeholder for demonstration


def collate_fn(batch):
   """Custom collate function to handle None values."""
   batch = [item for item in batch if item is not None]
   if len(batch) == 0:
       return None
   return torch.utils.data.dataloader.default_collate(batch)
