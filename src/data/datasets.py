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
        meta_path = os.path.join(self.meta
