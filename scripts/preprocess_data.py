"""
Data preprocessing script for rPPG datasets.
Handles face detection, cropping, and signal preprocessing.

Usage:
    python scripts/preprocess_data.py --dataset UBFC --input_path /path/to/ubfc --output_path /path/to/processed
    python scripts/preprocess_data.py --dataset PURE --input_path /path/to/pure --output_path /path/to/processed

Demonstration code - no confidential information.
"""

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import glob
import json
from multiprocessing import Pool
import functools

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import FacePreprocessor, SignalPreprocessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess rPPG datasets')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['UBFC', 'PURE', 'CUSTOM'],
                       help='Dataset to preprocess')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to raw dataset')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save processed data')
    parser.add_argument('--crop_size', type=int, default=128,
                       help='Face crop size')
    parser.add_argument('--target_fps', type=int, default=30,
                       help='Target frame rate')
    parser.add_argument('--anonymize', action='store_true',
                       help='Apply anonymization (for healthcare data)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of worker processes')
    parser.add_argument('--quality_threshold', type=float, default=0.5,
                       help='Quality threshold for filtering')
    return parser.parse_args()


class UBFCPreprocessor:
    """Preprocessor for UBFC-rPPG dataset."""
    
    def __init__(self, input_path, output_path, crop_size=128, target_fps=30, anonymize=False):
        self.input_path = input_path
        self.output_path = output_path
        self.crop_size = crop_size
        self.target_fps = target_fps
        self.anonymize = anonymize
        
        self.face_processor = FacePreprocessor(crop_size=(crop_size, crop_size))
        self.signal_processor = SignalPreprocessor(fs=target_fps)
        
        # Create output directories
        os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'signals'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'metadata'), exist_ok=True)

    def process_subject(self, subject_dir):
        """Process a single subject."""
        subject_name = os.path.basename(subject_dir)
        
        try:
            # Find video file
            video_files = glob.glob(os.path.join(subject_dir, '*.avi'))
            if not video_files:
                print(f"No video file found for {subject_name}")
                return False
            
            video_path = video_files[0]
            
            # Find ground truth file
            gt_files = glob.glob(os.path.join(subject_dir, '*.txt'))
            if not gt_files:
                print(f"No ground truth file found for {subject_name}")
                return False
            
            gt_path = gt_files[0]
            
            # Process video
            frames = self.process_video(video_path)
            if frames is None:
                return False
            
            # Process ground truth
            bvp_signal, fps = self.process_ground_truth(gt_path)
            if bvp_signal is None:
                return False
            
            # Resample signal to match target fps
            if fps != self.target_fps:
                bvp_signal = self.signal_processor.resample_signal(bvp_signal, self.target_fps)
            
            # Apply signal preprocessing
            bvp_signal = self.signal_processor.butter_bandpass_filter(bvp_signal)
            bvp_signal = self.signal_processor.normalize_signal(bvp_signal)
            
            # Quality assessment
            quality_metrics = self.signal_processor.quality_assessment(bvp_signal)
            
            # Align temporal dimensions
            min_length = min(len(frames), len(bvp_signal))
            frames = frames[:min_length]
            bvp_signal = bvp_signal[:min_length]
            
            # Save processed data
            self.save_processed_data(subject_name, frames, bvp_signal, quality_metrics)
            
            return True
            
        except Exception as e:
            print(f"Error processing {subject_name}: {e}")
            return False

    def process_video(self, video_path):
        """Process video file and extract face crops."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return None
        
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for _ in tqdm(range(frame_count), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply anonymization if requested
            if self.anonymize:
                frame_rgb = self.face_processor.anonymize_face(frame_rgb)
            
            # Crop face
            face_crop = self.face_processor.crop_face(frame_rgb)
            frames.append(face_crop)
        
        cap.release()
        return np.array(frames)

    def process_ground_truth(self, gt_path):
        """Process ground truth physiological signal."""
        try:
            # UBFC ground truth format: time, signal_value
            data = np.loadtxt(gt_path)
            
            if data.ndim == 2:
                timestamps = data[:, 0]
                signal = data[:, 1]
                
                # Calculate sampling rate
                fps = 1.0 / np.mean(np.diff(timestamps))
            else:
                # If only signal values
                signal = data
                fps = 30.0  # Default assumption
            
            return signal, fps
            
        except Exception as e:
            print(f"Error processing ground truth {gt_path}: {e}")
            return None, None

    def save_processed_data(self, subject_name, frames, bvp_signal, quality_metrics):
        """Save processed data to files."""
        # Save frames
        frames_path = os.path.join(self.output_path, 'images', f'{subject_name}.npz')
        np.savez_compressed(frames_path, frames=frames)
        
        # Save signal
        signal_path = os.path.join(self.output_path, 'signals', f'{subject_name}.npz')
        np.savez_compressed(signal_path, 
                           wave=bvp_signal,
                           fps_cal=self.target_fps)
        
        # Save metadata
        metadata = {
            'subject': subject_name,
            'num_frames': len(frames),
            'signal_length': len(bvp_signal),
            'fps': self.target_fps,
            'crop_size': self.crop_size,
            'quality_metrics': quality_metrics
        }
        
        metadata_path = os.path.join(self.output_path, 'metadata', f'{subject_name}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def process_dataset(self):
        """Process entire UBFC dataset."""
        subject_dirs = [d for d in glob.glob(os.path.join(self.input_path, '*'))
                       if os.path.isdir(d)]
        
        successful = 0
        for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
            if self.process_subject(subject_dir):
                successful += 1
        
        print(f"Successfully processed {successful}/{len(subject_dirs)} subjects")


class PUREPreprocessor:
    """Preprocessor for PURE dataset."""
    
    def __init__(self, input_path, output_path, crop_size=128, target_fps=30, anonymize=False):
        self.input_path = input_path
        self.output_path = output_path
        self.crop_size = crop_size
        self.target_fps = target_fps
        self.anonymize = anonymize
        
        self.face_processor = FacePreprocessor(crop_size=(crop_size, crop_size))
        self.signal_processor = SignalPreprocessor(fs=target_fps)
        
        # Create output directories
        os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'signals'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'metadata'), exist_ok=True)

    def process_session(self, session_dir):
        """Process a single session."""
        session_name = os.path.basename(session_dir)
        
        try:
            # Find image files
            image_files = sorted(glob.glob(os.path.join(session_dir, '*.png')))
            if not image_files:
                print(f"No image files found for {session_name}")
                return False
            
            # Find pulse data
            pulse_file = os.path.join(session_dir, 'pulse.json')
            if not os.path.exists(pulse_file):
                print(f"No pulse file found for {session_name}")
                return False
            
            # Process images
            frames = self.process_images(image_files)
            if frames is None:
                return False
            
            # Process pulse data
            bvp_signal, hr_signal = self.process_pulse_data(pulse_file)
            if bvp_signal is None:
                return False
            
            # Apply signal preprocessing
            bvp_signal = self.signal_processor.butter_bandpass_filter(bvp_signal)
            bvp_signal = self.signal_processor.normalize_signal(bvp_signal)
            
            # Quality assessment
            quality_metrics = self.signal_processor.quality_assessment(bvp_signal)
            
            # Align temporal dimensions
            min_length = min(len(frames), len(bvp_signal))
            frames = frames[:min_length]
            bvp_signal = bvp_signal[:min_length]
            hr_signal = hr_signal[:min_length] if hr_signal is not None else None
            
            # Save processed data
            self.save_processed_data(session_name, frames, bvp_signal, hr_signal, quality_metrics)
            
            return True
            
        except Exception as e:
            print(f"Error processing {session_name}: {e}")
            return False

    def process_images(self, image_files):
        """Process image files and extract face crops."""
        frames = []
        
        for img_path in tqdm(image_files, desc="Processing images"):
            # Load image
            frame = cv2.imread(img_path)
            if frame is None:
                continue
                
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply anonymization if requested
            if self.anonymize:
                frame_rgb = self.face_processor.anonymize_face(frame_rgb)
            
            # Crop face
            face_crop = self.face_processor.crop_face(frame_rgb)
            frames.append(face_crop)
        
        return np.array(frames)

    def process_pulse_data(self, pulse_file):
        """Process pulse data from JSON file."""
        try:
            with open(pulse_file, 'r') as f:
                pulse_data = json.load(f)
            
            # Extract BVP signal
            bvp_signal = np.array(pulse_data.get('/FullPackage', []))
            
            # Extract HR signal if available
            hr_signal = None
            if '/FullPackage/HR' in pulse_data:
                hr_signal = np.array(pulse_data['/FullPackage/HR'])
            
            return bvp_signal, hr_signal
            
        except Exception as e:
            print(f"Error processing pulse data {pulse_file}: {e}")
            return None, None

    def save_processed_data(self, session_name, frames, bvp_signal, hr_signal, quality_metrics):
        """Save processed data to files."""
        # Save frames
        frames_path = os.path.join(self.output_path, 'images', f'{session_name}.npz')
        np.savez_compressed(frames_path, frames=frames)
        
        # Save signal
        signal_data = {
            'wave': bvp_signal,
            'fps_cal': self.target_fps
        }
        if hr_signal is not None:
            signal_data['hr'] = hr_signal
            
        signal_path = os.path.join(self.output_path, 'signals', f'{session_name}.npz')
        np.savez_compressed(signal_path, **signal_data)
        
        # Save metadata
        metadata = {
            'session': session_name,
            'num_frames': len(frames),
            'signal_length': len(bvp_signal),
            'fps': self.target_fps,
            'crop_size': self.crop_size,
            'quality_metrics': quality_metrics,
            'has_hr_signal': hr_signal is not None
        }
        
        metadata_path = os.path.join(self.output_path, 'metadata', f'{session_name}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def process_dataset(self):
        """Process entire PURE dataset."""
        session_dirs = [d for d in glob.glob(os.path.join(self.input_path, '*'))
                       if os.path.isdir(d)]
        
        successful = 0
        for session_dir in tqdm(session_dirs, desc="Processing sessions"):
            if self.process_session(session_dir):
                successful += 1
        
        print(f"Successfully processed {successful}/{len(session_dirs)} sessions")


def filter_by_quality(output_path, quality_threshold):
    """Filter processed data by quality metrics."""
    metadata_dir = os.path.join(output_path, 'metadata')
    high_quality_subjects = []
    
    for metadata_file in os.listdir(metadata_dir):
        if not metadata_file.endswith('.json'):
            continue
            
        with open(os.path.join(metadata_dir, metadata_file), 'r') as f:
            metadata = json.load(f)
        
        quality_score = metadata['quality_metrics']['quality_score']
        if quality_score >= quality_threshold:
            high_quality_subjects.append(metadata_file.replace('.json', ''))
    
    # Save high quality subject list
    quality_list_path = os.path.join(output_path, 'high_quality_subjects.txt')
    with open(quality_list_path, 'w') as f:
        for subject in high_quality_subjects:
            f.write(f"{subject}\n")
    
    print(f"Found {len(high_quality_subjects)} high-quality subjects")
    print(f"Quality list saved to {quality_list_path}")


def main():
    """Main preprocessing function."""
    args = parse_args()
    
    print(f"Preprocessing {args.dataset} dataset")
    print(f"Input path: {args.input_path}")
    print(f"Output path: {args.output_path}")
    print(f"Crop size: {args.crop_size}")
    print(f"Target FPS: {args.target_fps}")
    print(f"Anonymize: {args.anonymize}")
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Initialize preprocessor
    if args.dataset == 'UBFC':
        preprocessor = UBFCPreprocessor(
            args.input_path, args.output_path,
            crop_size=args.crop_size,
            target_fps=args.target_fps,
            anonymize=args.anonymize
        )
    elif args.dataset == 'PURE':
        preprocessor = PUREPreprocessor(
            args.input_path, args.output_path,
            crop_size=args.crop_size,
            target_fps=args.target_fps,
            anonymize=args.anonymize
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Process dataset
    preprocessor.process_dataset()
    
    # Filter by quality
    filter_by_quality(args.output_path, args.quality_threshold)
    
    print("Preprocessing completed!")


if __name__ == '__main__':
    main()
