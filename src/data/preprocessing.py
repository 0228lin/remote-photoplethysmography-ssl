"""
Preprocessing utilities for face detection and signal processing.
Includes privacy-preserving techniques for healthcare data.

Demonstration code - no confidential information.
"""

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
import face_alignment
from scipy.signal import butter, filtfilt, resample


class FacePreprocessor:
    """Face detection and preprocessing for rPPG."""
    
    def __init__(self, device='cuda', crop_size=(128, 128)):
        self.device = device
        self.crop_size = crop_size
        self.mtcnn = MTCNN(device=device)
        self.face_alignment = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, 
            flip_input=False
        )

    def detect_face(self, image):
        """Detect face bounding box using MTCNN."""
        result = self.mtcnn.detect(image)
        
        if result[0] is None:
            # Default bounding box if no face detected
            h, w = image.shape[:2]
            return [w//4, h//4, 3*w//4, 3*h//4]
        else:
            return result[0][0]

    def crop_face(self, image, bounding_box=None):
        """Crop face from image."""
        if bounding_box is None:
            bounding_box = self.detect_face(image)
            
        x1, y1, x2, y2 = [max(0, int(coord)) for coord in bounding_box]
        crop_img = image[y1:y2, x1:x2]
        
        if crop_img.size == 0:
            # Fallback to center crop
            h, w = image.shape[:2]
            crop_img = image[h//4:3*h//4, w//4:3*w//4]
            
        return cv2.resize(crop_img, self.crop_size)

    def extract_landmarks(self, image):
        """Extract facial landmarks."""
        landmarks = self.face_alignment.get_landmarks(image)
        return landmarks[0] if landmarks else None

    def anonymize_face(self, image, landmarks=None):
        """
        Apply anonymization techniques for privacy protection.
        This is a demonstration - real implementation would be more sophisticated.
        """
        if landmarks is None:
            landmarks = self.extract_landmarks(image)
            
        if landmarks is not None:
            # Simple anonymization: blur eye region
            eye_region = self.get_eye_region(landmarks)
            if eye_region is not None:
                image = self.blur_region(image, eye_region)
                
        return image

    def get_eye_region(self, landmarks):
        """Get eye region from landmarks."""
        # Eye landmarks (approximate indices)
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        all_eyes = np.concatenate([left_eye, right_eye])
        x_min, y_min = np.min(all_eyes, axis=0).astype(int)
        x_max, y_max = np.max(all_eyes, axis=0).astype(int)
        
        return (x_min, y_min, x_max, y_max)

    def blur_region(self, image, region):
        """Blur specified region for anonymization."""
        x_min, y_min, x_max, y_max = region
        roi = image[y_min:y_max, x_min:x_max]
        blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)
        image[y_min:y_max, x_min:x_max] = blurred_roi
        return image

    def process_video_frames(self, video_path, output_path=None, anonymize=False):
        """Process all frames in a video."""
        cap = cv2.VideoCapture(video_path)
        processed_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if anonymize:
                frame_rgb = self.anonymize_face(frame_rgb)
                
            cropped_face = self.crop_face(frame_rgb)
            processed_frames.append(cropped_face)
            
        cap.release()
        
        if output_path:
            np.save(output_path, np.array(processed_frames))
            
        return np.array(processed_frames)


class SignalPreprocessor:
    """Signal preprocessing utilities for physiological signals."""
    
    def __init__(self, fs=30):
        self.fs = fs

    def butter_bandpass_filter(self, signal, lowcut=0.6, highcut=4.0, order=2):
        """Apply butterworth bandpass filter."""
        signal = np.reshape(signal, -1)
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def resample_signal(self, signal, target_fs):
        """Resample signal to target frequency."""
        if self.fs == target_fs:
            return signal
        
        num_samples = int(len(signal) * target_fs / self.fs)
        return resample(signal, num_samples)

    def normalize_signal(self, signal):
        """Normalize signal to zero mean and unit variance."""
        signal = np.array(signal)
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    def remove_artifacts(self, signal, threshold=3.0):
        """Remove artifacts using z-score thresholding."""
        signal = np.array(signal)
        z_scores = np.abs((signal - np.mean(signal)) / np.std(signal))
        
        # Replace outliers with median
        median_val = np.median(signal)
        signal[z_scores > threshold] = median_val
        
        return signal

    def segment_signal(self, signal, segment_length, overlap=0.5):
        """Segment signal into overlapping windows."""
        signal = np.array(signal)
        step = int(segment_length * (1 - overlap))
        segments = []
        
        for start in range(0, len(signal) - segment_length + 1, step):
            end = start + segment_length
            segments.append(signal[start:end])
            
        return np.array(segments)

    def quality_assessment(self, signal):
        """Assess signal quality using multiple metrics."""
        signal = np.array(signal)
        
        # Signal-to-noise ratio estimate
        signal_power = np.mean(signal ** 2)
        noise_power = np.var(np.diff(signal))
        snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
        
        # Kurtosis (measure of signal peakedness)
        kurtosis = self.calculate_kurtosis(signal)
        
        # Percentage of signal in physiological range
        valid_range = self.calculate_valid_range(signal)
        
        return {
            'snr': snr,
            'kurtosis': kurtosis,
            'valid_range': valid_range,
            'quality_score': (snr + valid_range) / 2  # Simple composite score
        }

    def calculate_kurtosis(self, signal):
        """Calculate kurtosis of signal."""
        mean = np.mean(signal)
        std = np.std(signal)
        normalized = (signal - mean) / std
        return np.mean(normalized ** 4) - 3

    def calculate_valid_range(self, signal):
        """Calculate percentage of signal in valid physiological range."""
        # Normalize to [0, 1] range
        signal_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)
        # Count values in reasonable range (exclude extreme outliers)
        valid_count = np.sum((signal_norm > 0.05) & (signal_norm < 0.95))
        return (valid_count / len(signal)) * 100


class DataAugmentor:
    """Data augmentation for rPPG training."""
    
    def __init__(self):
        pass

    def temporal_shift(self, frames, max_shift=10):
        """Apply random temporal shift."""
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift > 0:
            return frames[:, shift:, :, :]
        elif shift < 0:
            return frames[:, :shift, :, :]
        return frames

    def frequency_augmentation(self, signal, factor_range=(0.8, 1.2)):
        """Apply frequency domain augmentation."""
        factor = np.random.uniform(*factor_range)
        target_length = int(len(signal) * factor)
        resampled = resample(signal, target_length)
        
        # Pad or truncate to original length
        if len(resampled) > len(signal):
            return resampled[:len(signal)]
        else:
            padded = np.zeros(len(signal))
            padded[:len(resampled)] = resampled
            return padded

    def spatial_augmentation(self, frames):
        """Apply spatial augmentation to frames."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            frames = frames[:, :, :, ::-1]
            
        return frames
