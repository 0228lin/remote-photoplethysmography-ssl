"""
Test suite for model components.
Demonstration tests using synthetic data only.
"""

import pytest
import torch
import numpy as np
from src.models.physnet import PhysNet
from src.models.frequency_contrast import FrequencyContrast


class TestPhysNet:
    """Test suite for PhysNet model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = PhysNet(S=2, in_ch=3)
        self.batch_size = 2
        self.time_frames = 300
        self.height = 128
        self.width = 128
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.S == 2
        assert isinstance(self.model, torch.nn.Module)
        
        # Check parameter count
        param_count = sum(p.numel() for p in self.model.parameters())
        assert param_count > 0
    
    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        input_tensor = torch.randn(
            self.batch_size, 3, self.time_frames, self.height, self.width
        )
        
        output = self.model(input_tensor)
        
        # Expected output shape: (B, S²+1, T)
        expected_shape = (self.batch_size, self.model.S**2 + 1, self.time_frames)
        assert output.shape == expected_shape
    
    def test_different_input_sizes(self):
        """Test model with different input sizes."""
        test_cases = [
            (1, 3, 150, 64, 64),
            (4, 3, 450, 128, 128),
            (2, 3, 240, 96, 96)
        ]
        
        for batch, channels, time, height, width in test_cases:
            input_tensor = torch.randn(batch, channels, time, height, width)
            output = self.model(input_tensor)
            
            expected_shape = (batch, self.model.S**2 + 1, time)
            assert output.shape == expected_shape
    
    def test_gradient_flow(self):
        """Test gradient computation."""
        input_tensor = torch.randn(
            1, 3, self.time_frames, self.height, self.width, requires_grad=True
        )
        
        output = self.model(input_tensor)
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist
        assert input_tensor.grad is not None
        assert input_tensor.grad.shape == input_tensor.shape


class TestFrequencyContrast:
    """Test suite for FrequencyContrast model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        backbone = PhysNet(S=2, in_ch=3)
        self.model = FrequencyContrast(backbone, window_size=150, num_views=2)
        self.batch_size = 2
        self.time_frames = 300
    
    def test_frequency_contrast_output(self):
        """Test frequency contrast learning output."""
        input_tensor = torch.randn(self.batch_size, 3, self.time_frames, 128, 128)
        
        backbone_out, branches = self.model(input_tensor)
        
        # Check backbone output
        assert backbone_out.shape == (self.batch_size, self.time_frames)
        
        # Check branches
        assert isinstance(branches, dict)
        assert 'anc' in branches
        assert 'pos' in branches
        assert 'neg' in branches
        
        # Check each branch has correct number of views
        for branch_name, views in branches.items():
            assert len(views) == 2  # num_views


@pytest.fixture
def synthetic_rppg_signal():
    """Generate synthetic rPPG signal for testing."""
    time_frames = 300
    fps = 30
    heart_rate = 70
    
    t = np.linspace(0, time_frames/fps, time_frames)
    frequency = heart_rate / 60.0
    signal = np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.randn(time_frames)
    
    return torch.tensor(signal, dtype=torch.float32)


def test_signal_processing():
    """Test signal processing utilities."""
    from src.utils.signal_processing import hr_fft_small_Fs_internal
    
    # Generate test signal
    signal = np.sin(2 * np.pi * 1.167 * np.linspace(0, 10, 300))  # 70 BPM
    
    hr, psd, freq = hr_fft_small_Fs_internal(signal, 30, 1800)
    
    # Check reasonable heart rate estimate
    assert 60 <= hr <= 80  # Should be close to 70 BPM
    assert len(psd) > 0
    assert len(freq) > 0


if __name__ == '__main__':
    pytest.main([__file__])
