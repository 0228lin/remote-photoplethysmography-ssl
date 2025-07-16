"""
Demonstration script showing model capabilities with synthetic data.
This demo uses NO real data and is for technical demonstration only.

IMPORTANT: This is demonstration code only - not for practical use.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.physnet import PhysNet
from src.models.frequency_contrast import FrequencyContrast
from src.utils.signal_processing import hr_fft_small_Fs_internal, torch_power_spectral_density


def generate_synthetic_video(batch_size=2, channels=3, time_frames=300, height=128, width=128):
    """Generate synthetic video data for demonstration."""
    print(" Generating synthetic video data...")
    
    # Create realistic-looking synthetic video with some temporal patterns
    video = torch.randn(batch_size, channels, time_frames, height, width)
    
    # Add some temporal consistency (simulate face-like patterns)
    for t in range(1, time_frames):
        video[:, :, t] = 0.8 * video[:, :, t-1] + 0.2 * video[:, :, t]
    
    # Normalize to image range
    video = (video - video.min()) / (video.max() - video.min())
    video = video * 255.0
    
    print(f" Generated synthetic video: {video.shape}")
    return video


def generate_synthetic_bvp(batch_size=2, time_frames=300, heart_rates=[65, 80]):
    """Generate synthetic BVP signals for demonstration."""
    print(" Generating synthetic BVP signals...")
    
    bvp_signals = []
    fps = 30
    
    for i in range(batch_size):
        hr = heart_rates[i] if i < len(heart_rates) else 70
        
        # Generate synthetic BVP signal
        t = np.linspace(0, time_frames/fps, time_frames)
        frequency = hr / 60.0  # Convert to Hz
        
        # Create realistic BVP waveform (combination of sine waves)
        signal = (np.sin(2 * np.pi * frequency * t) + 
                 0.3 * np.sin(2 * np.pi * frequency * 2 * t) +
                 0.1 * np.random.randn(time_frames))
        
        # Add some noise and filtering
        signal = signal + 0.05 * np.random.randn(time_frames)
        bvp_signals.append(signal)
    
    bvp_tensor = torch.tensor(np.array(bvp_signals), dtype=torch.float32)
    hr_tensor = torch.tensor(heart_rates[:batch_size], dtype=torch.float32)
    
    print(f" Generated synthetic BVP signals: {bvp_tensor.shape}")
    print(f" Target heart rates: {heart_rates[:batch_size]} BPM")
    
    return bvp_tensor, hr_tensor


def demo_physnet_model():
    """Demonstrate PhysNet model inference."""
    print("\n PhysNet Model Demonstration")
    print("=" * 50)
    
    # Initialize model
    model = PhysNet(S=2, in_ch=3)
    model.eval()
    
    print(f" Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate synthetic input
    demo_input = generate_synthetic_video(batch_size=1, time_frames=300)
    
    # Model inference
    print(" Running model inference...")
    with torch.no_grad():
        output = model(demo_input)
    
    print(f" Model output shape: {output.shape}")
    print(f" Output contains {output.shape[1]} spatial signals + 1 averaged signal")
    
    # Extract rPPG signal (last channel is the averaged signal)
    rppg_signal = output[0, -1, :].numpy()
    
    # Estimate heart rate
    estimated_hr, psd, freq_axis = hr_fft_small_Fs_internal(rppg_signal, 30, 1800)
    print(f" Estimated heart rate: {estimated_hr} BPM")
    
    return rppg_signal, estimated_hr


def demo_frequency_contrast():
    """Demonstrate frequency contrast learning."""
    print("\n Frequency Contrast Learning Demonstration")
    print("=" * 55)
    
    # Initialize models
    backbone = PhysNet(S=2, in_ch=3)
    freq_contrast = FrequencyContrast(backbone, window_size=150, num_views=2)
    freq_contrast.eval()
    
    # Generate synthetic input
    demo_input = generate_synthetic_video(batch_size=2, time_frames=300)
    
    print(" Running frequency contrast learning...")
    with torch.no_grad():
        backbone_out, branches = freq_contrast(demo_input)
    
    print(f" Backbone output shape: {backbone_out.shape}")
    print(f" Number of branches: {len(branches)}")
    
    for branch_name, branch_data in branches.items():
        print(f"   - {branch_name}: {len(branch_data)} views")
    
    return backbone_out, branches


def demo_signal_processing():
    """Demonstrate signal processing capabilities."""
    print("\n Signal Processing Demonstration")
    print("=" * 45)
    
    # Generate synthetic signals
    bvp_signals, true_hrs = generate_synthetic_bvp(batch_size=3, time_frames=300)
    
    # Process each signal
    estimated_hrs = []
    for i, bvp in enumerate(bvp_signals):
        hr, psd, freq = hr_fft_small_Fs_internal(bvp.numpy(), 30, 1800)
        estimated_hrs.append(hr)
        
        print(f"Signal {i+1}: True HR = {true_hrs[i]:.1f} BPM, "
              f"Estimated HR = {hr} BPM, "
              f"Error = {abs(hr - true_hrs[i]):.1f} BPM")
    
    # Demonstrate power spectral density analysis
    print("\n Power Spectral Density Analysis:")
    test_signal = bvp_signals[0].unsqueeze(0)  # Add batch dimension
    
    freqs, psd = torch_power_spectral_density(
        test_signal, nfft=1800, fps=30, 
        low_hz=0.6, high_hz=3.0, normalize=True, bandpass=True
    )
    
    print(f" Frequency range: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz")
    print(f" PSD shape: {psd.shape}")
    
    return bvp_signals, estimated_hrs


def demo_distributed_training_setup():
    """Demonstrate distributed training setup (without actual training)."""
    print("\n Distributed Training Setup Demonstration")
    print("=" * 52)
    
    # This would normally be called in a distributed setting
    # Here we just show the setup process
    
    from src.training.distributed_utils import get_world_size, get_rank, is_main_process
    
    print(f" World size: {get_world_size()}")
    print(f" Current rank: {get_rank()}")
    print(f" Is main process: {is_main_process()}")
    
    # Show how model would be wrapped for distributed training
    model = PhysNet(S=2, in_ch=3)
    print(f" Model ready for distributed training")
    print(f" Model device: {next(model.parameters()).device}")
    
    return model


def visualize_results(rppg_signal, estimated_hr):
    """Create visualizations of the demo results."""
    print("\n Creating Visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('rPPG Model Demonstration Results', fontsize=16)
    
    # Plot 1: Raw rPPG signal
    axes[0, 0].plot(rppg_signal[:200])  # Show first 200 samples
    axes[0, 0].set_title('Extracted rPPG Signal (first 200 samples)')
    axes[0, 0].set_xlabel('Time (frames)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Frequency domain analysis
    hr, psd, freq = hr_fft_small_Fs_internal(rppg_signal, 30, 1800)
    freq_hz = freq[:len(psd)] / 60  # Convert to Hz
    
    axes[0, 1].plot(freq_hz[:100], psd[:100])  # Show relevant frequency range
    axes[0, 1].axvline(x=estimated_hr/60, color='red', linestyle='--', 
                      label=f'Estimated HR: {estimated_hr} BPM')
    axes[0, 1].set_title('Power Spectral Density')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Power')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Model architecture visualization (simplified)
    axes[1, 0].text(0.5, 0.7, 'PhysNet Architecture', ha='center', va='center', 
                   fontsize=14, weight='bold')
    axes[1, 0].text(0.5, 0.5, 'Input: (B, 3, T, H, W)', ha='center', va='center')
    axes[1, 0].text(0.5, 0.3, '3D CNN Encoder-Decoder', ha='center', va='center')
    axes[1, 0].text(0.5, 0.1, 'Output: rPPG Signal', ha='center', va='center')
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis('off')
    
    # Plot 4: Demo info
    demo_info = f"""
    Demo Configuration:
    • Video: 300 frames @ 30fps
    • Resolution: 128×128 pixels
    • Model: PhysNet (S=2)
    • Estimated HR: {estimated_hr} BPM
    • Processing: Real-time capable
    
    ⚠️ DEMONSTRATION ONLY
    No real data used
    """
    
    axes[1, 1].text(0.05, 0.95, demo_info, ha='left', va='top', 
                   fontsize=10, family='monospace')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('demo_outputs', exist_ok=True)
    plt.savefig('demo_outputs/rppg_demo_results.png', dpi=150, bbox_inches='tight')
    print(" Visualization saved to: demo_outputs/rppg_demo_results.png")
    
    # Show plot if running interactively
    try:
        plt.show()
    except:
        print(" Running in non-interactive mode - plot saved only")


def main():
    """Main demonstration function."""
    print(" Remote Photoplethysmography - Technical Demonstration")
    print("=" * 60)
    print("  IMPORTANT: This is demonstration code only!")
    print("  No real data is used - all inputs are synthetic")
    print("  Not intended for any practical application")
    print("=" * 60)
    
    try:
        # Demo 1: Basic PhysNet model
        rppg_signal, estimated_hr = demo_physnet_model()
        
        # Demo 2: Frequency contrast learning
        demo_frequency_contrast()
        
        # Demo 3: Signal processing
        demo_signal_processing()
        
        # Demo 4: Distributed training setup
        demo_distributed_training_setup()
        
        # Create visualizations
        visualize_results(rppg_signal, estimated_hr)
        
        print("\n Demonstration completed successfully!")

        print(f"\n Model successfully processed synthetic video and estimated heart rate")

        
    except Exception as e:
        print(f"\n Demo encountered an error: {e}")
        print(" This is expected in some environments due to dependencies")
    
    print("\n" + "=" * 60)
    print("  REMINDER: This demonstration contains NO confidential information")
    print("  For portfolio purposes only")
    print("  Not for commercial, academic, or practical use")
    print("=" * 60)


if __name__ == '__main__':
    main()
