"""
Demo script for rPPG model inference.
Demonstration purposes only - no confidential data.
"""

import torch
from src.models.physnet import PhysNet

def demo_inference():
    """Demonstrate model inference capability."""
    # Initialize model
    model = PhysNet(S=4, in_ch=3)
    model.eval()
    
    # Demo input (batch_size=1, channels=3, time=300, height=128, width=128)
    demo_input = torch.randn(1, 3, 300, 128, 128)
    
    with torch.no_grad():
        output = model(demo_input)
    
    print(f"Input shape: {demo_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Demo inference completed successfully!")

if __name__ == "__main__":
    demo_inference()
