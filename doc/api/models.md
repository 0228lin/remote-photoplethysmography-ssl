# Models API Reference

## PhysNet

### Class: `PhysNet`

3D CNN architecture for remote photoplethysmography signal extraction.

#### Parameters
- `S` (int): Spatial dimension of output ST-rPPG block (default: 2)
- `in_ch` (int): Number of input channels (default: 3 for RGB)

#### Methods

##### `forward(x: torch.Tensor) -> torch.Tensor`
Perform forward pass through the network.

**Parameters:**
- `x`: Input tensor of shape `(B, C, T, H, W)`
  - B: Batch size
  - C: Number of channels (3 for RGB)
  - T: Temporal length (number of frames)
  - H, W: Spatial dimensions

**Returns:**
- Output tensor of shape `(B, S²+1, T)` containing S² spatial signals plus averaged signal

**Example:**
```python
from src.models.physnet import PhysNet

model = PhysNet(S=2, in_ch=3)
input_video = torch.randn(1, 3, 300, 128, 128)
output = model(input_video)  # Shape: (1, 5, 300)
rppg_signal = output[0, -1, :]  # Extract averaged rPPG signal
```

## FrequencyContrast

### Class: `FrequencyContrast`

Self-supervised learning wrapper implementing frequency domain augmentation.

#### Parameters
- `backbone`: Backbone model (e.g., PhysNet)
- `window_size` (int): Temporal window size for multi-view learning
- `num_views` (int): Number of temporal views to generate

#### Methods

##### `forward(x: torch.Tensor) -> Tuple[torch.Tensor, Dict]`
Apply frequency augmentation and extract multi-view representations.

**Example:**
```python
from src.models.physnet import PhysNet
from src.models.frequency_contrast import FrequencyContrast

backbone = PhysNet(S=2, in_ch=3)
freq_model = FrequencyContrast(backbone, window_size=150, num_views=2)

input_video = torch.randn(2, 3, 300, 128, 128)
backbone_out, branches = freq_model(input_video)
```
