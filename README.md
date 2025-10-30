# RibbonSnake: Width-Aware Vessel Segmentation with Snake Optimization

A deep learning approach for retinal vessel segmentation that combines U-Net predictions with active contour (snake) optimization to learn accurate vessel widths and positions.

## Overview

This project implements a novel training strategy that:
1. Uses a U-Net to predict signed distance maps of blood vessels
2. Optimizes snake models with width parameters on the predictions
3. Uses the optimized snakes as supervision signal to improve the network

The key innovation is the **RibbonSnake** model that optimizes both centerline position AND vessel width during training.

---

## Project Structure

```
├── main.py                          # Main training script
├── main.config                      # Training configuration
├── compare_mse_vs_snake.py         # Comparison visualization script
├── create_signed_distance_maps.py  # GT distance map generation
├── README.md                        # This file
│
├── Codes/
│   ├── dataset.py                  # DRIVE dataset loader
│   ├── training.py                 # Training loop and epoch handler
│   ├── unet.py                     # U-Net architecture
│   ├── augmentations.py            # Data augmentation utilities
│   ├── utils.py                    # Helper functions
│   │
│   ├── Losses/
│   │   ├── losses.py               # Loss functions (MSE, SnakeSimpleLoss)
│   │   ├── rib.py                  # RibbonSnake class (width optimization)
│   │   ├── gradRib.py              # GradImRib (image gradient-based snake)
│   │   └── snake.py                # Base Snake class
│   │
│   └── drive/                      # DRIVE dataset directory
│       └── training/
│           ├── images_npy/         # Retinal images (.npy)
│           ├── 1st_manual_npy/     # Binary vessel masks (.npy)
│           ├── signed_distance_maps/ # Ground truth distance maps
│           └── graphs_oversampled/  # Skeleton graphs (centerlines)
│
└── Test Scripts/                   # Width optimization debugging
    ├── test_width_simple.py        # Autograd-based width test (synthetic)
    ├── test_width_simple_rib.py    # RibbonSnake width test (synthetic)
    └── test_width_real_data.py     # Width test on real DRIVE images
```

---

## Key Files Explained

### Training Files

#### `main.py`
Main entry point for training. Handles:
- Configuration loading
- Model initialization
- Dataset loading
- Loss function setup (MSE warmup + SnakeSimpleLoss)
- Training loop orchestration
- Checkpoint saving/loading

**Usage:**
```bash
python main.py --config_file main.config
```

#### `main.config`
YAML configuration file containing all hyperparameters:
- **Network**: U-Net architecture settings (channels, levels, dropout)
- **Training**: Batch size, learning rate, number of epochs
- **Snake Loss**: Step size, smoothness weights, optimization steps
- **Data**: Paths, crop sizes, thresholds

**Key Parameters:**
- `ours_start: 100` - Start SnakeSimpleLoss after 100 epochs of MSE
- `stepsz: 0.1` - Learning rate for snake position/width optimization
- `nsteps: 40` - Number of optimization steps per snake
- `num_iters: 200` - Total training epochs

#### `Codes/training.py`
Implements the training epoch loop:
- Handles batch processing
- Switches between MSE and SnakeSimpleLoss based on epoch
- Generates training visualizations (2-row plots: distance maps + binary vessels)
- Saves checkpoints to Google Drive with datetime stamps

**Visualization:** Creates comprehensive plots showing:
- Row 1: Input, GT DMap, MSE Pred, Snake BEFORE, Snake AFTER
- Row 2: Input, GT Binary, Pred Binary, Snake BEFORE Binary, Snake AFTER Binary

### Dataset Files

#### `Codes/dataset.py`
DRIVE dataset loader that provides:
- Retinal images (grayscale converted from RGB)
- Signed distance maps (ground truth)
- Skeleton graphs (centerline annotations)
- Random cropping for training
- Batch collation

**Data split:**
- Training: Samples 21-36 (16 images)
- Validation: Samples 37-40 (4 images)

#### `create_signed_distance_maps.py`
Preprocessing script to generate ground truth distance maps from binary masks:
```python
# Inside vessels: negative (distance to edge)
# Outside vessels: positive (distance to edge)
# Enhancement: negative values × 2.0 for stronger supervision
```

**Run once before training to create GT distance maps.**

### Loss Functions

#### `Codes/Losses/losses.py`

**`SnakeSimpleLoss`**: Main loss function for snake-based training
1. Network predicts distance map
2. Compute gradients for position and width optimization
3. Initialize snake from skeleton graph (width = 5px)
4. Optimize snake position and width using gradients (40 steps)
5. Render optimized snake as distance map
6. Compute MSE between rendered snake and network prediction
7. Backpropagate to improve network

**Key features:**
- 2× enhancement of negative values (vessel interiors) to match GT
- Separate position and width optimization phases
- Width optimization uses absolute distance map gradients

#### `Codes/Losses/rib.py`

**`RibbonSnake`**: Core width-aware snake implementation

**Width optimization formula:**
```python
# Sample left and right edges based on current width
left_pts = center_pts + normals * half_width
right_pts = center_pts - normals * half_width

# Compute radial gradient (edge strength difference)
grad_R - grad_L  # Positive = expand, Negative = shrink

# Smoothness term (prevent jagged widths)
smoothness = 2*w[i] - w[i-1] - w[i+1]

# Combined width update
w = w - stepsz * (radial_gradient + alpha * smoothness)
```

**Special features:**
- `endpoint_alpha_scale = 0.3`: Reduces smoothness coupling for first/last nodes, allowing endpoints to converge independently
- Separate 2D and 3D implementations
- Distance map rendering with width information

#### `Codes/Losses/gradRib.py`

**`GradImRib`**: Image gradient-based snake that inherits from RibbonSnake

**Key functions:**
- `makeGaussEdgeFltr()`: Creates Gaussian-derivative filters for gradient computation
- `cmptGradIm()`: Computes image gradients using convolution
- `cmptExtGrad()`: Samples gradients at floating-point positions using `grid_sample`

**Optimization strategy:**
- First half of iterations: Position optimization (move toward vessels)
- Second half: Width optimization (adjust vessel thickness)

### Comparison & Visualization

#### `compare_mse_vs_snake.py`
Generates comprehensive comparison plots between MSE-only and Snake-optimized models.

**Output:** 3 rows × 6 columns grid
- **Row 1 (Distance Maps):** Input, GT, MSE Pred, Snake Pred, Snake BEFORE, Snake AFTER
- **Row 2 (Binary Vessels):** Shows vessel width information by thresholding at 0
- **Row 3 (Overlays):** TP (white), FP (red), FN (blue) with metrics

**Metrics computed:**
- Precision, Recall, F1 Score, Quality (TP/(TP+FP+FN))
- Aggregate statistics across all samples

**Usage:**
```bash
python compare_mse_vs_snake.py
```

### Test Scripts (Debugging Tools)

#### `test_width_simple.py`
**Autograd-based width optimization on synthetic data**

Tests width convergence using PyTorch's automatic differentiation:
- Creates synthetic straight lines (horizontal/vertical)
- Tests growing (start=3px → target=7px)
- Tests shrinking (start=7px → target=3px)
- Renders ribbon distance maps
- Computes gradients via `torch.autograd.grad`
- Visualizes convergence curves

**Usage:**
```bash
python test_width_simple.py
```

**Output:** Plots showing width evolution, loss curves, and distance map visualization

#### `test_width_simple_rib.py`
**RibbonSnake-style width optimization on synthetic data**

Exactly replicates the training width optimization logic:
- Uses same gradient computation (`makeGaussEdgeFltr`, `cmptGradIm`, `cmptExtGrad`)
- Same smoothness term formula
- Same endpoint smoothness reduction
- Tests on synthetic lines to isolate width mechanism

**Key parameters:**
- `width_stepsz`: Learning rate for width updates
- `n_steps`: Number of optimization iterations
- `endpoint_alpha_scale`: Endpoint smoothness reduction

**Usage:**
```bash
python test_width_simple_rib.py
```

#### `test_width_real_data.py`
**Width optimization test on real DRIVE dataset**

Validates width convergence on actual retinal images:
- Loads DRIVE samples with ground truth
- Extracts skeleton from binary mask
- Initializes snake with fixed width
- Runs width optimization
- Compares against true distance map

**Usage:**
```bash
python test_width_real_data.py
```

---

## Installation & Setup

### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate    # On Windows

# Install dependencies
pip install torch torchvision numpy scipy scikit-image networkx matplotlib pillow
```

### 2. Prepare DRIVE Dataset

**Directory structure:**
```
Codes/drive/training/
├── images/              # Original .tif images
├── 1st_manual/          # Binary vessel masks (.gif)
├── images_npy/          # Converted images (.npy)
├── 1st_manual_npy/      # Converted masks (.npy)
├── signed_distance_maps/  # Generated distance maps
└── graphs_oversampled/    # Skeleton graphs
```

**Steps:**
1. Download DRIVE dataset (https://drive.grand-challenge.org/)
2. Convert images and masks to `.npy` format
3. Generate signed distance maps:
   ```bash
   python create_signed_distance_maps.py
   ```
4. Generate skeleton graphs (oversampled with spacing=5px)

### 3. Training

**Quick start:**
```bash
python main.py --config_file main.config
```

**Resume from checkpoint:**
```bash
python main.py --config_file main.config --resume_from path/to/checkpoint.pt --start_epoch 100
```

**Training strategy:**
- Epochs 0-99: MSE Loss (network learns basic distance maps)
- Epochs 100-199: SnakeSimpleLoss (width optimization + annotation adjustment)

**Monitor training:**
- Loss printed every 10 epochs
- Plots saved to Google Drive: `/content/drive/MyDrive/ribbs/october/RibbonSertac/{datetime}/`
- Checkpoints saved every 50 epochs

---

## Configuration Guide

### Key Hyperparameters

#### Snake Optimization
```yaml
stepsz: 0.1              # Learning rate for both position and width
alpha: 0.0001            # Smoothness weight (internal energy)
beta: 0.01               # Curvature weight
nsteps: 40               # Optimization iterations per snake
fltrstdev: 0.5           # Gaussian filter std for gradient computation
```

#### Network Architecture
```yaml
m_channels: 32           # Base number of channels
n_convs: 2               # Convolutions per level
n_levels: 3              # U-Net depth
dropout: 0.1             # Dropout rate
batch_norm: True         # Use batch normalization
```

#### Training Schedule
```yaml
ours_start: 100          # Epoch to switch from MSE to SnakeSimpleLoss
num_iters: 200           # Total training epochs
batch_size: 4            # Batch size
lr: 0.001                # Initial learning rate
lr_decay: True           # Use learning rate decay
```

### Advanced Parameters

**Width Optimization Settings:**
- `endpoint_alpha_scale: 0.3` (in `rib.py`): Reduces smoothness for endpoints
- `enhancement_factor: 2.0` (in `losses.py`): Amplifies vessel interiors

**Distance Map Settings:**
```yaml
dmax: 15                 # Maximum distance value (clipping)
extgradfac: 2.0          # External gradient amplification
maxedgelength: 5         # Maximum edge length in snake
```

---

## How It Works

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: MSE Warmup (Epochs 0-99)                         │
│                                                              │
│  Input Image → U-Net → Predicted DMap                       │
│                              ↓                               │
│                         MSE Loss ← Ground Truth DMap        │
│                              ↓                               │
│                        Backprop to U-Net                     │
│                                                              │
│  Result: Network learns basic vessel distance maps          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Phase 2: Snake Optimization (Epochs 100-199)              │
│                                                              │
│  Input Image → U-Net → Predicted DMap                       │
│                              ↓                               │
│                    Compute Gradients (position & width)     │
│                              ↓                               │
│     Skeleton Graph → Initialize Snake (width=5px)           │
│                              ↓                               │
│              Optimize Snake (40 steps):                      │
│              • Position: Move toward vessel                  │
│              • Width: Shrink/Grow to match gradients        │
│                              ↓                               │
│                 Render Snake as Distance Map                │
│                 (with 2× enhancement for negatives)         │
│                              ↓                               │
│              MSE Loss ← Ground Truth DMap                   │
│                              ↓                               │
│                        Backprop to U-Net                     │
│                                                              │
│  Result: Network learns to predict distance maps that       │
│          lead to accurate vessel widths after optimization  │
└─────────────────────────────────────────────────────────────┘
```

### Width Optimization Mechanism

The snake's width at each node is optimized based on:

1. **Radial Gradient** (Data term):
   - Sample image gradients at left and right edges
   - If `grad_R > grad_L`: Expand width (right edge is stronger)
   - If `grad_R < grad_L`: Shrink width (left edge is stronger)

2. **Smoothness Term** (Regularization):
   - Penalizes width changes between neighboring nodes
   - `smoothness = 2*w[i] - w[i-1] - w[i+1]`
   - Reduced for endpoints to allow independent convergence

3. **Balanced Update**:
   - `alpha = |data_term| / (|smoothness| + ε)`
   - Automatically balances between data fit and smoothness
   - Large gradients → trust data more
   - Small gradients → trust smoothness more

---

## Troubleshooting

### Common Issues

**1. Width not converging:**
- Check `stepsz` is not too small (try 0.1-0.5)
- Increase `nsteps` (40-100)
- Verify gradients are non-zero (check training plots)

**2. Unstable training:**
- Ensure MSE warmup is sufficient (`ours_start ≥ 100`)
- Reduce `stepsz` if widths explode
- Check for NaN/Inf in loss values

**3. Network not learning vessel interiors:**
- Verify 2× enhancement is active in `losses.py`
- Check GT distance maps have strong negative values
- Ensure snake widths are optimizing (test with synthetic data)

**4. Out of memory:**
- Reduce `batch_size` in config
- Reduce `crop_size` in config
- Use gradient checkpointing in U-Net

**5. Snake endpoints not converging:**
- Verify `endpoint_alpha_scale = 0.3` in `rib.py`
- Increase `nsteps` for more optimization time
- Check endpoint gradient sampling (no out-of-bounds)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{ribbonsnake2024,
  title={Width-Aware Vessel Segmentation with Active Contour Optimization},
  author={Your Name},
  booktitle={Conference},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact

For questions or issues, please open a GitHub issue or contact:
- Email: sepehrmaleki88@gmail.com
- GitHub: https://github.com/sepehrmaleki369/Ribbon

---

## Acknowledgments

- DRIVE dataset: https://drive.grand-challenge.org/
- U-Net architecture: Ronneberger et al., 2015
- Active contour methods: Kass et al., 1988
