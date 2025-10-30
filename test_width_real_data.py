"""
Test width convergence on REAL DRIVE data.
Loads actual vessel masks, extracts skeletons, and tests if width optimization works.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import sys
import os

# Auto-detect environment
if os.path.exists('/content/Ribbon'):
    sys.path.append('/content/Ribbon')
    DRIVE_PATH = '/content/Ribbon/Codes/drive'  # Colab path
    print("Running in Colab environment")
else:
    print("Running in local environment")
    DRIVE_PATH = './Codes/drive'  # Local path

from Codes import utils

# Auto-detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")


def load_drive_sample(sample_idx=0, train=True, crop_size=256):
    """
    Load a DRIVE image, vessel mask, and compute skeleton + distance map.
    Extracts a crop from the center with vessels.
    
    Args:
        sample_idx: which DRIVE image to load
        train: whether to use training or test set
        crop_size: size of crop to extract (crop_size x crop_size)
    
    Returns:
        image: grayscale image crop
        mask: binary vessel mask crop
        skeleton: binary skeleton crop
        true_dmap: signed distance map crop
        centerline_coords: skeleton coordinates (adjusted to crop)
    """
    subset = 'training' if train else 'test'
    
    # Load image (training images are numbered 21-40, test 01-20)
    if train:
        img_idx = sample_idx + 21  # Training: 21-40
        img_path = f'{DRIVE_PATH}/{subset}/images/{img_idx:02d}_training.tif'
        mask_path = f'{DRIVE_PATH}/{subset}/1st_manual/{img_idx:02d}_manual1.gif'
    else:
        img_idx = sample_idx + 1  # Test: 01-20
        img_path = f'{DRIVE_PATH}/{subset}/images/{img_idx:02d}_test.tif'
        mask_path = f'{DRIVE_PATH}/{subset}/mask/{img_idx:02d}_test_mask.gif'
    
    print(f"Loading from: {img_path}")
    
    try:
        from PIL import Image
        
        # Load and convert to grayscale
        image = Image.open(img_path).convert('L')
        image = np.array(image).astype(np.float32) / 255.0
        
        # Load vessel mask
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = (mask > 128).astype(np.uint8)  # Binarize
        
        print(f"Full image shape: {image.shape}, Mask shape: {mask.shape}")
        print(f"Full vessel pixels: {mask.sum()}")
        
        # Extract crop from center with vessels
        H, W = image.shape
        center_y, center_x = H // 2, W // 2
        
        # Calculate crop boundaries
        y_start = max(0, center_y - crop_size // 2)
        y_end = min(H, y_start + crop_size)
        x_start = max(0, center_x - crop_size // 2)
        x_end = min(W, x_start + crop_size)
        
        # Adjust if crop extends beyond boundaries
        if y_end - y_start < crop_size:
            y_start = max(0, y_end - crop_size)
        if x_end - x_start < crop_size:
            x_start = max(0, x_end - crop_size)
        
        # Extract crops
        image = image[y_start:y_end, x_start:x_end]
        mask = mask[y_start:y_end, x_start:x_end]
        
        print(f"Crop region: [{y_start}:{y_end}, {x_start}:{x_end}]")
        print(f"Crop shape: {image.shape}, Vessel pixels in crop: {mask.sum()}")
        
        if mask.sum() == 0:
            print("⚠️  Warning: No vessels in this crop! Consider adjusting crop location.")
        
        # Extract skeleton
        skeleton = skeletonize(mask).astype(np.uint8)
        print(f"Skeleton pixels: {skeleton.sum()}")
        
        # Get centerline coordinates (already in crop coordinates)
        centerline_coords = np.column_stack(np.where(skeleton > 0))  # (N, 2) [y, x]
        print(f"Centerline points: {len(centerline_coords)}")
        
        # Compute signed distance map
        dist_outside = distance_transform_edt(1 - mask)
        dist_inside = distance_transform_edt(mask)
        true_dmap = dist_outside - dist_inside
        
        return image, mask, skeleton, true_dmap, centerline_coords
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None


def render_ribbon_distance_map_real(pts, widths, img_shape):
    """
    Render a ribbon distance map for real data (potentially non-square).
    
    Args:
        pts: (N, 2) tensor of centerline points [y, x]
        widths: (N,) tensor of widths at each point
        img_shape: (H, W) tuple
    
    Returns:
        distance_map: (H, W) tensor
    """
    N = pts.shape[0]
    device = pts.device
    H, W = img_shape
    
    # Create coordinate grid
    y_coords = torch.arange(H, dtype=torch.float32, device=device)
    x_coords = torch.arange(W, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    grid = torch.stack([yy, xx], dim=-1)  # (H, W, 2)
    
    # Compute distance from each pixel to each centerline point
    distances_to_pts = torch.norm(grid.unsqueeze(2) - pts.unsqueeze(0).unsqueeze(0), dim=-1)
    
    # For each pixel, find the closest centerline point
    min_dist_to_centerline, closest_idx = torch.min(distances_to_pts, dim=2)  # (H, W)
    
    # Get the width at the closest centerline point for each pixel
    width_at_pixel = widths[closest_idx]  # (H, W)
    
    # Signed distance map: negative inside, positive outside
    distance_map = min_dist_to_centerline - width_at_pixel / 2.0
    
    return distance_map


def test_width_on_real_data(image, mask, skeleton, true_dmap, centerline_coords, 
                             init_width=5.0, n_steps=100, lr=0.1, sample_idx=0):
    """
    Test width optimization on real DRIVE data.
    """
    print(f"\n{'='*60}")
    print(f"Testing on REAL DRIVE Data - Sample {sample_idx}")
    print(f"Initial Width: {init_width}px, Learning Rate: {lr}")
    print(f"{'='*60}\n")
    
    img_shape = image.shape
    
    # Convert to torch
    true_dmap_torch = torch.from_numpy(true_dmap).float().to(device)
    pts = torch.from_numpy(centerline_coords).float().to(device)
    
    # Initialize widths uniformly
    N = len(centerline_coords)
    widths = torch.full((N,), float(init_width), dtype=torch.float32, device=device, requires_grad=True)
    
    print(f"Centerline points: {N}")
    print(f"Image shape: {img_shape}")
    
    # Store initial state
    with torch.no_grad():
        initial_dmap = render_ribbon_distance_map_real(pts, widths, img_shape).cpu().numpy()
    
    # Optimization loop
    width_history = [widths.detach().cpu().numpy().copy()]
    loss_history = []
    
    print("Starting optimization...\n")
    for step in range(n_steps):
        # Zero gradients
        if widths.grad is not None:
            widths.grad.zero_()
        
        # Render predicted distance map
        pred_dmap = render_ribbon_distance_map_real(pts, widths, img_shape)
        
        # Compute MSE loss
        loss = torch.mean((pred_dmap - true_dmap_torch) ** 2)
        
        # Compute gradient
        grad_w = torch.autograd.grad(loss, widths, create_graph=False)[0]
        loss_history.append(loss.item())
        
        # Update widths (gradient descent)
        with torch.no_grad():
            widths.data = widths.data - lr * grad_w
            # Clamp widths to reasonable range
            widths.data = torch.clamp(widths.data, min=1.0, max=20.0)
        
        # Record width
        current_widths = widths.detach().cpu().numpy().copy()
        width_history.append(current_widths)
        
        if step % 10 == 0 or step == n_steps - 1:
            mean_width = current_widths.mean()
            std_width = current_widths.std()
            print(f"  Step {step:3d}: Loss={loss.item():.6f}, "
                  f"Mean Width={mean_width:.3f}px (±{std_width:.3f}), "
                  f"Grad norm={grad_w.abs().mean().item():.6f}")
    
    # Get final state
    with torch.no_grad():
        final_dmap = render_ribbon_distance_map_real(pts, widths, img_shape).cpu().numpy()
    final_widths = widths.detach().cpu().numpy()
    
    print(f"\nResults:")
    print(f"  Initial Width: {width_history[0].mean():.3f}px (±{width_history[0].std():.3f})")
    print(f"  Final Width:   {final_widths.mean():.3f}px (±{final_widths.std():.3f})")
    print(f"  Width Change:  {final_widths.mean() - width_history[0].mean():+.3f}px")
    print(f"  Min Width:     {final_widths.min():.3f}px")
    print(f"  Max Width:     {final_widths.max():.3f}px")
    
    return {
        'image': image,
        'mask': mask,
        'skeleton': skeleton,
        'true_dmap': true_dmap,
        'initial_dmap': initial_dmap,
        'final_dmap': final_dmap,
        'init_width': init_width,
        'width_history': np.array(width_history),
        'loss_history': loss_history,
        'sample_idx': sample_idx
    }


def visualize_real_results(results, save_path='./width_convergence_real.png'):
    """Visualize width convergence on real data."""
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    sample_idx = results['sample_idx']
    init_width = results['init_width']
    
    fig.suptitle(f'Width Convergence on Real DRIVE Data (Sample {sample_idx})', 
                 fontsize=16, fontweight='bold')
    
    # === ROW 1: IMAGE & MASK ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(results['image'], cmap='gray', origin='lower')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(results['mask'], cmap='gray', origin='lower')
    ax2.set_title(f'Vessel Mask ({results["mask"].sum()} pixels)')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(results['skeleton'], cmap='gray', origin='lower')
    ax3.set_title(f'Skeleton ({results["skeleton"].sum()} points)')
    ax3.axis('off')
    
    # === ROW 2: DISTANCE MAPS (INITIAL) ===
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(results['true_dmap'], cmap='RdBu_r', origin='lower', vmin=-15, vmax=15)
    ax4.set_title(f'GT Distance Map\n(min:{results["true_dmap"].min():.1f}, max:{results["true_dmap"].max():.1f})')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(results['initial_dmap'], cmap='RdBu_r', origin='lower', vmin=-15, vmax=15)
    initial_mean = results['width_history'][0].mean()
    ax5.set_title(f'Initial Ribbon (w={initial_mean:.2f}px)\n(min:{results["initial_dmap"].min():.1f}, max:{results["initial_dmap"].max():.1f})')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = fig.add_subplot(gs[1, 2])
    error_initial = np.abs(results['true_dmap'] - results['initial_dmap'])
    im6 = ax6.imshow(error_initial, cmap='hot', origin='lower', vmin=0, vmax=10)
    ax6.set_title(f'Initial Error\n(mean:{error_initial.mean():.2f})')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    # === ROW 3: DISTANCE MAPS (FINAL) ===
    ax7 = fig.add_subplot(gs[2, 0])
    im7 = ax7.imshow(results['true_dmap'], cmap='RdBu_r', origin='lower', vmin=-15, vmax=15)
    ax7.set_title('GT Distance Map')
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046)
    
    ax8 = fig.add_subplot(gs[2, 1])
    im8 = ax8.imshow(results['final_dmap'], cmap='RdBu_r', origin='lower', vmin=-15, vmax=15)
    final_mean = results['width_history'][-1].mean()
    ax8.set_title(f'Final Ribbon (w={final_mean:.2f}px)\n(min:{results["final_dmap"].min():.1f}, max:{results["final_dmap"].max():.1f})')
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046)
    
    ax9 = fig.add_subplot(gs[2, 2])
    error_final = np.abs(results['true_dmap'] - results['final_dmap'])
    im9 = ax9.imshow(error_final, cmap='hot', origin='lower', vmin=0, vmax=10)
    ax9.set_title(f'Final Error\n(mean:{error_final.mean():.2f})')
    ax9.axis('off')
    plt.colorbar(im9, ax=ax9, fraction=0.046)
    
    # === ROW 4: WIDTH EVOLUTION ===
    width_history = results['width_history']
    n_steps = len(width_history)
    
    ax10 = fig.add_subplot(gs[3, :])
    
    # Plot statistics
    mean_widths = width_history.mean(axis=1)
    std_widths = width_history.std(axis=1)
    min_widths = width_history.min(axis=1)
    max_widths = width_history.max(axis=1)
    steps = np.arange(n_steps)
    
    ax10.plot(steps, mean_widths, 'b-', linewidth=2, label='Mean Width')
    ax10.fill_between(steps, mean_widths - std_widths, mean_widths + std_widths, 
                     alpha=0.3, color='blue', label='±1 Std')
    ax10.plot(steps, min_widths, 'g--', linewidth=1, alpha=0.7, label='Min Width')
    ax10.plot(steps, max_widths, 'r--', linewidth=1, alpha=0.7, label='Max Width')
    ax10.axhline(y=init_width, color='orange', linestyle=':', linewidth=2, label=f'Initial ({init_width}px)')
    
    ax10.set_xlabel('Optimization Step', fontsize=12)
    ax10.set_ylabel('Width (pixels)', fontsize=12)
    ax10.set_title('Width Evolution on Real DRIVE Data', fontsize=14, fontweight='bold')
    ax10.grid(True, alpha=0.3)
    ax10.legend(loc='best', fontsize=10)
    
    # Add summary
    final_mean = mean_widths[-1]
    width_change = final_mean - init_width
    
    summary_text = (f"Initial: {init_width:.2f}px → Final: {final_mean:.2f}px (±{std_widths[-1]:.2f})\n"
                   f"Change: {width_change:+.2f}px | Range: [{min_widths[-1]:.2f}, {max_widths[-1]:.2f}]")
    ax10.text(0.98, 0.98, summary_text, transform=ax10.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {save_path}\n")
    plt.close()


def main():
    """Test width convergence on real DRIVE data (using crops for efficiency)."""
    print("\n" + "="*60)
    print("WIDTH CONVERGENCE TEST ON REAL DRIVE DATA (CROP)")
    print("="*60 + "\n")
    
    # Load real data with cropping for efficiency
    sample_idx = 0
    crop_size = 256  # Use 256x256 crop for manageable computation
    print(f"Using {crop_size}x{crop_size} crop from center of image\n")
    
    image, mask, skeleton, true_dmap, centerline = load_drive_sample(
        sample_idx, train=True, crop_size=crop_size
    )
    
    if image is None:
        print("❌ Failed to load DRIVE data. Make sure DRIVE dataset is available.")
        return
    
    if len(centerline) == 0:
        print("❌ No vessels found in crop. Try a different crop location.")
        return
    
    # Test with small step size (current config)
    print("\n### TEST 1: Small step size (lr=0.1) ###")
    results_small = test_width_on_real_data(
        image, mask, skeleton, true_dmap, centerline,
        init_width=5.0,
        n_steps=50,  # Reduced for faster testing
        lr=0.1,
        sample_idx=sample_idx
    )
    visualize_real_results(results_small, save_path='./width_real_small_lr.png')
    
    # Test with large step size (proposed fix)
    print("\n### TEST 2: Large step size (lr=2.5) ###")
    results_large = test_width_on_real_data(
        image, mask, skeleton, true_dmap, centerline,
        init_width=5.0,
        n_steps=50,  # Reduced for faster testing
        lr=2.5,
        sample_idx=sample_idx
    )
    visualize_real_results(results_large, save_path='./width_real_large_lr.png')
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"\nSmall LR (0.1):")
    print(f"  Width change: {results_small['width_history'][-1].mean() - results_small['width_history'][0].mean():+.3f}px")
    print(f"  Final loss: {results_small['loss_history'][-1]:.6f}")
    
    print(f"\nLarge LR (2.5):")
    print(f"  Width change: {results_large['width_history'][-1].mean() - results_large['width_history'][0].mean():+.3f}px")
    print(f"  Final loss: {results_large['loss_history'][-1]:.6f}")
    
    print("\n" + "="*60)
    print("✅ Done! Check the generated PNG files for visualizations.")


if __name__ == "__main__":
    main()

