"""
Simplified test to verify width gradient computation and convergence.
Tests the core width optimization logic directly.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import distance_transform_edt
import torch.nn.functional as F

# Auto-detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")


def create_synthetic_line(img_size=64, line_width=7, orientation='horizontal'):
    """Create a synthetic image with a straight line of known width."""
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    center = img_size // 2
    half_width = line_width // 2
    
    if orientation == 'horizontal':
        mask[center - half_width : center + half_width + 1, :] = 1
        centerline_coords = np.array([[center, x] for x in range(img_size)])
    else:
        mask[:, center - half_width : center + half_width + 1] = 1
        centerline_coords = np.array([[y, center] for y in range(img_size)])
    
    # Create signed distance map
    dist_outside = distance_transform_edt(1 - mask)
    dist_inside = distance_transform_edt(mask)
    distance_map = dist_outside - dist_inside
    
    # Create synthetic image
    image = np.ones((img_size, img_size), dtype=np.float32)
    image[mask == 1] = 0.3
    
    return image, distance_map, mask, centerline_coords


def render_ribbon_distance_map(pts, widths, img_size):
    """
    Render a ribbon (vessel with width) as a signed distance map.
    
    Args:
        pts: (N, 2) tensor of centerline points [y, x]
        widths: (N,) tensor of widths at each point
        img_size: int, size of the output image
    
    Returns:
        distance_map: (img_size, img_size) tensor
    """
    N = pts.shape[0]
    device = pts.device
    
    # Create coordinate grid
    y_coords = torch.arange(img_size, dtype=torch.float32, device=device)
    x_coords = torch.arange(img_size, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    grid = torch.stack([yy, xx], dim=-1)  # (H, W, 2)
    
    # Compute distance from each pixel to each centerline point
    # grid: (H, W, 2), pts: (N, 2) -> distances: (H, W, N)
    distances_to_pts = torch.norm(grid.unsqueeze(2) - pts.unsqueeze(0).unsqueeze(0), dim=-1)
    
    # For each pixel, find the closest centerline point
    min_dist_to_centerline, closest_idx = torch.min(distances_to_pts, dim=2)  # (H, W)
    
    # Get the width at the closest centerline point for each pixel
    width_at_pixel = widths[closest_idx]  # (H, W)
    
    # Signed distance map: negative inside, positive outside
    # Inside if distance < width/2
    distance_map = min_dist_to_centerline - width_at_pixel / 2.0
    
    return distance_map


def _compute_normals_along_polyline(pts):
    """
    Approximate unit normals for a 2D polyline pts (N,2).
    Returns normals (N,2).
    """
    N = pts.shape[0]
    eps = 1e-8
    tangents = torch.zeros_like(pts)
    if N > 1:
        tangents[1:-1] = (pts[2:] - pts[:-2]) * 0.5
        tangents[0] = pts[1] - pts[0]
        tangents[-1] = pts[-1] - pts[-2]
    tangents = tangents / (tangents.norm(dim=1, keepdim=True) + eps)
    normals = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=1)
    normals = normals / (normals.norm(dim=1, keepdim=True) + eps)
    return normals


def compute_width_gradient(pts, widths, true_dmap, img_size):
    """
    Compute gradient of MSE loss w.r.t. widths using autograd.

    Loss = MSE(render_ribbon_distance_map(pts, widths), true_dmap)
    Returns grad_widths, loss, pred_dmap.
    """
    pred_dmap = render_ribbon_distance_map(pts, widths, img_size)
    loss = torch.mean((pred_dmap - true_dmap) ** 2)
    grad_widths = torch.autograd.grad(loss, widths, create_graph=False)[0]
    return grad_widths, loss, pred_dmap


def _sample_grad_at_points(grad, pts):
    # Unused in autograd-based test; kept for reference
    H, W = grad.shape[1:]
    scale = torch.tensor([H - 1.0, W - 1.0], device=pts.device, dtype=pts.dtype)
    sp = 2.0 * pts / scale - 1.0
    grid = torch.stack([sp[:, 1], sp[:, 0]], dim=1).view(1, -1, 1, 2)
    g = F.grid_sample(grad.unsqueeze(0), grid, align_corners=True)
    g = g.view(2, -1).t()
    return g


def _width_smoothness_term(widths):
    # Unused in autograd-based test; kept for reference
    w = widths.view(1, 1, -1)
    kernel = torch.tensor([1.0, -4.0, 6.0, -4.0, 1.0], device=widths.device, dtype=widths.dtype)
    k = kernel.view(1, 1, 5)
    smooth = F.conv1d(w, k, padding=2).view(-1)
    if widths.numel() > 1:
        smooth[0] = widths[0] - widths[1]
        smooth[-1] = widths[-1] - widths[-2]
    return smooth


def rib_style_width_step(pts, widths, abs_dmap, width_stepsz=2.5):
    # Not used in yesterday's simple test; kept in file for reference only
    normals = _compute_normals_along_polyline(pts)
    half_r = (widths * 0.5).unsqueeze(1)
    left_pts = pts - normals * half_r
    right_pts = pts + normals * half_r
    # Placeholder: direct no-op in simple version
    return widths


def test_width_convergence(true_width, init_width, img_size=64, n_steps=100, 
                           lr=0.1, orientation='horizontal'):
    """
    Test if width optimization converges from init_width to true_width.
    """
    print(f"{'='*60}")
    print(f"Testing: True Width={true_width}px, Init Width={init_width}px")
    print(f"Orientation: {orientation}")
    print(f"{'='*60}\n")
    
    # Create synthetic data
    image, true_dmap_np, mask, centerline = create_synthetic_line(
        img_size=img_size, 
        line_width=true_width, 
        orientation=orientation
    )
    
    # Convert to torch
    true_dmap = torch.from_numpy(true_dmap_np).float().to(device)
    pts = torch.from_numpy(centerline).float().to(device)
    
    # Initialize widths (with gradient tracking for autograd)
    N = len(centerline)
    widths = torch.full((N,), float(init_width), dtype=torch.float32, device=device, requires_grad=True)
    
    # Store initial state
    with torch.no_grad():
        initial_dmap = render_ribbon_distance_map(pts, widths, img_size).cpu().numpy()
    
    # Optimization loop (autograd)
    width_history = [widths.detach().cpu().numpy().copy()]
    loss_history = []
    
    print("Starting optimization...")
    for step in range(n_steps):
        if widths.grad is not None:
            widths.grad.zero_()
        grad_w, loss, pred_dmap = compute_width_gradient(pts, widths, true_dmap, img_size)
        loss_history.append(loss.item())
        with torch.no_grad():
            widths.data = widths.data - lr * grad_w
            widths.data = torch.clamp(widths.data, min=1.0, max=15.0)
        
        # Record width
        current_widths = widths.detach().cpu().numpy().copy()
        width_history.append(current_widths)
        
        if step % 20 == 0 or step == n_steps - 1:
            mean_width = current_widths.mean()
            print(f"  Step {step:3d}: Loss={loss.item():.6f}, Mean Width={mean_width:.3f}px")
    
    # Get final state
    with torch.no_grad():
        final_dmap = render_ribbon_distance_map(pts, widths, img_size).cpu().numpy()
    final_widths = widths.detach().cpu().numpy()
    
    print(f"\nResults:")
    print(f"  True Width:    {true_width:.3f}px")
    print(f"  Initial Width: {width_history[0].mean():.3f}px")
    print(f"  Final Width:   {final_widths.mean():.3f}px (±{final_widths.std():.3f})")
    print(f"  Width Change:  {final_widths.mean() - width_history[0].mean():+.3f}px")
    
    return {
        'image': image,
        'true_dmap': true_dmap_np,
        'initial_dmap': initial_dmap,
        'final_dmap': final_dmap,
        'true_width': true_width,
        'init_width': init_width,
        'width_history': np.array(width_history),
        'loss_history': loss_history,
        'orientation': orientation
    }


def visualize_results(results, save_path='./width_convergence_test.png'):
    """Create comprehensive visualization of width convergence test."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    true_width = results['true_width']
    init_width = results['init_width']
    orientation = results['orientation']
    
    fig.suptitle(f'Width Convergence Test: {orientation.capitalize()}, '
                 f'True={true_width}px, Init={init_width}px', 
                 fontsize=16, fontweight='bold')
    
    # === ROW 1: INITIAL STATE ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(results['image'], cmap='gray', origin='lower')
    ax1.set_title('Synthetic Image')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(results['true_dmap'], cmap='RdBu_r', origin='lower', vmin=-10, vmax=10)
    ax2.set_title(f'True Distance Map\n(Width={true_width}px)')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(results['initial_dmap'], cmap='RdBu_r', origin='lower', vmin=-10, vmax=10)
    initial_mean = results['width_history'][0].mean()
    ax3.set_title(f'Initial Ribbon DMap\n(Width={initial_mean:.2f}px)')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # === ROW 2: FINAL STATE ===
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(results['image'], cmap='gray', origin='lower')
    ax4.set_title('Synthetic Image')
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(results['true_dmap'], cmap='RdBu_r', origin='lower', vmin=-10, vmax=10)
    ax5.set_title(f'True Distance Map\n(Width={true_width}px)')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(results['final_dmap'], cmap='RdBu_r', origin='lower', vmin=-10, vmax=10)
    final_mean = results['width_history'][-1].mean()
    ax6.set_title(f'Final Ribbon DMap\n(Width={final_mean:.2f}px)')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    # === ROW 3: WIDTH EVOLUTION ===
    width_history = results['width_history']
    n_steps = len(width_history)
    
    ax7 = fig.add_subplot(gs[2, :])
    
    # Plot mean width over time
    mean_widths = width_history.mean(axis=1)
    std_widths = width_history.std(axis=1)
    steps = np.arange(n_steps)
    
    ax7.plot(steps, mean_widths, 'b-', linewidth=2, label=f'Mean Width')
    ax7.fill_between(steps, mean_widths - std_widths, mean_widths + std_widths, 
                     alpha=0.3, color='blue', label='±1 Std')
    ax7.axhline(y=true_width, color='g', linestyle='--', linewidth=2, label=f'True Width ({true_width}px)')
    ax7.axhline(y=init_width, color='r', linestyle=':', linewidth=2, label=f'Initial Width ({init_width}px)')
    
    ax7.set_xlabel('Optimization Step', fontsize=12)
    ax7.set_ylabel('Width (pixels)', fontsize=12)
    ax7.set_title('Width Evolution During Optimization', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.legend(loc='best', fontsize=10)
    
    # Add text summary
    final_mean = mean_widths[-1]
    width_change = final_mean - init_width
    convergence = abs(final_mean - true_width)
    
    summary_text = (f"Initial: {init_width:.2f}px → Final: {final_mean:.2f}px\n"
                   f"Change: {width_change:+.2f}px | Error: {convergence:.2f}px")
    ax7.text(0.98, 0.02, summary_text, transform=ax7.transAxes,
            fontsize=11, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {save_path}\n")
    plt.close()


def main():
    """Run width convergence tests."""
    print("\n" + "="*60)
    print("SIMPLIFIED WIDTH CONVERGENCE TEST")
    print("="*60 + "\n")
    
    # Test parameters
    img_size = 64
    n_steps = 200
    lr = 2.5
    
    # Test Case 1: Thick vessel, start with thin width (should grow)
    print("### TEST CASE 1: Width should INCREASE (3 → 7) ###\n")
    results1 = test_width_convergence(
        true_width=7,
        init_width=3,
        img_size=img_size,
        n_steps=n_steps,
        lr=lr,
        orientation='horizontal'
    )
    visualize_results(results1, save_path='./width_convergence_grow.png')
    
    # Test Case 2: Thin vessel, start with thick width (should shrink)
    print("\n### TEST CASE 2: Width should DECREASE (7 → 3) ###\n")
    results2 = test_width_convergence(
        true_width=3,
        init_width=7,
        img_size=img_size,
        n_steps=n_steps,
        lr=lr,
        orientation='horizontal'
    )
    visualize_results(results2, save_path='./width_convergence_shrink.png')
    
    # Test Case 3: Vertical grow (3 → 7)
    print("\n### TEST CASE 3: Vertical Width should INCREASE (3 → 7) ###\n")
    results3 = test_width_convergence(
        true_width=7,
        init_width=3,
        img_size=img_size,
        n_steps=n_steps,
        lr=lr,
        orientation='vertical'
    )
    visualize_results(results3, save_path='./width_convergence_vertical_grow.png')

    # Test Case 4: Vertical shrink (7 → 3)
    print("\n### TEST CASE 4: Vertical Width should DECREASE (7 → 3) ###\n")
    results4 = test_width_convergence(
        true_width=3,
        init_width=7,
        img_size=img_size,
        n_steps=n_steps,
        lr=lr,
        orientation='vertical'
    )
    visualize_results(results4, save_path='./width_convergence_vertical_shrink.png')
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60 + "\n")
    
    # Analyze results
    case1_converged = abs(results1['width_history'][-1].mean() - 7.0) < 1.0
    case2_converged = abs(results2['width_history'][-1].mean() - 3.0) < 1.0
    
    print(f"Case 1 (3→7): {'✓ CONVERGED' if case1_converged else '✗ FAILED'}")
    print(f"  Final width: {results1['width_history'][-1].mean():.2f}px (target: 7.00px)")
    
    print(f"\nCase 2 (7→3): {'✓ CONVERGED' if case2_converged else '✗ FAILED'}")
    print(f"  Final width: {results2['width_history'][-1].mean():.2f}px (target: 3.00px)")
    
    if case1_converged and case2_converged:
        print("\n🎉 SUCCESS: Width optimization is working correctly!")
    else:
        print("\n⚠️  WARNING: Width optimization may not be working as expected.")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()


