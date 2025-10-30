"""
Simplified test to verify width gradient computation and convergence (Rib-style update).
This mirrors RibbonSnake.step_widths logic in a standalone script for synthetic lines.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import distance_transform_edt
import torch.nn.functional as F
from functools import reduce

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
    """
    device = pts.device
    y_coords = torch.arange(img_size, dtype=torch.float32, device=device)
    x_coords = torch.arange(img_size, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    grid = torch.stack([yy, xx], dim=-1)  # (H, W, 2)
    distances_to_pts = torch.norm(grid.unsqueeze(2) - pts.unsqueeze(0).unsqueeze(0), dim=-1)
    min_dist_to_centerline, closest_idx = torch.min(distances_to_pts, dim=2)
    width_at_pixel = widths[closest_idx]
    distance_map = min_dist_to_centerline - width_at_pixel / 2.0
    return distance_map


def _compute_normals_along_polyline(pts):
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


def makeGaussEdgeFltr(stdev, d):
    """
    Make a Gaussian-derivative-based edge filter (same as gradRib.py).
    Returns fltr: (d, 1, k, k) for 2D or (d, 1, k, k, k) for 3D.
    """
    fsz = round(2 * stdev) * 2 + 1
    n = np.arange(0, fsz).astype(float) - (fsz - 1) / 2.0
    s2 = stdev * stdev
    v = np.exp(-(n ** 2) / (2 * s2))
    g = n / s2 * v
    shps = np.eye(d, dtype=int) * (fsz - 1) + 1
    reshaped = [x.reshape(y) for x, y in zip([g] + [v] * (d - 1), shps)]
    fltr = reduce(np.multiply, reshaped)
    fltr = fltr / np.sum(np.abs(fltr))
    fltr_ = fltr[np.newaxis, np.newaxis]
    fltr_multidir = np.concatenate([np.moveaxis(fltr_, 2, k) for k in range(2, 2 + d)], axis=0)
    return fltr_multidir


def cmptGradIm(img, fltr):
    """
    Convolves img with fltr using replication padding (same as gradRib.py).
    img: (1, 1, H, W) tensor
    fltr: (2, 1, k, k) tensor
    Returns: (2, H, W) gradient image
    """
    if img.dim() == 4:
        img_p = torch.nn.ReplicationPad2d(fltr.shape[2] // 2).forward(img)
        return torch.nn.functional.conv2d(img_p, fltr).squeeze(0)
    else:
        raise ValueError("img should have 4 dimensions")


def cmptExtGrad(snakepos, eGradIm):
    """
    Sample gradient image at snake positions (exact copy from gradRib.py).
    snakepos: (N, d) tensor of positions
    eGradIm: (d, H, W) gradient image for 2D or (d, H, W, D) for 3D
    Returns: (N, d) sampled gradients
    """
    scale = torch.tensor(eGradIm.shape[1:]).reshape((1, -1)).type_as(snakepos) - 1.0
    sp = 2 * snakepos / scale - 1.0
    
    if eGradIm.shape[0] == 3:
        spi = torch.einsum('km,md->kd', [sp, torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]]).type_as(sp).to(sp.device)])
        egrad = torch.nn.functional.grid_sample(eGradIm[None], spi[None, None, None], align_corners=True)
        egrad = egrad.permute(0, 2, 3, 4, 1)
    if eGradIm.shape[0] == 2:
        spi = torch.einsum('kl,ld->kd', [sp, torch.tensor([[0, 1], [1, 0]]).type_as(sp).to(sp.device)])
        egrad = torch.nn.functional.grid_sample(eGradIm[None], spi[None, None], align_corners=True)
        egrad = egrad.permute(0, 2, 3, 1)
        
    return egrad.reshape_as(snakepos)


def _edge_gradient_from_abs_dmap(dmap, fltr_stdev=1.0):
    """
    Compute gradient of |dmap| using Gaussian-derivative filters (same as training).
    Returns tensor of shape (2, H, W): dy, dx.
    """
    img = torch.abs(dmap).unsqueeze(0).unsqueeze(0)  # 1 x 1 x H x W
    fltr_np = makeGaussEdgeFltr(fltr_stdev, 2)  # (2, 1, k, k)
    fltr_t = torch.from_numpy(fltr_np).type(img.dtype).to(img.device)
    grad = cmptGradIm(img, fltr_t)  # (2, H, W)
    return grad


def _sample_grad_at_points(grad, pts):
    H, W = grad.shape[1:]
    scale = torch.tensor([H - 1.0, W - 1.0], device=pts.device, dtype=pts.dtype)
    sp = 2.0 * pts / scale - 1.0
    grid = torch.stack([sp[:, 1], sp[:, 0]], dim=1).view(1, -1, 1, 2)
    g = F.grid_sample(grad.unsqueeze(0), grid, align_corners=True)
    g = g.view(2, -1).t()  # N x 2
    return g


def _width_smoothness_term(widths):
    w = widths.view(1, 1, -1)
    kernel = torch.tensor([1.0, -4.0, 6.0, -4.0, 1.0], device=widths.device, dtype=widths.dtype)
    k = kernel.view(1, 1, 5)
    smooth = F.conv1d(w, k, padding=2).view(-1)
    if widths.numel() > 1:
        smooth[0] = widths[0] - widths[1]
        smooth[-1] = widths[-1] - widths[-2]
    return smooth


def rib_style_width_step(pts, widths, abs_dmap, width_stepsz=2.5, endpoint_alpha_scale=0.3):
    normals = _compute_normals_along_polyline(pts)  # N x 2
    half_r = (widths * 0.5).unsqueeze(1)            # N x 1
    left_pts = pts - normals * half_r
    right_pts = pts + normals * half_r
    gimgW = _edge_gradient_from_abs_dmap(abs_dmap)  # 2 x H x W
    # Use cmptExtGrad from training code (same as rib.py)
    grad_L = cmptExtGrad(left_pts, gimgW)
    grad_R = cmptExtGrad(right_pts, gimgW)
    grad_w = ((grad_R - grad_L) * normals).sum(dim=1)  # (N,)
    internal = _width_smoothness_term(widths)
    alpha = grad_w.abs() / (internal.abs() + 1e-8)
    
    # Reduce smoothness weight for endpoints to let them converge independently
    N = len(widths)
    if N > 2:
        alpha[0] = alpha[0] * endpoint_alpha_scale
        alpha[-1] = alpha[-1] * endpoint_alpha_scale
    
    total = grad_w + alpha * internal
    new_widths = widths - width_stepsz * total
    return new_widths


def test_width_convergence(true_width, init_width, img_size=64, n_steps=100, 
                           lr=0.1, orientation='horizontal', width_stepsz=2.5, endpoint_alpha_scale=0.3):
    print(f"{'='*60}")
    print(f"Testing: True Width={true_width}px, Init Width={init_width}px")
    print(f"Orientation: {orientation}")
    print(f"{'='*60}\n")
    image, true_dmap_np, mask, centerline = create_synthetic_line(
        img_size=img_size, line_width=true_width, orientation=orientation
    )
    true_dmap = torch.from_numpy(true_dmap_np).float().to(device)
    pts = torch.from_numpy(centerline).float().to(device)
    N = len(centerline)
    widths = torch.full((N,), float(init_width), dtype=torch.float32, device=device)
    with torch.no_grad():
        initial_dmap = render_ribbon_distance_map(pts, widths, img_size).cpu().numpy()
    abs_dmap = torch.from_numpy(np.abs(true_dmap_np)).float().to(device)
    width_history = [widths.detach().cpu().numpy().copy()]
    loss_history = []
    print("Starting optimization...")
    for step in range(n_steps):
        pred_dmap = render_ribbon_distance_map(pts, widths, img_size)
        loss = torch.mean((pred_dmap - true_dmap) ** 2)
        loss_history.append(loss.item())
        with torch.no_grad():
            widths = rib_style_width_step(pts, widths, abs_dmap, width_stepsz=width_stepsz, endpoint_alpha_scale=endpoint_alpha_scale)
            widths = torch.clamp(widths, min=1.0, max=15.0)
        current_widths = widths.detach().cpu().numpy().copy()
        width_history.append(current_widths)
        if step % 20 == 0 or step == n_steps - 1:
            mean_width = current_widths.mean()
            print(f"  Step {step:3d}: Loss={loss.item():.6f}, Mean Width={mean_width:.3f}px")
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
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    true_width = results['true_width']
    init_width = results['init_width']
    orientation = results['orientation']
    fig.suptitle(f'Width Convergence Test: {orientation.capitalize()}, '
                 f'True={true_width}px, Init={init_width}px', 
                 fontsize=16, fontweight='bold')
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
    width_history = results['width_history']
    n_steps = len(width_history)
    ax7 = fig.add_subplot(gs[2, :])
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
    print("\n" + "="*60)
    print("SIMPLIFIED WIDTH CONVERGENCE TEST (RIB-STYLE)")
    print("="*60 + "\n")
    img_size = 64
    n_steps = 40
    width_stepsz = 0.2
    fltr_stdev = 1.0
    endpoint_alpha_scale = 0.3
    print(f"Parameters: n_steps={n_steps}, width_stepsz={width_stepsz}, fltr_stdev={fltr_stdev}, endpoint_alpha_scale={endpoint_alpha_scale}\n")
    print("### TEST CASE 1: Width should INCREASE (3 → 7) ###\n")
    results1 = test_width_convergence(
        true_width=7,
        init_width=3,
        img_size=img_size,
        n_steps=n_steps,
        lr=0.0,
        orientation='horizontal',
        width_stepsz=width_stepsz,
        endpoint_alpha_scale=endpoint_alpha_scale,
    )
    visualize_results(results1, save_path='./width_convergence_grow_rib.png')
    print("\n### TEST CASE 2: Width should DECREASE (7 → 3) ###\n")
    results2 = test_width_convergence(
        true_width=3,
        init_width=7,
        img_size=img_size,
        n_steps=n_steps,
        lr=0.0,
        orientation='horizontal',
        width_stepsz=width_stepsz,
        endpoint_alpha_scale=endpoint_alpha_scale,
    )
    visualize_results(results2, save_path='./width_convergence_shrink_rib.png')
    # Vertical cases
    print("\n### TEST CASE 3: Vertical Width should INCREASE (3 → 7) ###\n")
    results3 = test_width_convergence(
        true_width=7,
        init_width=3,
        img_size=img_size,
        n_steps=n_steps,
        lr=0.0,
        orientation='vertical',
        width_stepsz=width_stepsz,
        endpoint_alpha_scale=endpoint_alpha_scale,
    )
    visualize_results(results3, save_path='./width_convergence_vertical_grow_rib.png')
    print("\n### TEST CASE 4: Vertical Width should DECREASE (7 → 3) ###\n")
    results4 = test_width_convergence(
        true_width=3,
        init_width=7,
        img_size=img_size,
        n_steps=n_steps,
        lr=0.0,
        orientation='vertical',
        width_stepsz=width_stepsz,
        endpoint_alpha_scale=endpoint_alpha_scale,
    )
    visualize_results(results4, save_path='./width_convergence_vertical_shrink_rib.png')
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60 + "\n")
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


