import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('/content/Ribbon')

from Codes.dataset import DriveDataset, collate_fn
from Codes.network import UNet
from Codes.Losses.losses import SnakeSimpleLoss
from Codes import utils
from torch.utils.data import DataLoader
from skimage.morphology import skeletonize

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

# Configuration
config = utils.yaml_read('main.config')

# Checkpoint paths
mse_checkpoint = "/content/drive/MyDrive/ribbs/october/RibbonSertac/checkpoint_epoch_900_mse.pth"
snake_checkpoint = "/content/drive/MyDrive/ribbs/october/RibbonSertac/2025-10-15_13-32-50- fixedwidth after 900/checkpoint_epoch_1500_snake.pth"

# Load training dataset with crops (same as training)
crop_size = tuple(config["crop_size"])
dataset_train = DriveDataset(train=True, cropSize=crop_size)
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Create networks
network_mse = UNet(in_channels=config["in_channels"],
                   m_channels=config["m_channels"],
                   out_channels=config["num_classes"],
                   n_convs=config["n_convs"],
                   n_levels=config["n_levels"],
                   dropout=config["dropout"],
                   batch_norm=config["batch_norm"],
                   upsampling=config["upsampling"],
                   pooling=config["pooling"],
                   three_dimensional=config["three_dimensional"]).cuda()

network_snake = UNet(in_channels=config["in_channels"],
                     m_channels=config["m_channels"],
                     out_channels=config["num_classes"],
                     n_convs=config["n_convs"],
                     n_levels=config["n_levels"],
                     dropout=config["dropout"],
                     batch_norm=config["batch_norm"],
                     upsampling=config["upsampling"],
                     pooling=config["pooling"],
                     three_dimensional=config["three_dimensional"]).cuda()

# Load checkpoints
print("Loading MSE checkpoint...")
network_mse.load_state_dict(torch.load(mse_checkpoint))
network_mse.eval()

print("Loading Snake checkpoint...")
network_snake.load_state_dict(torch.load(snake_checkpoint))
network_snake.eval()

# Initialize snake loss for final snake distance map
snake_loss = SnakeSimpleLoss(
    stepsz=config["stepsz"],
    alpha=config["alpha"],
    beta=config["beta"],
    fltrstdev=config["fltrstdev"],
    ndims=config["ndims"],
    nsteps=config["nsteps"],
    cropsz=config["cropsz"],
    dmax=config["dmax"],
    maxedgelen=config["maxedgelength"],
    extgradfac=config["extgradfac"]
).cuda()

def create_overlay(gt_binary, pred_binary):
    """
    Create RGB overlay image:
    - Green: True Positives (both GT and prediction have vessels)
    - Red: False Negatives (GT has vessels, prediction missed)
    - Blue: False Positives (prediction has vessels, GT doesn't)
    - Black: True Negatives (neither has vessels)
    """
    h, w = gt_binary.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # True Positives: Green
    tp_mask = (gt_binary == 1) & (pred_binary == 1)
    overlay[tp_mask] = [0, 255, 0]
    
    # False Negatives: Red
    fn_mask = (gt_binary == 1) & (pred_binary == 0)
    overlay[fn_mask] = [255, 0, 0]
    
    # False Positives: Blue
    fp_mask = (gt_binary == 0) & (pred_binary == 1)
    overlay[fp_mask] = [0, 0, 255]
    
    # True Negatives: Black (already zeros)
    
    return overlay, tp_mask.sum(), fn_mask.sum(), fp_mask.sum()

def calculate_metrics(tp, fn, fp):
    """
    Calculate precision, recall, F1 score, and quality metrics
    """
    # Avoid division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Quality metric (similar to IoU)
    quality = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0
    
    return precision, recall, f1, quality

# Process each training sample (cropped)
output_dir = "./comparison_plots"
utils.mkdir(output_dir)

print(f"\nGenerating comparisons for {min(10, len(dataloader_train))} cropped samples...")

# Store aggregate metrics
all_metrics = {
    'mse': {'tp': [], 'fn': [], 'fp': [], 'precision': [], 'recall': [], 'f1': [], 'quality': []},
    'snake_pred': {'tp': [], 'fn': [], 'fp': [], 'precision': [], 'recall': [], 'f1': [], 'quality': []},
    'snake_initial': {'tp': [], 'fn': [], 'fp': [], 'precision': [], 'recall': [], 'f1': [], 'quality': []},
    'snake_final': {'tp': [], 'fn': [], 'fp': [], 'precision': [], 'recall': [], 'f1': [], 'quality': []}
}

with torch.no_grad():
    for idx, data_batch in enumerate(dataloader_train):
        if idx >= 10:  # Limit to first 10 samples
            break
            
        image, label, graphs, slices, original_shape = data_batch
        image = image.cuda()
        label = label.cuda()
        
        # Get predictions on crops (fast, no chunking needed)
        pred_mse = network_mse(image)
        pred_snake = network_snake(image)
        
        # Run snake optimization on snake prediction to get final snake distance map
        snake_loss(pred_snake, graphs, slices, None, original_shape)
        initial_snake_dm = snake_loss.snake_dm_initial  # Get initial snake (before optimization)
        final_snake_dm = snake_loss.snake_dm  # Get optimized snake distance map
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Convert to numpy
        img_np = utils.from_torch(image[0].cpu())[0]
        label_np = utils.from_torch(label[0].cpu())[0]
        pred_mse_np = utils.from_torch(pred_mse[0].cpu())[0]
        pred_snake_np = utils.from_torch(pred_snake[0].cpu())[0]
        initial_snake_np = utils.from_torch(initial_snake_dm[0].cpu())[0]
        final_snake_np = utils.from_torch(final_snake_dm[0].cpu())[0]
        
        # Debug: Print shapes
        if idx == 0:
            print(f"Shapes - img:{img_np.shape}, label:{label_np.shape}, pred_mse:{pred_mse_np.shape}")
        
        # Create binarized versions (threshold at 0 for vessels)
        # Handle both (C, H, W) and (H, W) formats
        if len(label_np.shape) == 3:  # (C, H, W)
            gt_binary = (label_np[0] < 0).astype(np.uint8)
            mse_binary = (pred_mse_np[0] < 0).astype(np.uint8)
            snake_binary = (pred_snake_np[0] < 0).astype(np.uint8)
            initial_snake_binary = (initial_snake_np[0] < 0).astype(np.uint8)
            final_snake_binary = (final_snake_np[0] < 0).astype(np.uint8)
            # Also extract 2D slices for display
            label_display = label_np[0]
            pred_mse_display = pred_mse_np[0]
            pred_snake_display = pred_snake_np[0]
            initial_snake_display = initial_snake_np[0]
            final_snake_display = final_snake_np[0]
            img_display = img_np[0] if len(img_np.shape) == 3 else img_np
        else:  # (H, W)
            gt_binary = (label_np < 0).astype(np.uint8)
            mse_binary = (pred_mse_np < 0).astype(np.uint8)
            snake_binary = (pred_snake_np < 0).astype(np.uint8)
            initial_snake_binary = (initial_snake_np < 0).astype(np.uint8)
            final_snake_binary = (final_snake_np < 0).astype(np.uint8)
            label_display = label_np
            pred_mse_display = pred_mse_np
            pred_snake_display = pred_snake_np
            initial_snake_display = initial_snake_np
            final_snake_display = final_snake_np
            img_display = img_np
        
        # Create overlays
        mse_overlay, mse_tp, mse_fn, mse_fp = create_overlay(gt_binary, mse_binary)
        snake_pred_overlay, snake_pred_tp, snake_pred_fn, snake_pred_fp = create_overlay(gt_binary, snake_binary)
        initial_overlay, initial_tp, initial_fn, initial_fp = create_overlay(gt_binary, initial_snake_binary)
        final_overlay, final_tp, final_fn, final_fp = create_overlay(gt_binary, final_snake_binary)
        
        # Calculate metrics
        mse_prec, mse_rec, mse_f1, mse_qual = calculate_metrics(mse_tp, mse_fn, mse_fp)
        snake_pred_prec, snake_pred_rec, snake_pred_f1, snake_pred_qual = calculate_metrics(snake_pred_tp, snake_pred_fn, snake_pred_fp)
        initial_prec, initial_rec, initial_f1, initial_qual = calculate_metrics(initial_tp, initial_fn, initial_fp)
        final_prec, final_rec, final_f1, final_qual = calculate_metrics(final_tp, final_fn, final_fp)
        
        # Store metrics
        all_metrics['mse']['tp'].append(mse_tp)
        all_metrics['mse']['fn'].append(mse_fn)
        all_metrics['mse']['fp'].append(mse_fp)
        all_metrics['mse']['precision'].append(mse_prec)
        all_metrics['mse']['recall'].append(mse_rec)
        all_metrics['mse']['f1'].append(mse_f1)
        all_metrics['mse']['quality'].append(mse_qual)
        
        all_metrics['snake_pred']['tp'].append(snake_pred_tp)
        all_metrics['snake_pred']['fn'].append(snake_pred_fn)
        all_metrics['snake_pred']['fp'].append(snake_pred_fp)
        all_metrics['snake_pred']['precision'].append(snake_pred_prec)
        all_metrics['snake_pred']['recall'].append(snake_pred_rec)
        all_metrics['snake_pred']['f1'].append(snake_pred_f1)
        all_metrics['snake_pred']['quality'].append(snake_pred_qual)
        
        all_metrics['snake_initial']['tp'].append(initial_tp)
        all_metrics['snake_initial']['fn'].append(initial_fn)
        all_metrics['snake_initial']['fp'].append(initial_fp)
        all_metrics['snake_initial']['precision'].append(initial_prec)
        all_metrics['snake_initial']['recall'].append(initial_rec)
        all_metrics['snake_initial']['f1'].append(initial_f1)
        all_metrics['snake_initial']['quality'].append(initial_qual)
        
        all_metrics['snake_final']['tp'].append(final_tp)
        all_metrics['snake_final']['fn'].append(final_fn)
        all_metrics['snake_final']['fp'].append(final_fp)
        all_metrics['snake_final']['precision'].append(final_prec)
        all_metrics['snake_final']['recall'].append(final_rec)
        all_metrics['snake_final']['f1'].append(final_f1)
        all_metrics['snake_final']['quality'].append(final_qual)
        
        # Create 3-row, 6-column plot
        fig, axes = plt.subplots(3, 6, figsize=(30, 18))
        fig.suptitle(f"MSE vs Snake Comparison - Crop {idx}", fontsize=18, fontweight='bold')
        
        # === ROW 1: DISTANCE MAPS ===
        axes[0,0].imshow(img_display, cmap='gray', origin='lower')
        axes[0,0].set_title('Input Image', fontsize=11)
        axes[0,0].axis('off')
        
        im1 = axes[0,1].imshow(label_display, cmap='RdBu_r', origin='lower', vmin=-15, vmax=15)
        axes[0,1].set_title(f'GT Signed DMap\n(min:{label_display.min():.1f}, max:{label_display.max():.1f})', fontsize=11)
        axes[0,1].axis('off')
        plt.colorbar(im1, ax=axes[0,1], fraction=0.046)
        
        im2 = axes[0,2].imshow(pred_mse_display, cmap='RdBu_r', origin='lower', vmin=-15, vmax=15)
        axes[0,2].set_title(f'MSE Prediction\n(min:{pred_mse_display.min():.1f}, max:{pred_mse_display.max():.1f})', fontsize=11)
        axes[0,2].axis('off')
        plt.colorbar(im2, ax=axes[0,2], fraction=0.046)
        
        im3 = axes[0,3].imshow(pred_snake_display, cmap='RdBu_r', origin='lower', vmin=-15, vmax=15)
        axes[0,3].set_title(f'Snake Prediction (Model)\n(min:{pred_snake_display.min():.1f}, max:{pred_snake_display.max():.1f})', fontsize=11)
        axes[0,3].axis('off')
        plt.colorbar(im3, ax=axes[0,3], fraction=0.046)
        
        im4 = axes[0,4].imshow(initial_snake_display, cmap='RdBu_r', origin='lower', vmin=-15, vmax=15)
        axes[0,4].set_title(f'Snake BEFORE Opt\n(min:{initial_snake_display.min():.1f}, max:{initial_snake_display.max():.1f})', fontsize=11)
        axes[0,4].axis('off')
        plt.colorbar(im4, ax=axes[0,4], fraction=0.046)
        
        im5 = axes[0,5].imshow(final_snake_display, cmap='RdBu_r', origin='lower', vmin=-15, vmax=15)
        axes[0,5].set_title(f'Snake AFTER Opt\n(min:{final_snake_display.min():.1f}, max:{final_snake_display.max():.1f})', fontsize=11)
        axes[0,5].axis('off')
        plt.colorbar(im5, ax=axes[0,5], fraction=0.046)
        
        # === ROW 2: BINARY VESSELS (WITH WIDTH) ===
        axes[1,0].imshow(img_display, cmap='gray', origin='lower')
        axes[1,0].set_title('Input Image', fontsize=11)
        axes[1,0].axis('off')
        
        axes[1,1].imshow(gt_binary, cmap='gray', origin='lower')
        axes[1,1].set_title(f'GT Binary Vessels\n({gt_binary.sum()} pixels)', fontsize=11)
        axes[1,1].axis('off')
        
        axes[1,2].imshow(mse_binary, cmap='gray', origin='lower')
        axes[1,2].set_title(f'MSE Binary\n({mse_binary.sum()} pixels)', fontsize=11)
        axes[1,2].axis('off')
        
        axes[1,3].imshow(snake_binary, cmap='gray', origin='lower')
        axes[1,3].set_title(f'Snake Pred Binary\n({snake_binary.sum()} pixels)', fontsize=11)
        axes[1,3].axis('off')
        
        axes[1,4].imshow(initial_snake_binary, cmap='gray', origin='lower')
        axes[1,4].set_title(f'Snake BEFORE Binary\n({initial_snake_binary.sum()} pixels)', fontsize=11)
        axes[1,4].axis('off')
        
        axes[1,5].imshow(final_snake_binary, cmap='gray', origin='lower')
        axes[1,5].set_title(f'Snake AFTER Binary\n({final_snake_binary.sum()} pixels)', fontsize=11)
        axes[1,5].axis('off')
        
        # === ROW 3: OVERLAYS WITH METRICS ===
        axes[2,0].imshow(img_display, cmap='gray', origin='lower')
        axes[2,0].set_title('Input Image\n(Reference)', fontsize=11)
        axes[2,0].axis('off')
        
        # Legend
        axes[2,1].axis('off')
        legend_text = ('Color Legend:\n'
                      '🟢 Green = TP\n'
                      '🔴 Red = FN\n'
                      '🔵 Blue = FP\n'
                      '⬛ Black = TN')
        axes[2,1].text(0.5, 0.5, legend_text, ha='center', va='center', 
                      fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[2,2].imshow(mse_overlay, origin='lower')
        axes[2,2].set_title(f'MSE vs GT\nTP:{mse_tp} FN:{mse_fn} FP:{mse_fp}\nPrec:{mse_prec:.3f} Rec:{mse_rec:.3f}\nF1:{mse_f1:.3f} Qual:{mse_qual:.3f}', 
                           fontsize=10)
        axes[2,2].axis('off')
        
        axes[2,3].imshow(snake_pred_overlay, origin='lower')
        axes[2,3].set_title(f'Snake Pred vs GT\nTP:{snake_pred_tp} FN:{snake_pred_fn} FP:{snake_pred_fp}\nPrec:{snake_pred_prec:.3f} Rec:{snake_pred_rec:.3f}\nF1:{snake_pred_f1:.3f} Qual:{snake_pred_qual:.3f}', 
                           fontsize=10)
        axes[2,3].axis('off')
        
        axes[2,4].imshow(initial_overlay, origin='lower')
        axes[2,4].set_title(f'Snake BEFORE vs GT\nTP:{initial_tp} FN:{initial_fn} FP:{initial_fp}\nPrec:{initial_prec:.3f} Rec:{initial_rec:.3f}\nF1:{initial_f1:.3f} Qual:{initial_qual:.3f}', 
                           fontsize=10)
        axes[2,4].axis('off')
        
        axes[2,5].imshow(final_overlay, origin='lower')
        axes[2,5].set_title(f'Snake AFTER vs GT\nTP:{final_tp} FN:{final_fn} FP:{final_fp}\nPrec:{final_prec:.3f} Rec:{final_rec:.3f}\nF1:{final_f1:.3f} Qual:{final_qual:.3f}', 
                           fontsize=10)
        axes[2,5].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comparison_crop_{idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved comparison for crop {idx}")

# Print aggregate statistics
print("\n" + "="*80)
print("AGGREGATE METRICS ACROSS ALL SAMPLES")
print("="*80)

for method_name, method_data in all_metrics.items():
    print(f"\n{method_name.upper().replace('_', ' ')}:")
    print(f"  Precision: {np.mean(method_data['precision']):.4f} ± {np.std(method_data['precision']):.4f}")
    print(f"  Recall:    {np.mean(method_data['recall']):.4f} ± {np.std(method_data['recall']):.4f}")
    print(f"  F1 Score:  {np.mean(method_data['f1']):.4f} ± {np.std(method_data['f1']):.4f}")
    print(f"  Quality:   {np.mean(method_data['quality']):.4f} ± {np.std(method_data['quality']):.4f}")

print(f"\n✓ All comparisons saved to {output_dir}/")
print("\nTo download:")
print("!zip -r comparison_plots.zip ./comparison_plots")
print("from google.colab import files")
print("files.download('comparison_plots.zip')")
