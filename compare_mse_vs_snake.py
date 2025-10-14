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
snake_checkpoint = "/content/drive/MyDrive/ribbs/october/RibbonSertac/checkpoint_epoch_900_snake.pth"

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

# Process each training sample (cropped)
output_dir = "./comparison_plots"
utils.mkdir(output_dir)

print(f"\nGenerating comparisons for {min(10, len(dataloader_train))} cropped samples...")

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
        final_snake_dm = snake_loss.snake_dm  # Get optimized snake distance map
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Convert to numpy
        img_np = utils.from_torch(image[0].cpu())[0]
        label_np = utils.from_torch(label[0].cpu())[0]
        pred_mse_np = utils.from_torch(pred_mse[0].cpu())[0]
        pred_snake_np = utils.from_torch(pred_snake[0].cpu())[0]
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
            final_snake_binary = (final_snake_np[0] < 0).astype(np.uint8)
            # Also extract 2D slices for display
            label_display = label_np[0]
            pred_mse_display = pred_mse_np[0]
            pred_snake_display = pred_snake_np[0]
            final_snake_display = final_snake_np[0]
            img_display = img_np[0] if len(img_np.shape) == 3 else img_np
        else:  # (H, W)
            gt_binary = (label_np < 0).astype(np.uint8)
            mse_binary = (pred_mse_np < 0).astype(np.uint8)
            snake_binary = (pred_snake_np < 0).astype(np.uint8)
            final_snake_binary = (final_snake_np < 0).astype(np.uint8)
            label_display = label_np
            pred_mse_display = pred_mse_np
            pred_snake_display = pred_snake_np
            final_snake_display = final_snake_np
            img_display = img_np
        
        # Binary vessels show width information (no skeletonization)
        
        # Create 2-row, 5-column plot
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        fig.suptitle(f"MSE vs Snake Comparison - Crop {idx}", fontsize=16)
        
        # === ROW 1: DISTANCE MAPS ===
        axes[0,0].imshow(img_display, cmap='gray', origin='lower')
        axes[0,0].set_title('Input Image')
        axes[0,0].axis('off')
        
        im1 = axes[0,1].imshow(label_display, cmap='RdBu_r', origin='lower', vmin=-15, vmax=15)
        axes[0,1].set_title(f'GT Signed DMap\n(min:{label_display.min():.1f}, max:{label_display.max():.1f})')
        axes[0,1].axis('off')
        plt.colorbar(im1, ax=axes[0,1], fraction=0.046)
        
        im2 = axes[0,2].imshow(pred_mse_display, cmap='RdBu_r', origin='lower', vmin=-15, vmax=15)
        axes[0,2].set_title(f'MSE Prediction\n(min:{pred_mse_display.min():.1f}, max:{pred_mse_display.max():.1f})')
        axes[0,2].axis('off')
        plt.colorbar(im2, ax=axes[0,2], fraction=0.046)
        
        im3 = axes[0,3].imshow(pred_snake_display, cmap='RdBu_r', origin='lower', vmin=-15, vmax=15)
        axes[0,3].set_title(f'Snake Prediction\n(min:{pred_snake_display.min():.1f}, max:{pred_snake_display.max():.1f})')
        axes[0,3].axis('off')
        plt.colorbar(im3, ax=axes[0,3], fraction=0.046)
        
        im4 = axes[0,4].imshow(final_snake_display, cmap='RdBu_r', origin='lower', vmin=-15, vmax=15)
        axes[0,4].set_title(f'Final Snake DMap\n(min:{final_snake_display.min():.1f}, max:{final_snake_display.max():.1f})')
        axes[0,4].axis('off')
        plt.colorbar(im4, ax=axes[0,4], fraction=0.046)
        
        # === ROW 2: BINARY VESSELS (WITH WIDTH) ===
        axes[1,0].imshow(img_display, cmap='gray', origin='lower')
        axes[1,0].set_title('Input Image')
        axes[1,0].axis('off')
        
        axes[1,1].imshow(gt_binary, cmap='gray', origin='lower')
        axes[1,1].set_title(f'GT Binary Vessels\n({gt_binary.sum()} pixels)')
        axes[1,1].axis('off')
        
        axes[1,2].imshow(mse_binary, cmap='gray', origin='lower')
        axes[1,2].set_title(f'MSE Binary Vessels\n({mse_binary.sum()} pixels)')
        axes[1,2].axis('off')
        
        axes[1,3].imshow(snake_binary, cmap='gray', origin='lower')
        axes[1,3].set_title(f'Snake Binary Vessels\n({snake_binary.sum()} pixels)')
        axes[1,3].axis('off')
        
        axes[1,4].imshow(final_snake_binary, cmap='gray', origin='lower')
        axes[1,4].set_title(f'Final Snake Binary\n({final_snake_binary.sum()} pixels)')
        axes[1,4].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comparison_crop_{idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved comparison for crop {idx}")

print(f"\n✓ All comparisons saved to {output_dir}/")
print("\nTo download:")
print("!zip -r comparison_plots.zip ./comparison_plots")
print("from google.colab import files")
print("files.download('comparison_plots.zip')")
