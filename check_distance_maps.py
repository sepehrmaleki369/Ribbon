import numpy as np
import matplotlib.pyplot as plt
import os

# Check and visualize distance maps
data_dir = "/content/Ribbon/Codes/drive/training"

# Load a few samples
sample_ids = [21, 22, 23]

fig, axes = plt.subplots(len(sample_ids), 4, figsize=(20, 5*len(sample_ids)))
fig.suptitle('Distance Map Analysis - Check for Negative Values', fontsize=16)

for idx, sample_id in enumerate(sample_ids):
    # Load data
    image_path = f"{data_dir}/images_npy/{sample_id}_training.npy"
    dmap_path = f"{data_dir}/distance_maps/{sample_id}_training_distance_map.npy"
    signed_dmap_path = f"{data_dir}/signed_distance_maps/{sample_id}_training_signed_dmap.npy"
    inverted_path = f"{data_dir}/inverted_labels/{sample_id}_manual1.npy"
    
    image = np.load(image_path)
    dmap = np.load(dmap_path)
    signed_dmap = np.load(signed_dmap_path)
    inverted = np.load(inverted_path)
    
    # Convert RGB to grayscale
    if len(image.shape) == 3:
        image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    
    # Print statistics
    print(f"\n=== Sample {sample_id} ===")
    print(f"Image shape: {image.shape}")
    
    print(f"\nOld Distance Map - Min: {dmap.min():.2f}, Max: {dmap.max():.2f}")
    print(f"  Negative pixels: {(dmap < 0).sum()} ({100*(dmap < 0).sum()/dmap.size:.2f}%)")
    print(f"  Zero pixels: {(dmap == 0).sum()}")
    print(f"  Positive pixels: {(dmap > 0).sum()} ({100*(dmap > 0).sum()/dmap.size:.2f}%)")
    
    print(f"\n✓ NEW Signed Distance Map - Min: {signed_dmap.min():.2f}, Max: {signed_dmap.max():.2f}")
    print(f"  Negative pixels: {(signed_dmap < 0).sum()} ({100*(signed_dmap < 0).sum()/signed_dmap.size:.2f}%)")
    print(f"  Zero pixels: {(signed_dmap == 0).sum()}")
    print(f"  Positive pixels: {(signed_dmap > 0).sum()} ({100*(signed_dmap > 0).sum()/signed_dmap.size:.2f}%)")
    
    print(f"\nInverted Label - Min: {inverted.min():.2f}, Max: {inverted.max():.2f}")
    print(f"  Negative pixels: {(inverted < 0).sum()} ({100*(inverted < 0).sum()/inverted.size:.2f}%)")
    
    # Plot
    row = axes[idx] if len(sample_ids) > 1 else axes
    
    # 1. Image
    row[0].imshow(image, cmap='gray', origin='lower')
    row[0].set_title(f'Sample {sample_id}: Image')
    row[0].axis('off')
    
    # 2. Old Distance Map (no negatives)
    im1 = row[1].imshow(dmap, cmap='RdBu_r', origin='lower', vmin=-15, vmax=15)
    row[1].set_title(f'OLD Distance Map\nNeg:{(dmap<0).sum()} (BAD!)')
    row[1].axis('off')
    plt.colorbar(im1, ax=row[1], fraction=0.046)
    
    # 3. NEW Signed Distance Map (with negatives!)
    im2 = row[2].imshow(signed_dmap, cmap='RdBu_r', origin='lower', vmin=-15, vmax=15)
    row[2].set_title(f'NEW Signed DMap\nNeg:{(signed_dmap<0).sum()} (GOOD!)')
    row[2].axis('off')
    plt.colorbar(im2, ax=row[2], fraction=0.046)
    
    # 4. Inverted Label
    im3 = row[3].imshow(inverted, cmap='RdBu_r', origin='lower', vmin=-15, vmax=15)
    row[3].set_title(f'Inverted Label\nNeg:{(inverted<0).sum()}')
    row[3].axis('off')
    plt.colorbar(im3, ax=row[3], fraction=0.046)

plt.tight_layout()
plt.savefig('distance_map_check.png', dpi=150, bbox_inches='tight')
print("\n" + "="*60)
print("✓ Saved plot to: distance_map_check.png")
print("="*60)

# Display in Colab
from IPython.display import Image, display
display(Image('distance_map_check.png'))
