import numpy as np
from scipy.ndimage import distance_transform_edt
import os

# Create signed distance maps from binary masks
data_dir = "/content/Ribbon/Codes/drive/training"

# Process all training images (21-40)
for sample_id in range(21, 41):
    print(f"Processing sample {sample_id}...")
    
    # Load binary mask
    mask_path = f"{data_dir}/1st_manual_npy/{sample_id}_manual1.npy"
    mask = np.load(mask_path)
    
    # Binary mask: 1 = vessel, 0 = background
    # Invert for distance transform: True = background, False = vessel
    background = (mask == 0)
    vessel = (mask > 0)
    
    # Compute distance transforms
    dist_outside = distance_transform_edt(background)  # Distance from background to nearest vessel
    dist_inside = distance_transform_edt(vessel)       # Distance from vessel to nearest background
    
    # Create signed distance map
    # Inside vessels: negative
    # Outside vessels: positive
    signed_dmap = np.where(vessel, -dist_inside, dist_outside)
    
    # Enhance negative values to make vessels more prominent
    enhancement_factor = 2.0  # Make negatives 2x stronger
    signed_dmap = np.where(signed_dmap < 0, signed_dmap * enhancement_factor, signed_dmap)
    
    # Clip to max distance
    max_dist = 15
    signed_dmap = np.clip(signed_dmap, -max_dist, max_dist)
    
    # Print stats
    print(f"  Signed DMap - Min: {signed_dmap.min():.2f}, Max: {signed_dmap.max():.2f}")
    print(f"  Negative: {(signed_dmap < 0).sum()}, Zero: {(signed_dmap == 0).sum()}, Positive: {(signed_dmap > 0).sum()}")
    
    # Save
    output_dir = f"{data_dir}/signed_distance_maps"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{sample_id}_training_signed_dmap.npy"
    np.save(output_path, signed_dmap.astype(np.float32))
    
    print(f"  ✓ Saved to {output_path}")

print("\n✓ All signed distance maps created!")
print(f"Location: {data_dir}/signed_distance_maps/")

