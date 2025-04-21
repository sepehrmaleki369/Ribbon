import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import numpy as np
from .augmentations import crop
from .utils import load_graph_txt, to_torch
from PIL import Image
import os

class SynthDataset(Dataset):
    
    def __init__(self, train=True, cropSize=(96,96,96), th=15, noise=False):
        
        image_path = {
            "train": ["/content/drive/MyDrive/synth_data/images/data_{}.npy".format(i) for i in range(25)],
            "val":  ["/content/drive/MyDrive/synth_data/images/data_{}.npy".format(i) for i in range(25,30)]
        }
        label_path = {
            # Using noise_2_dist_labels instead of dist_labels
            "train": ["/content/drive/MyDrive/synth_data/noise_2_dist_labels/data_{}.npy".format(i) for i in range(25)],
            "noise": ["/content/drive/MyDrive/synth_data/noise_2_dist_labels/data_{}.npy".format(i) for i in range(25)],
            "val":  ["/content/drive/MyDrive/synth_data/noise_2_dist_labels/data_{}.npy".format(i) for i in range(25,30)]
        }
        graph_path = {
            "train": ["/content/drive/MyDrive/synth_data/graphs/data_{}.graph".format(i) for i in range(25)],
            "noise": ["/content/drive/MyDrive/synth_data/noise_2_graphs/data_{}.graph".format(i) for i in range(25)],
            "val":  ["/content/drive/MyDrive/synth_data/graphs/data_{}.graph".format(i) for i in range(25,30)]
        }
        
        self.images = image_path["train"] if train else image_path["val"]
        self.labels = label_path["train"] if train and not noise else label_path["noise"] if train and noise else label_path["val"]
        self.graphs = graph_path["train"] if train and not noise else graph_path["noise"] if train and noise else graph_path["val"]
            
        self.train = train
        self.cropSize = cropSize 
        self.th = th
        
    def __getitem__(self, index):

        image = np.load(self.images[index])
        label = np.load(self.labels[index])
        graph = load_graph_txt(self.graphs[index])
        
        for n in graph.nodes:
            graph.nodes[n]["pos"] = graph.nodes[n]["pos"][-1::-1]
            
        slices = None
        
        if self.train:
            image, label, slices = crop([image, label], self.cropSize)
            
        label[label>self.th] = self.th
        
        if self.train:
            return torch.tensor(image), torch.tensor(label), graph, slices
        
        return torch.tensor(image), torch.tensor(label)

    def __len__(self):
        return len(self.images)


class DRIVEDataset(Dataset):
    
    def __init__(self, base_dir, train=True, cropSize=(512, 512), th=15, transform=None):
        """
        Args:
            base_dir (str): Path to DRIVE dataset root directory
            train (bool): Whether to use training or test set
            cropSize (tuple): Size for cropping images
            th (float): Threshold value
            transform (callable): Optional transform to apply to the images
        """
        subset = 'training' if train else 'test'
        
        """ image_dir = os.path.join(base_dir, subset, 'images')
        mask_dir = os.path.join(base_dir, subset, 'mask')
        gt_dir = os.path.join(base_dir, subset, '1st_manual') """
        self.image_path = {
            "train": ["/drive/training/images/{i}_training.npy".format(i) for i in range(21,36)],
            "val":  ["/drive/training/images/{i}_training.npy".format(i) for i in range(36,41)]
        }
        self.label_path = {
            # Using noise_2_dist_labels instead of dist_labels
            "train": ["/drive/training/inverted_labels/{i}_manual1.npy".format(i) for i in range(21,36)],
            "val":  ["/drive/training/inverted_labels/{i}_manual1.npy".format(i) for i in range(36,41)]
        }
        self.graph_path = {
            "train": ["/drive/training/graphs/{i}_manual1.npy.graph".format(i) for i in range(21,36)],
            "val":  ["/drive/training/graphs/{i}_manual1.npy.graph".format(i) for i in range(36,41)]
        }
        self.masks = {
            "train": ["/drive/training/mask/{i}_training_mask.gif".format(i) for i in range(21,36)],
            "val":  ["/drive/training/mask/{i}_training_mask.gif".format(i) for i in range(36,41)]
        }
        
        """ self.images = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if f.endswith('.tif')]
        self.masks = [os.path.join(mask_dir, f) for f in sorted(os.listdir(mask_dir)) if f.endswith('.gif')]
        self.gt = [os.path.join(gt_dir, f) for f in sorted(os.listdir(gt_dir)) if f.endswith('.gif')] """
        
        self.train = train
        self.cropSize = cropSize
        self.th = th
        self.transform = transform
        
    def __getitem__(self, index):
        # Load image (tif)
        image = np.array(Image.open(self.images[index]), dtype=np.float32) / 255.0
        
        # Load mask (gif)
        mask = np.array(Image.open(self.masks[index]), dtype=np.float32) / 255.0
        
        # Load ground truth (gif)
        label = np.array(Image.open(self.gt[index]), dtype=np.float32) / 255.0
        
        # Apply mask to focus only on the retinal area
        image = image * mask[:,:,None] if image.ndim == 3 else image * mask
        
        # Convert RGB to grayscale if necessary
        if image.ndim == 3:
            image = np.mean(image, axis=2)
        
        # Create distance map from ground truth if needed
        # You may need to compute a distance transform from the binary vessels
        # For example: from scipy.ndimage import distance_transform_edt
        # distance_map = distance_transform_edt(1 - label)
        # label = distance_map
        # label[label > self.th] = self.th
        
        slices = None
        graph = None
        
        if self.train and self.cropSize is not None:
            # Implement 2D cropping
            image, label, slices = crop([image, label], self.cropSize)
            
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
            
        # Generate graph representation for snake algorithm if needed
        if self.train and hasattr(self, 'generate_graph'):
            graph = self.generate_graph(label, slices)
        
        if self.train and graph is not None:
            return torch.tensor(image), torch.tensor(label), graph, slices
        
        return torch.tensor(image), torch.tensor(label)

    def __len__(self):
        return len(self.images)


def collate_fn(data):
    transposed_data = list(zip(*data))
    images = torch.stack(transposed_data[0], 0)[:,None]
    labels = torch.stack(transposed_data[1], 0)[:,None]
    graphs = transposed_data[2]
    slices = transposed_data[3]
    
    return images, labels, graphs, slices

def drive_collate_fn(data):
    transposed_data = list(zip(*data))
    images = torch.stack(transposed_data[0], 0)[:,None]
    labels = torch.stack(transposed_data[1], 0)[:,None]
    
    if len(transposed_data) > 2:
        graphs = transposed_data[2]
        slices = transposed_data[3]
        return images, labels, graphs, slices
    
    return images, labels