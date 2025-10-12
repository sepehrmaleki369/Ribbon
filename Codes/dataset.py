import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import numpy as np
from .augmentations import crop
from .utils import load_graph_txt, to_torch
import networkx as nx

class DriveDataset(Dataset):
    
    def __init__(self, train=True, cropSize=(584, 565), th=15, noise=False):
        
        # DRIVE training image IDs: 21-40 (20 images)
        # Split: 21-36 for train (16 images), 37-40 for val (4 images)
        train_ids = list(range(21, 37))  # 21 to 36
        val_ids = list(range(37, 41))    # 37 to 40
        
        image_path = {
            "train": ["/content/Ribbon/Codes/drive/training/images_npy/{}_training.npy".format(i) for i in train_ids],
            "val":  ["/content/Ribbon/Codes/drive/training/images_npy/{}_training.npy".format(i) for i in val_ids]
        }
        
        label_path = {
            # Signed distance maps (negative inside vessels, positive outside)
            "train": ["/content/Ribbon/Codes/drive/training/signed_distance_maps/{}_training_signed_dmap.npy".format(i) for i in train_ids],
            "val":  ["/content/Ribbon/Codes/drive/training/signed_distance_maps/{}_training_signed_dmap.npy".format(i) for i in val_ids]
        }
        
        graph_path = {
            # Oversampled graphs for Snake loss (better for snake optimization)
            "train": ["/content/Ribbon/Codes/drive/training/graphs_oversampled/{}_oversampled_spacing5.npy".format(i) for i in train_ids],
            "val":  ["/content/Ribbon/Codes/drive/training/graphs_oversampled/{}_oversampled_spacing5.npy".format(i) for i in val_ids]
        }
        
        self.images = image_path["train"] if train else image_path["val"]
        self.labels = label_path["train"] if train else label_path["val"]
        self.graphs = graph_path["train"] if train else graph_path["val"]
            
        self.train = train
        self.cropSize = cropSize
        self.th = th
    
    def load_graph_npy(self, filename):
        """Load graph from numpy pickle file"""
        data = np.load(filename, allow_pickle=True).item()
        
        # If it's already a NetworkX graph, return it
        if isinstance(data, nx.Graph):
            return data
        
        # If it's a dictionary, convert to NetworkX graph
        if isinstance(data, dict):
            G = nx.Graph()
            
            # Check if it has 'nodes' and 'edges' keys
            if 'nodes' in data and 'edges' in data:
                for i, pos in enumerate(data['nodes']):
                    G.add_node(i, pos=np.array(pos))
                for edge in data['edges']:
                    G.add_edge(edge[0], edge[1])
            else:
                raise ValueError(f"Unknown graph format in {filename}: {data.keys()}")
            
            return G
        
        raise ValueError(f"Cannot load graph from {filename}, unexpected type: {type(data)}")
        
    def __getitem__(self, index):
        image = np.load(self.images[index])
        
        # Convert RGB to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        
        label = np.load(self.labels[index])
        
        # Load graph - check if it's .npy or .graph format
        if self.graphs[index].endswith('.npy'):
            graph = self.load_graph_npy(self.graphs[index])
        else:
            graph = load_graph_txt(self.graphs[index])
        
        for n in graph.nodes:
            graph.nodes[n]["pos"] = graph.nodes[n]["pos"][-1::-1]
        
        # Convert to float32
        image = image.astype(np.float32)
        label = label.astype(np.float32)
        
        original_image_shape = image.shape
        slices = None
        
        # Only crop if image is larger than crop size
        if self.train and image.shape[0] > self.cropSize[0] and image.shape[1] > self.cropSize[1]:
            image, label, slices = crop([image, label], self.cropSize)
            
        label[label>self.th] = self.th
        
        if self.train:
            return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), graph, slices, original_image_shape
        
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), original_image_shape

    def __len__(self):
        return len(self.images)


def collate_fn(data):
    transposed_data = list(zip(*data))
    images = torch.stack(transposed_data[0], 0)[:,None]  # Add channel dimension for grayscale
    labels = torch.stack(transposed_data[1], 0)[:,None]

    graphs = None
    slices = None
    original_shapes = None
    is_train = len(transposed_data) > 3

    if is_train:
        graphs = transposed_data[2]
        slices = transposed_data[3]
        original_shapes = transposed_data[4]
        return images, labels, graphs, slices, original_shapes
    else:
        original_shapes = transposed_data[2]
        return images, labels, original_shapes
