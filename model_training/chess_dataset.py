import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.nn import functional as F
from sklearn.decomposition import PCA

class ChessTensorDataset(Dataset):
    def __init__(self, metadata_csv, tensor_dir):
        self.df = pd.read_csv(metadata_csv)
        self.tensor_dir = tensor_dir
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        try:
            tensor_file = self.df.iloc[idx]['TensorFile']
            tensor_path = os.path.join(self.tensor_dir, tensor_file)
            
            # If the path doesn't exist, try alternative extension
            if not os.path.exists(tensor_path):
                base, ext = os.path.splitext(tensor_path)
                if ext == '.npy':
                    alt_path = base + '.npz'
                else:
                    alt_path = base + '.npy'
                
                if os.path.exists(alt_path):
                    tensor_path = alt_path
                else:
                    raise FileNotFoundError(f"Neither {tensor_path} nor {alt_path} exists")
            
            # Load the tensor
            if tensor_path.endswith('.npz'):
                loaded = np.load(tensor_path)
                x = loaded['arr']
            else:
                x = np.load(tensor_path)
                
            x = x.astype(np.float32)
            if len(x.shape) == 3 and x.shape[2] > x.shape[0]:
                x = np.transpose(x, (2, 0, 1))
            move_index = int(self.df.iloc[idx]['MoveIndex'])
            return torch.tensor(x), torch.tensor(move_index)
        except Exception as e:
            print(f"Error loading index {idx}: {e}")
            # Return a placeholder instead of None
            placeholder = np.zeros((60, 8, 8), dtype=np.float32)
            return torch.tensor(placeholder), torch.tensor(0)
