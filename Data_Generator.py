import torch
from torch.utils.data import Dataset
import numpy as np

from src.hyperparameters import max_int 

class TSPDataset(Dataset):
    """
    Random TSP dataset

    """
    
    def load_data(self, path, seq_lengths, metric):
        data = {}
        tour_lengths = {}
        for l in seq_lengths:
            data[l] = torch.from_numpy(np.load(path + "/" + str(l) + "/data.npy", "r")) / max_int
            tour_lengths[l] = torch.from_numpy(np.reshape(np.load(path + "/" + str(l) + f"/concorde_{metric}_length.npy", "r"), (-1,1)))
        return data, tour_lengths

    def __init__(self, path, seq_lengths, metric):
        self.path = path
        self.seq_lengths = seq_lengths
        self.metric = metric
        
        self.data, self.tour_lengths = self.load_data(self.path, self.seq_lengths, self.metric)
        
        self.max_sequence_length = max(seq_lengths)
        data_tensor = []
        for d in list(self.data.values()):
            data_tensor.append(torch.permute(torch.nn.ConstantPad1d((0, self.max_sequence_length - d.shape[1]), 0)(torch.permute(d,(0,2,1))), (0,2,1)))
        
        self.data_tensor = torch.cat(data_tensor)
        self.tour_lengths_tensor = torch.cat(list(self.tour_lengths.values()))
                                               
        self.permutation = torch.randperm(self.data_tensor.shape[0])
                                               
        self.data_tensor = self.data_tensor[self.permutation]
        self.tour_lengths_tensor = self.tour_lengths_tensor[self.permutation]

    def __len__(self):
        return self.data_tensor.shape[0]

    def __getitem__(self, idx):
        return {'data': self.data_tensor[idx], 'length': self.tour_lengths_tensor[idx]}
