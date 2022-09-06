from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

from src.hyperparameters import max_int 

class TSPDataset(Dataset):
    """
    Random TSP dataset

    """
    
    def load_data(self, path, seq_lengths, metric):
        data = []
        tour_lengths = []
        tours = []
        for l in seq_lengths:
            data.append(np.lib.format.open_memmap(f"{path}/{str(l)}/data.npy", dtype='float32', mode='r'))
            if not self.omit_tour_length:
                tour_lengths.append(np.lib.format.open_memmap(f"{path}/{str(l)}/concorde_{metric}_length.npy", dtype='float32', mode='r'))
            if self.data_source == "uniform":
                tours.append(np.lib.format.open_memmap(f"{path}/{str(l)}/concorde_{metric}_tour.npy", dtype='int32', mode='r'))
            elif self.data_source == "ptrnet_data":
                tours.append(np.lib.format.open_memmap(f"{path}/{str(l)}/tour.npy", dtype='int32', mode='r'))
        return data, tour_lengths, tours

    def __init__(self, path, seq_lengths, metric, samples_per_length, data_source="uniform", omit_tour_length=False, data_in_ints=True):
        self.path = path
        self.seq_lengths = seq_lengths
        self.metric = metric
        self.samples_per_length = samples_per_length
        self.data_source = data_source
        self.omit_tour_length = omit_tour_length
        self.data_in_ints = data_in_ints
        
        self.data, self.tour_lengths, self.tours = self.load_data(self.path, self.seq_lengths, self.metric)
        
        self.max_sequence_length = max(seq_lengths)

    def __len__(self):
        return len(self.data)*self.samples_per_length

    def __getitem__(self, idx): 
        d = defaultdict(list)
        if isinstance(idx, int):
            idx = [idx]
        for x in list(idx):
            d[x // self.samples_per_length].append(x % self.samples_per_length)
        
        max_sequence_length = self.seq_lengths[max(d.keys())]
        
        data_tensor = []
        seq_lengths_tensor = []
        tours_tensor = []
        tours_lengths_tensor = []
        tour_ohe_tensor = []
        for l in d:
            if self.data_in_ints:
                data = torch.from_numpy(self.data[l][d[l]]) / max_int
            else:
                data = torch.from_numpy(self.data[l][d[l]]).float()
            if not self.omit_tour_length:
                if self.data_in_ints:
                    tour_lengths = torch.from_numpy(self.tour_lengths[l][d[l]]) / max_int
                else:
                    tour_lengths = torch.from_numpy(self.tour_lengths[l][d[l]])
            tours = torch.from_numpy(self.tours[l][d[l]])
            data_tensor.append(torch.permute(torch.nn.ConstantPad2d((0, max_sequence_length - self.seq_lengths[l], 0, 0), 0.0)(torch.permute(data,(0,2,1))), (0,2,1)))
            tours_tensor.append(torch.nn.ConstantPad1d((0, max_sequence_length - self.seq_lengths[l]), 0.0)(tours))
            seq_lengths_tensor.append(torch.IntTensor([int(data.shape[1])]*data.shape[0]))
            if not self.omit_tour_length:
                tours_lengths_tensor.append(tour_lengths)
            tour_ohe_tensor.append(torch.nn.ConstantPad2d((0, 0, 0, max_sequence_length - self.seq_lengths[l]), 0.0)(F.one_hot(tours.long(), num_classes=max_sequence_length).type(torch.float32)))
        return {
            'data': torch.cat(data_tensor) if self.data_source == "uniform" else torch.cat(data_tensor).float(), 
            'sequence_length': torch.cat(seq_lengths_tensor), 
            'tour': torch.cat(tours_tensor).long(),
            'tour_length': torch.cat(tours_lengths_tensor) if not self.omit_tour_length else torch.empty(1), 
            'tour_ohe': torch.cat(tour_ohe_tensor)
        }
