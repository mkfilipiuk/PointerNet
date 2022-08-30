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
            data.append(np.lib.format.open_memmap(f"{path}/{str(l)}/data.npy", dtype='int32', mode='r'))
            # torch.from_numpy(np.load(, "r")) / max_int
            tour_lengths.append(np.lib.format.open_memmap(f"{path}/{str(l)}/concorde_{metric}_length.npy", dtype='float32', mode='r'))
            # torch.from_numpy(np.reshape(np.load(path + "/" + str(l) + f"/concorde_{metric}_length.npy", "r"), (-1,1)))
            tours.append(np.lib.format.open_memmap(f"{path}/{str(l)}/concorde_{metric}_tour.npy", dtype='int32', mode='r'))
            # torch.from_numpy(np.load(path + "/" + str(l) + f"/concorde_{metric}_tour.npy", "r"))
        return data, tour_lengths, tours

    def __init__(self, path, seq_lengths, metric, samples_per_length):
        self.path = path
        self.seq_lengths = seq_lengths
        self.metric = metric
        self.samples_per_length = samples_per_length
        
        self.data, self.tour_lengths, self.tours = self.load_data(self.path, self.seq_lengths, self.metric)
        
        self.max_sequence_length = max(seq_lengths)
        
#         data_tensor = []
#         seq_lengths_tensor = []
#         for d in list(self.data.values()):
#             data_tensor.append(torch.permute(torch.nn.ConstantPad1d((0, self.max_sequence_length - d.shape[1]), 0)(torch.permute(d,(0,2,1))), (0,2,1)))
#             seq_lengths_tensor.append(torch.Tensor([d.shape[1]]*d.shape[0]))
#         tours_tensor = []
#         for d in list(self.tours.values()):
#             tours_tensor.append((F.one_hot(d.long(), num_classes=self.max_sequence_length).type(torch.float32)))
        
#         self.data_tensor = torch.cat(data_tensor)
#         self.seq_lengths_tensor = torch.cat(seq_lengths_tensor)
#         self.tour_lengths_tensor = torch.cat(list(self.tour_lengths.values()))
#         self.tours_tensor = torch.cat(tours_tensor)

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
            data = torch.from_numpy(self.data[l][d[l]]) / max_int
            tour_lengths = torch.from_numpy(self.tour_lengths[l][d[l]]) / max_int
            tours = torch.from_numpy(self.tours[l][d[l]])
            data_tensor.append(torch.permute(torch.nn.ConstantPad2d((0, max_sequence_length - self.seq_lengths[l], 0, 0), 0.0)(torch.permute(data,(0,2,1))), (0,2,1)))
            tours_tensor.append(torch.nn.ConstantPad1d((0, max_sequence_length - self.seq_lengths[l]), 0.0)(tours))
            seq_lengths_tensor.append(torch.IntTensor([int(data.shape[1])]*data.shape[0]))
            tours_lengths_tensor.append(tour_lengths)
            tour_ohe_tensor.append(torch.nn.ConstantPad2d((0, 0, 0, max_sequence_length - self.seq_lengths[l]), 0.0)(F.one_hot(tours.long(), num_classes=max_sequence_length).type(torch.float32)))
        return {
            'data': torch.cat(data_tensor), 
            'sequence_length': torch.cat(seq_lengths_tensor), 
            'tour': torch.cat(tours_tensor).long(),
            'tour_length': torch.cat(tours_lengths_tensor), 
            'tour_ohe': torch.cat(tour_ohe_tensor)
        }
