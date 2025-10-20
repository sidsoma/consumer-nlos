from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import List
import torch

class Histograms(Dataset):
    """
    Dataset class for multiple frames of space-time measurements. 

    Parameters:
    -----------
    hists : list of space-time array measurements of size (num_y, num_x, numBins)

    """
    def __init__(self, 
                 hists: List[torch.Tensor], 
                 pt_clouds : List[torch.Tensor],
            ):
        assert len(hists) == len(pt_clouds) 

        self.hists = hists
        self.pt_clouds = pt_clouds
        self.num_frames = len(hists)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, t):
        return t, self.hists[t], self.pt_clouds[t]

def create_data_loader(hists : List[torch.Tensor], 
                       pt_clouds : List,
                       batch_size : int,
                       shuffle : bool=False):
    """
    Creates data loader that can batch together frames of space-time measurements.

    Parameters:
    -----------
    hists       : list of space-time array measurements of size (num_y, num_x, numBins)
    pt_clouds   : list of point clouds containing the locations of sampled points (num_y, num_x, 3)
    batch_size  : batch size in each iteration
    shuffle     : toggles whether batches should be indexed sequentially or randomly

    Returns:
    --------
    data_loader   : torch DataLoader object

    """
    meas_dataset = Histograms(hists=hists, pt_clouds=pt_clouds)
    data_loader = DataLoader(meas_dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader
