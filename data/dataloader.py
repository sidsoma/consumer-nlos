import numpy as np
from glob import glob
import os
from typing import Tuple, List
from tqdm import tqdm 
from data.process import transform_measurements

def load_data(data_dir : str) -> Tuple[List[np.array], List[np.array]]:
    """
    Load data captured with ST VL853L8 device from directory. This dataloader
    assumes that the data is already processed, where the 1-bounce light is
    cropped and the histogram is normalized so that t=0 corresponds to when the 
    light bounces from the wall to the hidden scene. The raw data is in native space
    (not in light-cone transform space).

    Parameters:
    -----------
    data_dir : directory containing data

    Returns:
    --------
    hists : histograms (list of length num_frames, each entry is (n_y, n_x, num_bins))
    pt_clouds : point clouds (list of length num_frames, each entry is (n_y, n_x, num_bins))

    """

    filenames = sorted(glob(os.path.join(data_dir, '*.npz')))

    particles = []; hists = []; pt_clouds = []
    for filename in filenames:
        data = np.load(filename)
        hists.append(data['hists'].reshape(4, 4, -1)) # (num_pixels, num_bins)
        pt_clouds.append(data['pt_cloud'].reshape(4, 4, -1)) # (num_pixels, 3)

    return hists, pt_clouds

def compute_lct(hists_crop: List[np.array], 
                isDiffuse=False,
                num_lct_bins=128) -> List[np.array]:
    """
    Compute LCT histograms from raw histograms.

    Parameters:
    -----------
    hists : histograms with 1b peak removed (list of length num_frames, each entry is (n_y, n_x, num_bins))

    Returns:
    --------
    hists_lct : LCT histograms (list of length num_frames, each entry is (n_y, n_x, num_bins))
    """
    n_y, n_x = 4, 4

    hists_pad = []
    num_padded_bins = 128
    for meas in tqdm(hists_crop, desc="Padding zero to hists"):
        # === zero pad temporal dimension === #
        meas = np.concatenate([meas.reshape(n_y, n_x, -1), 
                            np.zeros((n_y, n_x, num_padded_bins-meas.shape[-1]))], 
                            axis=-1)

        # === Save measurement === #
        hists_pad.append(meas) # (n_y, n_x, newNumBins)

    
    # === Transform measurement === #
    hists_lct = transform_measurements(hists_pad, diffuse=isDiffuse) 

    hists_lct = [hist[..., :num_lct_bins] for hist in hists_lct]

    return hists_lct