import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import loadmat
from typing import List 
from tqdm import tqdm

def transform_measurement(data : np.array, diffuse : bool):
    """
    Parameters:
    -----------
    data   : histogram (n_y, n_x, num_bins)
    diffuse : toggle for diffuse or retroreflective

    Returns:
    --------
    gt_canon_lct : transformed histogram (n_y, n_x, num_bins)
    """
    n_y, n_x, T = data.shape
    gt_canon_trans = np.transpose(data, (2, 1, 0))

    # === attenuate intensity along t axis === 
    grid_z = np.tile(np.linspace(0, 1, T)[:, np.newaxis, np.newaxis], (1, n_y, n_x))
    if diffuse:
        gt_canon_atten = gt_canon_trans * (grid_z ** 4)
    else:
        gt_canon_atten = gt_canon_trans * (grid_z ** 2)

    # === resample temporal axis === #
    mtx, mtxi = resampling_operator(T)
    canon_res = np.reshape(gt_canon_atten, (T, n_x*n_y), order='C')
    tcanon = mtx @ canon_res
    gt_canon_lct = np.reshape(tcanon, (T, n_y, n_x), order='C')
    gt_canon_lct = np.transpose(gt_canon_lct, (2, 1, 0))
    
    return gt_canon_lct

def transform_measurements(hists : List[np.array], diffuse : bool):
    """
    Parameters:
    -----------
    data   : histogram (n_y, n_x, num_bins)
    diffuse : toggle for diffuse or retroreflective

    Returns:
    --------
    gt_canon_lct : transformed histogram (n_y, n_x, num_bins)
    """
    n_y, n_x, T = hists[0].shape

    # === Compute resampling operator === #
    mtx, mtxi = resampling_operator(T)

    # === Loop through each histogram and transform === #
    hists_lct = []
    for data in tqdm(hists, desc="Computing LCT"):
        gt_canon_trans = np.transpose(data, (2, 1, 0))

        # === attenuate intensity along t axis === 
        grid_z = np.tile(np.linspace(0, 1, T)[:, np.newaxis, np.newaxis], (1, n_y, n_x))
        if diffuse:
            gt_canon_atten = gt_canon_trans * (grid_z ** 4)
        else:
            gt_canon_atten = gt_canon_trans * (grid_z ** 2)

        # === resample temporal axis === #
        canon_res = np.reshape(gt_canon_atten, (T, n_x*n_y), order='C')
        tcanon = mtx @ canon_res
        gt_canon_lct = np.reshape(tcanon, (T, n_y, n_x), order='C')
        gt_canon_lct = np.transpose(gt_canon_lct, (2, 1, 0))

        hists_lct.append(gt_canon_lct)
    
    return hists_lct


def resampling_operator(numBins):
    """
    Function adapted from O'Toole et al. "Confocal NLOS Imaging using Light Cone Transform".
    
    Parameters:
    -----------
    numBins : number of bins in original histogram 
    
    Returns:
    --------
    mtx  : matrix mapping from native -> LCT space (newNumBins, newNumBins)
    mtxi : inverse mapping from LCT space -> native (newNumBins, newNumBins)

    """
    mtx = csr_matrix(([], ([], [])), shape=(numBins**2, numBins))

    x = np.arange(1, numBins**2 + 1)
    mtx[x - 1, np.ceil(np.sqrt(x)) - 1] = 1

    # mtx  = spdiags(1./sqrt(x)', 0, M**2, M**2) * mtx
    x_sqrt_inv = 1 / np.sqrt(x)
    diag_mtx = csr_matrix((x_sqrt_inv, (np.arange(numBins**2), np.arange(numBins**2))), shape=(numBins**2, numBins**2))
    mtx = diag_mtx.dot(mtx)

    mtxi = mtx.transpose()

    K = np.round(np.log(numBins) / np.log(2))
    for k in range(int(K)):
        mtx = 0.5 * (mtx[0::2, :] + mtx[1::2, :])
        mtxi = 0.5 * (mtxi[:, 0::2] + mtxi[:, 1::2])

    return mtx, mtxi
