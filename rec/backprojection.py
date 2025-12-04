import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix

SPEED_OF_LIGHT = 3E8

def backprojection(pt_clouds : list, 
                   hists : list, 
                   voxel_grid : np.array, 
                   gates : list,
                   bin_width : float, 
                   voxel_params : list,
                   thresh: float = np.nan,
                   return_indiv: bool = False,
                   show_progress: bool = True):
        """
        Parameters:
        -----------
        pt_clouds     : List of length num_frames. Each entry contains 
                        array (num_pixels, 3)
        hists         : List of length num_frames. Each entry contains 
                        array (num_pixels, num_bins)
        voxel_grid    : Voxel locations (num_voxels, 3)
        gates         : list containing start gate, and end gate
        bin_width     : timing resolution in seconds
        voxel_params  : list containing number of x, y, and z voxels
        thresh        : threshold for backprojection
        return_indiv  : whether to return individual frame reconstructions
        show_progress : whether to show progress bar

        Returns:
        --------
        volume_filter  : filtered backprojected volume (num_x, num_y, num_z)
        """

        num_pixels, num_bins = hists[0].shape
        num_hists = len(hists)

        # === Extract voxel params === #
        num_x, num_y, num_z = voxel_params
        num_voxels = voxel_grid.shape[0]

        # === Extract other parameters === #
        if np.isnan(thresh):
            thresh = bin_width * SPEED_OF_LIGHT

        start_gate, end_gate = gates

        # === Compute backprojection by looping through pixels and time bins === #  
        volume = np.zeros((num_voxels, 1))
        indiv_reconst = []; max_voxels = []
        for k in tqdm(range(num_hists), 
                      desc="Computing backprojection", 
                      disable=not show_progress): # loop through frames
            cur_pt_cloud = pt_clouds[k]
            cur_frame = np.zeros((num_voxels, 1))
            for i, cur_pixel in enumerate(cur_pt_cloud): # loop through pixels
                if np.isnan(cur_pixel).any():
                    continue
                dists = np.linalg.norm(voxel_grid - cur_pixel.reshape(1, 3), axis=1).reshape(-1, 1)
                for j in range(start_gate, end_gate): # loop through time bins
                    cur_radius = j * bin_width * SPEED_OF_LIGHT / 2
                    mask = np.abs(dists - cur_radius) < thresh
                    cur_frame += hists[k][i, j] * mask

            volume += cur_frame

            # === Reshape and filter current frame === #
            cur_frame = cur_frame.reshape(num_x, num_y, num_z, order="C")
            # cur_frame = filter_volume(cur_frame, num_x, num_y)
            # cur_frame = np.transpose(cur_frame, [2, 1, 0])
            # cur_frame = np.flip(cur_frame, axis=(1, 2))
            indiv_reconst.append(cur_frame)
            
        volume = volume.reshape(num_x, num_y, num_z, order="C")

        # # === filtering step === #
        volume_filter = filter_volume(volume, num_x, num_y)
        # volume_filter = volume 
        # volume_filter = np.transpose(volume_filter, [2, 1, 0])
        # volume_filter = np.flip(volume_filter, axis=(1, 2))
        # if return_indiv:
        #      indiv_reconst = [indiv.reshape(num_x, num_y, num_z, order="C") for indiv in indiv_reconst]
            #  indiv_filter = [filter_volume(indiv, num_x, num_y) for indiv in indiv_reconst]


        if return_indiv:
            return volume_filter, indiv_reconst
        else:
             return volume_filter
            

def filter_volume(volume: np.array, num_x: int, num_y: int) -> np.array:
        """
        Compute Laplacian operator along z direction.

        Parameters:
        -----------
        volume : 3D volume to be filtered (num_x, num_y, num_z)
        num_x  : number of x voxels
        num_y  : number of y voxels

        Returns:
        --------
        volume : filtered volume with zero pad at edges

        """
        volume_unpadded = 2 * volume[:, :, 1:-1] - volume[:, :, :-2] - volume[:, :, 2:]
        zero_pad = np.zeros((num_x, num_y, 1))
        volume_padded = np.concatenate([zero_pad, volume_unpadded, zero_pad], axis=-1)
        return volume_padded


def lct(hist : np.array, 
        invpsf: np.array,
        mtxi: np.array,
        t_res : float,
        width : float,
        z_offset : int = 0
    ) -> np.array:
    """
    Reconstruct volume from histogram using light cone transform.

    Parameters:
    -----------
    hist     : histogram in LCT-transformed space (num_y, num_x, num_bins)
    invpsf   : FFT of parabolic point spread function (num_z, num_y, num_x)
    t_res    : time resolution of histogram
    width    : half width of the wall in meters (width and height should be same)
    z_offset : offset to clip in z direction

    Returns:
    --------
    volume : Reconstructed volume of shape (num_x, num_y, num_z)
    """

    assert hist.shape[-1] == invpsf.shape[0] // 2, f"Time dimension in hist ({hist.shape[-1]}) " + \
                                                    f"must match z/v dimension ({invpsf.shape[0]}) in PSF"
    assert hist.shape[0] == hist.shape[1], "This LCT implementation only supports square inputs"
    assert hist.shape[-1] & (hist.shape[-1] - 1) == 0, "Number of bins must be a power of 2"

    num_pixels, _, num_bins = hist.shape

    # === Extract data dimensions === #
    max_range = num_bins * SPEED_OF_LIGHT * t_res # Maximum range for histogram

    # === Permute data dimensions === #
    data = np.transpose(hist, (2, 1, 0)) # (num_bins, num_y, num_x)

    # === Step 1: Pad input histogram === #
    tdata = np.zeros((2*num_bins, 2*num_pixels, 2*num_pixels))
    tdata[0:num_bins, 0:num_pixels, 0:num_pixels] = data # (2*num_bins, 2*num_y, 2*num_x)

    # === Step 2: Convolve with inverse filter === #
    tvol = np.fft.ifftn(np.fft.fftn(tdata)*invpsf)
    tvol = tvol[0:num_bins, 0:num_pixels, 0:num_pixels] # (num_bins, num_y, num_x)

    # === Step 3: Resample depth axis and clamp results === #
    tvol_res = np.reshape(tvol, (num_bins, num_pixels**2), order='C')
    vol_res = mtxi @ tvol_res
    vol = np.reshape(vol_res, (num_bins, num_pixels, num_pixels), order='C')
    vol  = np.maximum(np.real(vol), 0)

    # === Crop and flip reconstructed volume for visualization === #
    ind = round(num_bins * 2 * width / (max_range / 2))
    # vol = vol[:,:,-1::-1]
    vol = vol[z_offset:min(ind+z_offset, vol.shape[0]), :, :]

    # === Permute dimensions === #
    vol = np.transpose(vol, (1, 2, 0)) # (num_x, num_y, num_z)

    return vol
    



def define_psf(U: int, V: int, slope: float):
    """
    Local function to compute NLOS blur kernel.

    Parameters:
    -----------
    U     : number of pixels in x and y direction
    V     : number of timing bins (or number of voxels in z)
    slope : slope of the parabolic surface

    Returns:
    --------
    psf   : 3D blur kernel of shape (2*V, 2*U, 2*U)

    """
    x = np.linspace(-1, 1, 2 * U)
    y = np.linspace(-1, 1, 2 * U)
    z = np.linspace(0, 2, 2 * V)
    grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing='ij')

    # === Define PSF === #
    psf = np.abs(((4 * slope) ** 2) * (grid_x ** 2 + grid_y ** 2) - grid_z)
    psf = psf == np.repeat(np.min(psf, axis=0)[np.newaxis, :, :], 2*V, axis=0)
    psf = psf.astype(float)

    psf /= np.sum(psf[:, U-1, U-1])
    psf /= np.linalg.norm(psf.ravel())
    psf = np.roll(psf, (0, U, U), axis=(0, 1, 2))
    return psf


def resampling_operator(num_bins):
    """
    Compute resampling and inverse resampling operator for LCT. Function adapted 
    from O'Toole et al. "Confocal NLOS Imaging based on the Light Cone Transform"

    Parameters:
    -----------
    num_bins : number of bins

    Returns:
    --------
    mtx  : sparse resampling matrix (num_bins, num_bins)
    mtxi : inverse resampling matrix (num_bins, num_bins)
    """
    #   numBins (scalar)
    # =============================================================== #
    #                              OUTPUTS  
    # =============================================================== #
    #   mtx = (newNumBins, newNumBins)
    #   mtxi = (newNumBins, newNumBins)
    # =============================================================== #
    mtx = csr_matrix(([], ([], [])), shape=(num_bins**2, num_bins))

    x = np.arange(1, num_bins**2 + 1)
    mtx[x - 1, np.ceil(np.sqrt(x)) - 1] = 1

    # mtx  = spdiags(1./sqrt(x)', 0, M**2, M**2) * mtx
    x_sqrt_inv = 1 / np.sqrt(x)
    diag_mtx = csr_matrix((x_sqrt_inv, (np.arange(num_bins**2), np.arange(num_bins**2))), shape=(num_bins**2, num_bins**2))
    mtx = diag_mtx.dot(mtx)

    mtxi = mtx.transpose()

    K = np.round(np.log(num_bins) / np.log(2))
    for k in range(int(K)):
        mtx = 0.5 * (mtx[0::2, :] + mtx[1::2, :])
        mtxi = 0.5 * (mtxi[:, 0::2] + mtxi[:, 1::2])

    return mtx, mtxi