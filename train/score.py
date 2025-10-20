import torch.nn.functional as F
import torch
import kornia
import cv2

def ssim(test_images: torch.Tensor, ref_img: torch.Tensor) -> torch.Tensor:
    """
    Compute SSIM between GT measurement and predicted measurement.

    Parameters:
    -----------
    test_images : rendered images from particles (batch_size, n_y, n_x, num_bins)
    ref_img      : ground truth image measurement (1, n_y, n_x, num_bins)

    Returns:
    --------
    score   : structural similarity index (batch_size, )

    """
    batch_size, n_y, n_x, num_bins = test_images.shape
    num_pixels = n_y * n_x

    # === Reshape images === #
    test_images = test_images.reshape(batch_size, 1, num_pixels, num_bins)
    ref_img = ref_img.reshape(1, 1, num_pixels, num_bins)

    # === Broadcast ref_img to match test_images === #
    ref_img = ref_img.repeat(batch_size, 1, 1, 1)

    # === Compute SSIM === #
    ssim_loss = kornia.losses.ssim_loss(test_images, ref_img, window_size=11, reduction='none')
    ssim_score = 1 - 2*ssim_loss
    score = ssim_score.mean(dim=(1, 2, 3))

    return score


def canny_edge_correlation(input: torch.Tensor, 
                           pred: torch.Tensor
                    ) -> torch.Tensor:
    """
    Compare edges detected by Canny edge detector. 

    Parameters:
    -----------
    input   : ground truth image (1, n_y, n_x, num_bins)
    pred    : predicted image for each particle (batch_size, n_y, n_x, num_bins)

    Returns:   
    --------
    score   : correlation between edges (batch_size, )
    """
    batch_size, n_y, n_x, num_bins = pred.shape

    # === Reshape 3D tensor to have channel dimension === #
    input = input.unsqueeze(1)
    input = input.reshape(1, 1, n_y*n_x, num_bins)

    pred = pred.unsqueeze(1)
    pred = pred.reshape(batch_size, 1, n_y*n_x, num_bins)


    # === Normalize images to range [0, 1] === #
    input = (input - input.min()) / (input.max() - input.min())

    dims = tuple(range(1, pred.dim()))

    pred = (pred - torch.amin(pred, dim=dims, keepdim=True)) / \
            (torch.amax(pred, dim=dims, keepdim=True) - torch.amin(pred, dim=dims, keepdim=True))

    # === Compute edges === #
    input_edges, _ = kornia.filters.Canny()(input)
    pred_edges, _ = kornia.filters.Canny()(pred)

    input_edges /= torch.linalg.vector_norm(input_edges, dim=dims, keepdim=True)
    pred_edges /= torch.linalg.vector_norm(pred_edges, dim=dims, keepdim=True)

    # === Compute correlation === #
    score = torch.sum(input_edges * pred_edges, dim=dims)
    
    return score


def mean_diff(input: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Computes the difference in peak location b/w the two histograms.

    Parameters:
    -----------
    input   : ground truth image (1, ..., num_bins)
    pred    : predicted image for each particle (batch_size, ..., num_bins)

    Returns:
    --------
    mode_diff   : difference in peak locations (batch_size, )

    """
    num_bins = input.shape[-1]
    x = torch.arange(num_bins).reshape(1, 1, 1, num_bins).to(pred.device)

    # === Normalize temporal dimension to have integral 1 === #
    input /= torch.linalg.norm(input + 1e-8, dim=-1, keepdim=True)
    pred /= torch.linalg.norm(pred + 1e-8, dim=-1, keepdim=True)

    # === Compute expected value (mean) along time axis === #
    input_mean = torch.sum(input * x, dim=-1)
    pred_mean = torch.sum(pred * x, dim=-1)

    # === Compute mean difference === #
    mean_difference = torch.nanmean(torch.abs(input_mean - pred_mean), dim=tuple(range(1, input_mean.dim()))) 
    mean_score = (num_bins - mean_difference) / num_bins

    return mean_score# + var_difference 


def gradient_correlation(input: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
    -----------
    input   : ground truth image (1, n_y, n_x num_bins)
    pred    : predicted image for each particle (batch_size, n_y, n_x, num_bins)

    Returns:
    --------
    corr   : average dot product of gradients along space and time axes (batch_size, )

    """

    # === Compute gradients === #
    input_grad_y = input[:, 1:, :, :] - input[:, :-1, :, :]
    input_grad_x = input[:, :, 1:, :] - input[:, :, :-1, :]
    input_grad_t = input[:, :, :, 1:] - input[:, :, :, :-1]

    pred_grad_y = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    pred_grad_x = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    pred_grad_t = pred[:, :, :, 1:] - pred[:, :, :, :-1]

    # # === Normalize gradients === #
    dims = tuple(range(1, pred.dim()))
    pred_grad_x /= torch.linalg.vector_norm(pred_grad_x, dim=dims, keepdim=True) + 1e-7
    pred_grad_y /= torch.linalg.vector_norm(pred_grad_y, dim=dims, keepdim=True) + 1e-7
    pred_grad_t /= torch.linalg.vector_norm(pred_grad_t, dim=dims, keepdim=True) + 1e-7

    # === Compute dot product of gradients === #
    dot_x = torch.sum(input_grad_x * pred_grad_x, dim=dims)
    dot_y = torch.sum(input_grad_y * pred_grad_y, dim=dims)
    dot_t = torch.sum(input_grad_t * pred_grad_t, dim=dims)
    
    # === Compute average === #
    corr = (1/3) * (dot_x + dot_y + dot_t)

    return corr


def mse_score(input: torch.Tensor, pred: torch.Tensor, k: float = None) -> torch.Tensor:
    """
    Parameters:
    -----------
    input   : ground truth image (1, n_y, n_x, num_bins)
    pred    : predicted image for each particle (batch_size, n_y, n_x, num_bins)

    Returns:
    --------
    loss   : mean squared error loss (batch_size, )

    """
    if k is None:
        # k = 5 / input.shape[-1]
        k = 1 / input.shape[-1]
    loss = torch.mean((input - pred)**2, dim=tuple(range(1, input.dim()))) # (batch_size, )
    # print(torch.min(loss), torch.max(loss))
    # score = torch.exp(-k * loss)
    # score = -loss
    scores = torch.max(loss[~torch.isnan(loss)])-loss

    return scores


def dot_product_score(input: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
    -----------
    input   : ground truth image (1, n_y, n_x, num_bins)
    pred    : predicted image for each particle (batch_size, n_y, n_x, num_bins)

    Returns:
    --------
    score   : dot product of input and pred (batch_size, )
    """

    # === Normalize images === #
    # input /= torch.linalg.vector_norm(input, dim=tuple(range(1, input.dim())), keepdim=True)
    # pred /= torch.linalg.vector_norm(pred, dim=tuple(range(1, pred.dim())), keepdim=True)
    # input /= torch.sum(input, dim=tuple(range(1, input.dim())), keepdim=True)
    # pred /= torch.sum(pred, dim=tuple(range(1, pred.dim())), keepdim=True)

    # === Compute dot product === #
    score = torch.sum(input * pred, dim=tuple(range(1, input.dim())))

    return score

def normalized_dot_product_score(input: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
    -----------
    input   : ground truth image (1, n_y, n_x, num_bins)
    pred    : predicted image for each particle (batch_size, n_y, n_x, num_bins)

    Returns:
    --------
    score   : dot product of input and pred (batch_size, )
    """

    # === Normalize images === #
    input /= torch.linalg.vector_norm(input, dim=tuple(range(1, input.dim())), keepdim=True)
    pred /= torch.linalg.vector_norm(pred, dim=tuple(range(1, pred.dim())), keepdim=True)

    # === Compute dot product === #
    score = torch.sum(input * pred, dim=tuple(range(1, input.dim())))

    return score

def filtered_dot_product_score(input: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
    -----------
    input   : ground truth image (1, n_y, n_x, num_bins)
    pred    : predicted image for each particle (batch_size, n_y, n_x, num_bins)

    Returns:
    --------
    score   : dot product of input and pred (batch_size, )
    """

    # === Normalize images === #
    input /= torch.linalg.vector_norm(input, dim=tuple(range(1, input.dim())), keepdim=True)
    pred /= torch.linalg.vector_norm(pred, dim=tuple(range(1, pred.dim())), keepdim=True)

    # === Compute dot product === #
    score = torch.sum(input * pred, dim=tuple(range(1, input.dim())))

    return score


def tof_diff(input : torch.Tensor, pred : torch.Tensor) -> torch.Tensor:
    """
    Parameters:
    -----------
    input   : ground truth image (1, n_y, n_x, num_bins)
    pred    : predicted image for each particle (batch_size, n_y, n_x, num_bins)

    Returns:
    --------
    score   : difference in peak location along time axis (batch_size, )

    """
    # === Compute tof === #
    input_tof = torch.argmax(input, dim=-1) # (1, n_y, n_x)
    pred_tof = torch.argmax(pred, dim=-1) # (batch_size, n_y, n_x)

    # === Compute difference === #
    diff = torch.abs(input_tof - pred_tof).float().mean(dim=tuple(range(1, input_tof.dim()))) 
    score = input.shape[-1] - diff

    return score

def weighted_score(input: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
    -----------
    input   : ground truth image (1, n_y, n_x, num_bins)
    pred    : predicted image for each particle (batch_size, n_y, n_x, num_bins)

    Returns:
    --------
    score   : weighted score (batch_size, )

    """
    tof_score = mean_diff(input, pred)
    corr_score = dot_product_score(input, pred)
    score = 0.5 * (tof_score + corr_score)

    return score