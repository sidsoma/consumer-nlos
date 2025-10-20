import torch.nn as nn
from train.canon import FastSumOfParabolas
from train.motion import Particle
from train.score import mean_diff, gradient_correlation, \
    ssim, mse_score, dot_product_score, filtered_dot_product_score, canny_edge_correlation
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from typing import List
import importlib
import numpy as np


class ParticleModel(nn.Module):
    """
    PyTorch model for particle filtering using a known canonical measurement.

    Parameters:
    -----------
    configs     : 
    canon_init  : initialized canonical measurement 
    delta_init  : initialized object motion 

    """
    def __init__(self, 
                 configs: dict, 
                 canons: List[FastSumOfParabolas], 
                 particles: Particle,
                 gt_traj : List[np.array] = None):
        super().__init__()

        # === Define canonical measurement === #   
        self.canons = canons

        # === Define particles === #
        self.particles = particles

        # === Camera or object tracking === #
        self.is_cam_tracking = configs['cam_tracking']

        # === Define score function === #
        module = importlib.import_module('train.score')
        self.score_function = getattr(module, configs['particle']['score_fn'])

        # === Define particle boundaries === #
        self.vol_size = configs['particle']['vol_size']
        self.vol_center = configs['particle']['vol_center']
        x_center, y_center, z_center = self.vol_center

        self.x_min = x_center - self.vol_size/2; self.x_max = x_center + self.vol_size/2
        self.y_min = y_center - self.vol_size/2; self.y_max = y_center + self.vol_size/2
        self.z_min = z_center - self.vol_size/2; self.z_max = z_center + self.vol_size/2

        # === Define volume for KDE === #
        self.sigma = 0.05 # bandwidth for KDE
        num_x = 30; num_y = 30; num_z = 30
        x_vals = torch.linspace(self.x_min, self.x_max, num_x)
        y_vals = torch.linspace(self.y_min, self.y_max, num_y)
        z_vals = torch.linspace(self.z_min, self.z_max, num_z)

        x_grid, y_grid, z_grid = torch.meshgrid(x_vals, y_vals, z_vals)
        self.volume = torch.stack([x_grid, y_grid, z_grid], dim=-1).to(self.particles.particles.device)

        # === Keep track of object trajectory (mean of distribution) === #
        self.est_traj = []
        self.gt_traj = gt_traj # List of length num_objects containing GT obj positions

        # === Define prior-data model weight (hyperparameter) === #
        self.eta = configs['particle']['eta']

    
    def evaluate_particles(self, 
                           cur_pos: torch.Tensor, 
                           measurement: torch.Tensor,
                           frame_num: int
                    ) -> torch.Tensor:
        """
        Compute a score for each particle based on the measurement.

        Parameters:
        -----------
        cur_pos : relative distance between pixels (1, n_y, n_x, 3)
        measurement : space-time measurement (n_y, n_x, numBins)
        frame_num : current frame number

        Returns:
        --------
        scores : (num_particles, )
        """
    
        particles = self.particles.particles.clone()
        num_batches = self.particles.num_batches
        num_objects = self.particles.num_objects  
        _, n_y, n_x, num_bins = measurement.shape
        batch_size = self.particles.batch_size

        # === reshape and normalize reference image === #
        y_gt = measurement.clone() # (1, n_y, n_x, num_bins)

        # === process particles in batches === #
        scores = torch.zeros((self.particles.num_particles, )) 

        for i in range(num_batches):
            # === Extract current batch of particles === #
            idx1 = i * batch_size
            idx2 = min((i+1) * batch_size, self.particles.num_particles)
            batch = torch.clone(particles[idx1:idx2, :]) # (batch_size, 3*num_objects)

            # === forward pass === # 
            y_hat = torch.zeros((idx2-idx1, n_y, n_x, num_bins)).to(cur_pos.device)
            renders = []
            for j in range(num_objects):
                batch[:, 3*j+2] = torch.sign(batch[:, 3*j+2]) * batch[:, 3*j+2] ** 2 # convert z to v space
                with torch.no_grad():
                    cur_render = self.canons[j](cur_pos=cur_pos[:idx2-idx1], 
                                                deltas=batch[:, 3*j:3*(j+1)], 
                                                is_cam_motion=self.is_cam_tracking) # (batch_size, n_y, n_x, numBins)
                    
                    # === Compute iLCT of y_hat === #
                    non_zero_entries = torch.sum(cur_render > 0.1, dim=(1, 2, 3), keepdim=True) # (batch_size, 1, 1, 1)
                    
                    # normalize rendering
                    cur_render /= torch.linalg.vector_norm(cur_render, dim=(1, 2, 3), keepdim=True)
                    
                    # find projection of rendering onto measurement
                    projection = torch.sum(cur_render * y_gt, dim=(1, 2, 3), keepdim=True) #/ non_zero_entries

                    # project rendering onto measurement
                    y_hat += projection * cur_render 

            # === compute similarity between images using score function === #
            cur_scores = self.score_function(y_gt, y_hat)

            # === store scores === #
            scores[idx1:idx2] = cur_scores

        # === Get non-NaN minimum score === #
        non_nan_scores = scores[~torch.isnan(scores)]
        min_score = torch.min(non_nan_scores)

        # === Ensure scores are non-negative === #
        scores -= min_score

        # === Remove nan entries === #
        scores[torch.isnan(scores)] = 0

        # === Set particles with -z to have zero probability === #
        mask = particles[:, 2] < 0
        mask_idxs = torch.where(mask)[0]
        scores[mask_idxs] = 0

        return scores

    def resample_particles(self, scores: torch.Tensor, pnr: float = None):
        self.particles.resample_particles(scores, pnr)

    def propagate_particles(self):
        self.particles.propagate_particles()

    def compute_kde(self) -> List[torch.Tensor]:
        """
        Compute the kernel density estimate of the particle distribution.

        Returns:
        --------
        pdfs : list of length num_objects containing (num_x, num_y, num_z) 
                    tensor of the KDE
        """
        num_x, num_y, num_z = self.volume.shape[0:3]

        pdfs = []
        for j in range(self.particles.num_objects):
            # === Get positions for jth object === #
            particles = self.particles.particles.clone()[:, 3*j:3*(j+1)]

            # === Convert v position to z position === #
            particles[:, 2] = torch.sign(particles[:, 2]) * torch.sqrt(torch.abs(particles[:, 2]))
            
            # === Compute KDE of jth object === #
            pdf = torch.zeros((num_x, num_y, num_z)).to(self.particles.particles.device)
            for i in range(self.particles.num_particles):
                dist = torch.linalg.norm(self.volume - particles[i].reshape(1, 1, 1, 3), dim=-1)
                pdf += (1 / self.particles.num_particles) * (1 / (np.sqrt(2 * np.pi ) * self.sigma)) \
                            * torch.exp(-dist**2 / (2 * self.sigma **2))

            pdfs.append(pdf)

        return pdfs

    def plot_canonical(self, 
                    cur_pos: torch.Tensor,
                    query_locs : torch.Tensor,
                    frame_meas: torch.Tensor,
                    log_dir: str, 
                    frame_num: int,
                ) -> None:
        """
        Plot the canonical measurements in the current frame.

        Parameters:
        -----------
        cur_pos   : relative position of pixels (num_particles, n_y, n_x, 3)
        query_locs : query locations for canonical measurements (num_queries, 3*num_objects)
        frame_meas : space-time measurement (n_y, n_x, num_bins)
        log_dir   : directory to save image to 
        frame_num : current frame number
        gt_pos    : ground truth position in the current frame (3, ) 
        """

        num_objects = self.particles.num_objects
        n_y, n_x, num_bins = frame_meas.shape
        num_queries = query_locs.shape[0]

        # === Convert query locations to v space === #
        batch = torch.clone(query_locs)
        for i in range(num_objects):
            batch[:, 3*i+2] = torch.sign(batch[:, 3*i+2]) * batch[:, 3*i+2] ** 2

        # === Render canonical at queried locations === #
        y_hat = torch.zeros((num_queries, n_y, n_x, num_bins)).to(cur_pos.device)
        for j in range(num_objects):
            with torch.no_grad():
                y_hat += self.canons[j](cur_pos=cur_pos[:num_queries], 
                                        deltas=batch[:, 3*j:3*(j+1)], 
                                        is_cam_motion=self.is_cam_tracking) 
                

        # === Plot canonical measurements === #
        num_cols = 3
        num_rows = int(np.ceil((num_queries + 1 / num_cols)))
        plt.figure(figsize=(num_cols * 5, num_rows * 5)); k = 1
        plt.suptitle(f"Frame number {frame_num+1}")

        plt.subplot(num_rows, num_cols, k); k += 1
        plt.imshow(frame_meas.cpu().numpy().reshape(-1, num_bins),
                   cmap='hot', 
                   norm='linear',
                   interpolation='none')
        
        for i in range(num_queries):
            plt.subplot(num_rows, num_cols, k); k += 1
            plt.imshow(y_hat[i].cpu().numpy().reshape(-1, num_bins),
                       cmap='hot', 
                       interpolation='none')
            plt.title(f'Canon at {np.round(query_locs[i].cpu().numpy(), 2)}', fontweight='bold')
            # plt.colorbar()

        plt.subplot(num_rows, num_cols, k); k += 1
        plt.imshow(frame_meas.cpu().numpy().reshape(-1, num_bins),
                   cmap='hot', 
                   interpolation='none')
        
        from matplotlib.colors import LinearSegmentedColormap
        colors = [(0, 1, 0, 0), (0, 1, 0, 1)]  # Black to Green
        cmap = LinearSegmentedColormap.from_list("black_to_green", colors)
        
        plt.imshow(y_hat[0].cpu().numpy().reshape(-1, num_bins),
                   cmap=cmap, 
                   interpolation='none')
        
        plt.title(f'Score: {np.round(np.sum(frame_meas.cpu().numpy() * y_hat[0].cpu().numpy()) / (np.linalg.norm(frame_meas.cpu().numpy()) * np.linalg.norm(y_hat[0].cpu().numpy())), 2)}')
        


        plt.savefig(os.path.join(log_dir, 
                                 f'out/imgs/canon_frame_{frame_num+1:04d}.png'), 
                                 dpi=100)
        plt.close()

        