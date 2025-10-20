from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from utils import get_all_subclasses
from train.canon import FastSumOfParabolas
from typing import List

TRAIN_Z = True 
Z_VAL = 0.66 

TRAIN_Y = True
Y_VAL = 0.3

class Particle(nn.Module):
    """
    Particle parametrization of motion. Each particle represents a guess
    of the current object or camera position. Each particle is 3N-dimensional,
    where N is the number of objects being tracked. This implementation supports
    arbitrary number of objects for object tracking, but only one object for 
    camera tracking.
    """
    def __init__(self, configs, num_objects, device):
        particle_configs = configs['particle']

        # === Extract number of objects being tracked === #
        self.num_objects = num_objects

        # === Determine if camera or object tracking === #
        self.cam_tracking = configs['cam_tracking'] 
        if self.cam_tracking: 
            assert self.num_objects == 1, "Camera tracking can only support one hidden object"           
        
        # === Particle parameters === #
        self.num_particles = particle_configs['num_particles']
        self.vol_size = particle_configs['vol_size'] # initialized region size
        self.vol_center = particle_configs['vol_center'] # initialized region center

        # === Motion model parameters === #
        self.radius = particle_configs['radius']

        # === Initialize particles and velocity === #    
        self.particles = torch.zeros((self.num_particles, 3 * self.num_objects)).to(device)
        self.velocity = torch.zeros((self.num_particles, 3 * self.num_objects)).to(device)
        
        for i in range(self.num_objects):     
            self.particles[:, 3*i:3*(i+1)] = random_points_in_box(self.num_particles, 
                                                                  self.vol_size, 
                                                                  self.vol_center).to(device)
            
            self.velocity[:, 3*i:3*(i+1)] = random_points_in_sphere(self.num_particles,
                                                                    self.radius).to(device)

            if not TRAIN_Z:
                self.particles[:, 3*i+2] = Z_VAL
                self.velocity[:, 3*i+2] = 0

            if not TRAIN_Y:
                self.particles[:, 3*i+1] = Y_VAL
                self.velocity[:, 3*i+1] = 0

        # === Forward pass parameters === #
        self.batch_size = particle_configs['batch_size']
        self.num_batches = int(np.ceil(self.num_particles / self.batch_size))

        # === Instantiate motion model === #
        motion_model = particle_configs['motion_model'] 
        motion_classes = get_all_subclasses(MotionModel)
        if motion_model in motion_classes:
            kwargs = {'num_particles': self.num_particles, 'radius': self.radius}
            self.motion_model = motion_classes[motion_model](**kwargs)
        else:
            raise ValueError(f"Motion model '{motion_model}' not found!")

        # === Instantiate resampling function === #
        resampling_fn = particle_configs['resample_fn']
        if resampling_fn in globals():
            self.resampling_function = globals()[resampling_fn]
        else:
            raise ValueError(f"Resampling function '{resampling_fn}' not found!")
        
        # === Maintain logs of particle locations in previous frames === #
        self.cur_frame = 0
        self.motion_prior = []
        self.states = []
        self.indices = []
        self.scores = []
        if self.cam_tracking:
            self.theta_states = []

    def resample_particles(self, scores: torch.Tensor, pnr : float) -> None:
        """
        Resample particles to focus on high-likelihood particles.

        Parameters:
        -----------
        scores : likelihood of each particle (num_particles, )
        pnr    : peak to noise floor ratio of measurement

        Returns:
        --------
        None

        """

        # === Store previous particle locations === #
        particles_to_save = self.particles.clone()         
        self.motion_prior.append(particles_to_save)

        # === Resample particles based on scores === #
        indices = self.resampling_function(scores)

        # === Update particles === #
        self.particles = self.particles[indices]
        self.velocity = self.velocity[indices]

        # === Store current particle locations === #
        particles_to_save = self.particles.clone() 
        
        self.states.append(particles_to_save)
        self.scores.append(scores.clone())

        

    def propagate_particles(self) -> None:
        """
        Propagate particles based on motion model.
        """
        # === add noise to velocity based on motion model === #
        self.velocity = self.motion_model.forward(velocity=self.velocity)
        if not TRAIN_Z:
            self.velocity[:, 2] = 0 
        if not TRAIN_Y:
            self.velocity[:, 1] = 0
        if self.cam_tracking:
            self.velocity[:, 2] = 0

        # === propagate particle positions === #
        self.particles = self.particles + self.velocity
        
        for i in range(self.num_objects):
            self.particles[:, 3*i+2] = self.particles[:, 3*i+2] * (self.particles[:, 3*i+2] > 0) # restrict to positive z space

        # === Update current frame === #
        self.cur_frame += 1


# =============== Resampling Techniques ================= #


def systematic(scores : torch.Tensor) -> torch.Tensor:
        """
        Perform systematic resampling on particles based on their scores.
        
        Parameters:
        -----------
        scores    : tensor of particle scores/weights (N, )
        
        Returns:
        --------
        indices   : Resampled particles with shape (N, 3)
        """

        num_particles = scores.shape[0]
        
        # === Normalize scores to probabilities === #
        probabilities = scores / scores.sum()
        
        # === Calculate cumulative sum of probabilities === #
        cumulative_sum = torch.cumsum(probabilities, dim=0)
        
        # === Generate a random starting point === #
        u = torch.rand(1) / num_particles
        
        # === Generate sample points === #
        sample_points = (torch.arange(num_particles, dtype=torch.float32) + u) / num_particles
        
        # === Find indices of particles to be resampled === #
        indices = torch.searchsorted(cumulative_sum, sample_points)
        
        # === Ensure indices are within bounds === #
        indices = torch.clamp(indices, 0, num_particles-1)
        
        return indices


def residual(scores : torch.Tensor) -> torch.Tensor:
    """
    Perform residual resampling on a set of particles based on their weights.
    
    Parameters:
    -----------
    scores  : tensor of particle scores/weights (N, )
    
    Returns:
    --------
    indices: Indices of resampled particles (N, )
    """
    N = scores.shape[0]  # Number of particles

    # === Normalize scores to probabilities === #
    probabilities = (scores / scores.sum()).cpu()

    # === Replace NaN values with 0 === #
    probabilities[torch.isnan(probabilities)] = 0
    
    # === Step 1: Compute the deterministic part (integer copies) === #
    num_copies = np.floor(probabilities.numpy() * N).astype(int)  # Integer copies for each particle
    residual = probabilities * N - num_copies # Residual weights
    residual[torch.isnan(residual) | (residual < 0)] = 0 # Replace NaN values with 0
    residual /= residual.sum()                     # Normalize residual weights
    # residual[torch.isnan(residual)] = 0 # Replace NaN values with 0

    # === Add deterministic copies to the indices list === #
    indices = []
    for i in range(N):
        if num_copies[i] != 0:
            indices.extend([i] * num_copies[i])

    # === Step 2: Redistribute remaining particles using multinomial sampling === #
    num_residual_particles = N - len(indices) # Remaining particles to sample
    if num_residual_particles > 0:
        residual_indices = np.random.choice(range(N), size=num_residual_particles, p=residual)
        indices.extend(residual_indices)
    
    return torch.Tensor(indices).long().to(scores.device)


def multinomial(scores : torch.Tensor, thresh_pct : float = 0.2) -> torch.Tensor:
    """
    Sample indices from multinomial distribution.

    Parameters:
    -----------
    scores : tensor of particle scores (num_particles, )

    Returns:
    --------
    indices : indices of resampled particles (num_particles, )
    """

    num_particles = scores.shape[0]

    # === Clip scores to be greater than 0 === #
    # scores = torch.clip(scores, min=0)        

    # === Only use top k particles === #
    k = int(num_particles * thresh_pct)
    scores = top_k_mask(scores, k=k) * scores

    # === Normalize scores to get probabilities === #
    probabilities = scores / torch.sum(scores)
    
    # === Resample particles based on their probabilities === #
    dist = torch.distributions.categorical.Categorical(probs=probabilities)
    indices = dist.sample((num_particles, ))

    return indices



        
# ============ Motion Model Parametrizations ============ #

class MotionModel(ABC):
    def __init__(self, num_particles):
        super().__init__()
        self.num_particles = num_particles

    def forward(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        velocity : velocity of particle in previous frame (num_points, 3*num_objects)

        Returns:
        --------
        dx : new velocities (num_points, 3*num_objects)
        """
        pass

class RandomWalk(MotionModel):  
    def __init__(self, num_particles: int, radius: float):
        super().__init__(num_particles)

        self.radius = radius

    def forward(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Simulates displacements resulting from a random walk in 3D.
        Velocity isn't used here because previous velocity doesn't affect
        future velocities in a random walk.
        """
        
        # === Create zero-mean 3D Gaussian with stddev = radius === #
        dx = torch.randn_like(velocity).to(velocity.device) * self.radius
        # dx = torch.randn_like(velocity).to(velocity.device) 
        # dx /= torch.norm(dx, dim=-1, keepdim=True) 
        # dx *= self.radius

        return dx

class ConstantVelocity(MotionModel):
    def __init__(self, num_particles, radius : float, **kwargs):
        super().__init__(num_particles)

        self.radius = radius


    def forward(self, velocity: torch.Tensor):
        """
        Adds Gaussian noise to velocities. The mean of the Gaussian
        is the previous velocity, variance is 0.1 m. 
        """
        assert velocity.shape[1] % 3 == 0, "Velocity must be of shape (num_particles, 3*num_objects)"
        num_objects = velocity.shape[1] // 3

        # === Variance of motion model === #
        variance = torch.Tensor([self.radius]).to(velocity.device)

        # === Mean velocity of all particles === #
        mean_velocity = torch.mean(velocity, axis=0).reshape(1, 3*num_objects)
 
        # === Zero-mean, constant variance Gaussian noise === #
        eps = torch.randn_like(velocity).to(velocity.device) * variance #torch.sqrt(variance)

        # === Add noise to velocity === #
        new_velocity = velocity + eps
        # new_velocity = mean_velocity + eps

        return new_velocity

# ============ Helper Functions ============ #

def random_points_in_box(num_points: int, 
                         box_width: float, 
                         center: list
            ) -> torch.Tensor:
    """
    Samples random points within a 3D cube as an initialization for 
    particle positions.

    Parameters:
    -----------
    num_points  : number of particles
    box_width   : width of initialized region
    center      : center of initialized region 

    Returns:
    --------
    points   : initialized particle locations (num_points, 3)
    """

    x = (torch.rand(num_points) - 0.5) * box_width + center[0]
    y = (torch.rand(num_points) - 0.5) * box_width + center[1]
    z = (torch.rand(num_points) - 0.5) * box_width + center[2]

    points = torch.stack([x, y, z], dim=-1) 
    
    return points

def random_points_in_sphere(num_points: int, 
                            radius: float
                    ) -> torch.Tensor:
    """
    Samples random points within a 3D sphere as an initialization for 
    particle positions.

    Parameters:
    -----------
    num_points : number of particles
    radius     : max radius of sphere

    Returns:
    --------
    points   : initialized particle locations (num_points, 3)
    """
    # === randomly choose radius in the range [0, r] === #
    r = radius * torch.pow(torch.rand(num_points), 1/3)

    # === uniformly distribute theta (azimuthal angle) in [0, 2*pi] === #
    phi = 2 * torch.pi * torch.rand(num_points)

    # === uniformly distribute phi (polar angle) === #
    theta = torch.pi * torch.rand(num_points)

    # === convert spherical to cartesian coordinates === #
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta) 

    # === concatneate coordinates === #
    points = torch.stack([x, y, z], dim=1)

    return points


def top_k_mask(vector, k=20):    
    if k >= vector.shape[0]:
        return torch.ones_like(vector).to(vector.device)

    # === Get the indices of the top k highest entries in the vector === #
    top_k_values, top_k_indices = torch.topk(vector, k)
    
    # Create a boolean mask with the same shape as the vector
    mask = torch.zeros_like(vector).to(vector.device)
    
    # Set the positions of the top k entries to True in the mask
    mask[top_k_indices] = 1
    
    return mask