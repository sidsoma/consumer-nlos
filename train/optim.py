from train.model import ParticleModel
from tqdm import tqdm
import logging
import torch


def particle_optimization(
        model: ParticleModel, 
        data_loader: torch.utils.data.DataLoader, 
        logger: logging.Logger, 
        log_dir: str, 
    ):

    num_frames = len(data_loader)
    data_iter = iter(data_loader)

    for canon in model.canons:
        canon.train_bg = False

    for i in tqdm(range(num_frames), desc="Particle Filtering"):
        # === load current frame === # s
        _, cur_frame, cur_pt_cloud = next(data_iter)
        cur_pt_cloud = cur_pt_cloud.repeat(model.particles.batch_size, 1, 1, 1)

        # === compute score === #
        logger.info(f"Evaluating particles in frame {i+1}...")
        scores = model.evaluate_particles(cur_pt_cloud, cur_frame, i+1)

        # === resample particles === #
        logger.info(f"Resampling particles...")
        model.resample_particles(scores ** model.eta)

        # === propagate particles based on motion model === #
        logger.info(f"Propagating particles to frame {i+1}...")
        model.propagate_particles()

