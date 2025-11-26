# === Functions for data loading === #
from data.dataloader import load_data, compute_lct
from train.dataloader import create_data_loader

# === Functions for motion/canonical representation and optimization === #
from train.canon import FastSumOfParabolas
from train.motion import Particle
from train.model import ParticleModel
from train.optim import particle_optimization

# === Functions for visualization === #
from utils import convert_pngs_to_video, convert_particles_to_image, load_yaml

# === Standard libraries === #
import numpy as np
import torch
import os
from tqdm import tqdm
import logging
import time
import warnings
warnings.filterwarnings('ignore')

start_time = time.time()

configs = load_yaml('configs/tracking.yaml')
device = 'cpu' 

cam_pos = [0, 0, 0.82] # depth of camera from wall pre-calibrated (used only for visualization)

# === Configure logger === #
log_dir = os.path.join('results', configs['capture_name'])
if not os.path.exists('results'):
    os.mkdir('results')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filename=os.path.join(log_dir, 'training.log'),  # Specify the file name here
    filemode='w'  # 'w' mode overwrites the file, 'a' would append
)
logger = logging.getLogger()
for handler in logger.handlers[:]:
    if isinstance(handler, logging.StreamHandler):
        logger.removeHandler(handler)

file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)     

# === Load histogram and point cloud data === #
#SWAP WITH YOUR OWN HISTOGRAM AND POINT CLOUD DATA #
# note: we assume that the histogram has already been processed to crop out the 1-bounce light.
hists, pt_clouds = load_data(os.path.join(configs['data_dir'], configs['capture_name'])) # Lists of length num_frames, each entry containing (n_y, n_x, num_bins) array

# compute the LCT of the histograms
hists_lct = compute_lct(hists, isDiffuse=configs['isDiffuse'], num_lct_bins=configs['num_lct_bins'])

# === Instantiate canonical representation === #
canon_reps = []
for i, canon_dir in enumerate(configs['canons']): 
    print(f"Canonical {i+1}: {canon_dir}") 
    points = np.load(canon_dir) # load point cloud

    if configs['load_voxel_from_file'] and os.path.exists(canon_dir.replace('.npy', '_voxel.npy')):
        loaded_voxel = np.load(canon_dir.replace('.npy', '_voxel.npy')) # load voxelized canonical
    else:
        loaded_voxel = None

    canon_rep = FastSumOfParabolas(configs, points, device, loaded_voxel=loaded_voxel) # instantiate canonical measurement

    if configs['load_voxel_from_file'] and loaded_voxel is None:
        np.save(canon_dir.replace('.npy', '_voxel.npy'), canon_rep.canon_voxel.detach().cpu().numpy())

    canon_reps.append(canon_rep.to(device))

# === Create data loader === #
hists_filter = [torch.Tensor(hist).to(device) for hist in hists_lct]
pt_clouds_filter = [torch.Tensor(pt_cloud).to(device) for pt_cloud in pt_clouds]
data_loader = create_data_loader(hists=hists_filter, 
                                 pt_clouds=pt_clouds_filter,
                                 batch_size=1, 
                                 shuffle=False)

# === Instantiate particle motion reprepresentation === #
motion_rep = Particle(configs, len(canon_reps), device)

# === Create particle filter model === #
model = ParticleModel(configs, canon_reps, motion_rep)

# === Optimization Loop === #
particle_optimization(
    model=model, 
    data_loader=data_loader, 
    logger=logger, 
    log_dir=log_dir, 
)

# === Save particles and scores === #
states = model.particles.states
states = [state.cpu().numpy() for state in states]

scores = model.particles.scores
scores = [score.cpu().numpy() for score in scores]

vol_center = configs['particle']['vol_center']
vol_size = configs['particle']['vol_size']
save_dict = {
    'particles': states, 
    'scores': scores,
    'vol_center': vol_center,
    'vol_size': vol_size,
    'pt_clouds': pt_clouds,
    'cam_pos': cam_pos         
}

torch.save(save_dict, os.path.join(log_dir, 'particles.pth'))

print(f"Particles saved to {log_dir}")

# === Convert particles to video === #
imgs = convert_particles_to_image(particles=states, cam_pos=cam_pos, pt_clouds=pt_clouds[0].reshape(-1, 3))

print(f"{len(imgs)} frames generated")

convert_pngs_to_video(out_dir=log_dir,
                      imgs=imgs,
                      fps=14.0)

print(f"Video saved to {log_dir}")

end_time = time.time()

print("Done! Runtime: ", np.round(end_time - start_time, 2), " seconds")