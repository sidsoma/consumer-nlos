import os 
import shutil
import yaml
import importlib
import cv2
import glob
from tqdm import tqdm
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import operator
# import seaborn as sns

def run_function(func_name, kwargs, module_name=None):
    if module_name:
        module = importlib.import_module(module_name)
        if hasattr(module, func_name):
            func = getattr(module, func_name)
            return func(**kwargs)
    if func_name in globals():
        return globals()[func_name](**kwargs)
    else:
        raise Exception(f"Function {func_name} not found")
    
def get_all_subclasses(cls):
    """
    Recursively find all subclasses of a given parent class.

    Parameters:
    ----------
    cls: The parent class.

    Returns:
    --------
    dict: A dictionary where keys are subclass names and values 
           are the class objects.
    
    """
    subclasses = {}
    for subclass in cls.__subclasses__():
        # Add the subclass to the dictionary
        subclasses[subclass.__name__] = subclass

    return subclasses

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")
            return None
        
def create_or_overwrite_directory(path):
    # Check if the directory already exists
    if os.path.exists(path):
        # Remove the existing directory and all its contents
        shutil.rmtree(path)
    
    # Create the new directory
    os.makedirs(path)

def load_configs(config_dir='.'):
    camera_params = load_yaml(os.path.join(config_dir, 'camera.yaml'))
    optim_params = load_yaml(os.path.join(config_dir, 'optim.yaml'))
    canon_params = load_yaml(os.path.join(config_dir, 'canonical.yaml'))

    data_params = load_yaml(os.path.join(config_dir, 'data/data.yaml'))
    if data_params['use_real_data']:
        data_params.update(load_yaml(os.path.join(config_dir, 'data/real.yaml')))
        obj_name = data_params['obj_name']
        data_params.update(load_yaml(os.path.join(config_dir, f'data/real/{obj_name}.yaml')))
    else:
        data_params.update(load_yaml(os.path.join(config_dir, 'data/sim.yaml')))
        obj_name = data_params['obj_name']
        data_params.update(load_yaml(os.path.join(config_dir, f'data/sim/{obj_name}.yaml')))

    log_params = optim_params['log']
    particle_params = optim_params['particle']
    particle_params.update(data_params['particle'])

    configs = {

        'camera': camera_params, 
        'optim': optim_params, 
        'canon': canon_params, 
        'log': log_params,
        'data': data_params,
        'particle': particle_params

    }

    return configs


def delete_from_dict(d: dict, keys: list):
    """
    Parameters:
    -----------
    d : dict
    keys : list of keys to delete from d

    Returns: 
    --------
    dict : d with keys removed
    """
    for key in keys:
        if key in d:
            del d[key]
    return d


def convert_pngs_to_video(out_dir, data_dir: str = None, imgs = None, fps = 6.0):
    """
    Convert a directory of PNG images to a video file.

    Parameters:
    -----------
    data_dir : the directory containing the PNG images.

    Returns:
    --------
    None
    """
    # Ensure output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    assert data_dir or imgs, "Either data_dir or imgs must be provided"
    assert not data_dir or not imgs, "Only one of data_dir or imgs should be provided"
    
    if data_dir:
        # === Get the list of PNG files in the directory === #
        img_files = sorted(glob.glob(os.path.join(data_dir, '*.png')))

        # === Read the first image to get the dimensions === #
        img = cv2.imread(img_files[0])
        if img is None:
            raise ValueError(f"Failed to read the first image: {img_files[0]}")
        height, width, _ = img.shape
    elif imgs:
        if not imgs:
            raise ValueError("The imgs list is empty.")
        img0 = imgs[0]
        if img0 is None:
            raise ValueError("The first image in imgs is None.")
        if img0.ndim != 3 or img0.shape[2] != 3:
            raise ValueError(f"Images must have shape (height, width, 3), got {img0.shape}")
        if img0.dtype != np.uint8:
            raise ValueError(f"Images must be of dtype uint8, got {img0.dtype}")
        height, width = img0.shape[0:2]

    # === Define the codec and create a VideoWriter object === #
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_path = os.path.join(out_dir, 'out.mp4')
    # out = cv2.VideoWriter(video_path, fourcc, fps, (width, height), isColor=True)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # or *'H264'
    video_path = os.path.join(out_dir, 'out.mp4')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height), isColor=True)
    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer at {video_path}. Check that the directory exists and the codec is supported.")

    if data_dir:
        # === Write each image to the video file === #
        for img_file in tqdm(img_files, desc="Saving video"):
            img = cv2.imread(img_file)
            if img is None:
                print(f"Warning: Failed to read image {img_file}, skipping.")
                continue
            if img.shape[0:2] != (height, width):
                img = cv2.resize(img, (width, height))
            out.write(img)
    else:
        for idx, img in enumerate(tqdm(imgs, desc="Saving video")):
            if img is None:
                print(f"Warning: Image at index {idx} is None, skipping.")
                continue
            if img.shape[0:2] != (height, width):
                img = cv2.resize(img, (width, height))
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            out.write(img)

    out.release()
    print("Video saved to ", video_path)

def convert_particles_to_image(particles : List[np.array], cam_pos : np.array, pt_clouds : np.array):
    """
    Plot particle locations and save figures as image arrays.

    Parameters:
    -----------
    particles  : list of particle locations, each entry contains (num_particles, 3*num_objects)
    gt_pos     : list containing positions of landmarks (num_landmarks, 3)

    Returns:
    --------
    imgs : list of image arrays of length num_frames
    """
    num_frames = len(particles)

    imgs = []
    for frame_num in tqdm(range(num_frames), "Creating plots"):
        # === Create scatter plot using seaborn === #
        fig = plt.figure(figsize=(10, 10))
        plt.grid(True)
        plt.xlim([-1.8, 0.1])
        plt.ylim([-0.1, 1.9])

        # === Set font to be times new roman === #
        plt.rcParams.update({'font.family': 'Times New Roman'})
        fontsize = 15
        title_fontsize = 20

        # === plot particles as KDE === #
        # sns.kdeplot(x=particles[frame_num][:, 0], 
        #             y=particles[frame_num][:, 2],
        #             alpha=1, fill=True, cmap='hot')

        # plot particles as points
        plt.plot(particles[frame_num][:, 0], particles[frame_num][:, 2], 'o', alpha=1)

        # === plot GT patches === #
        patch_width = [0.1, 0, 0.1]

        gt_pos = np.array([[-0.80, 0.3, 0.66],
                        [-1.4, 0.3, 0.66], 
                        [-0.80, 0.3, 1.25], 
                        [-1.4, 0.3, 1.25]])

        for patch_num in range(len(gt_pos)):
            rect = plt.Rectangle((gt_pos[patch_num, 0] - patch_width[0]/2, gt_pos[patch_num, 2] - patch_width[2]/2), 
                                patch_width[0], patch_width[2], 
                                fill=True, 
                                color='green', 
                                linewidth=0)
            plt.gca().add_patch(rect)

        # === plot occluder === #
        occluder_x = -0.30

        occluder_y_min = 0.65; occluder_y_max = 1.6
        plt.vlines(occluder_x, occluder_y_min, occluder_y_max, color='black', linewidth=6)

        # === plot camera position and camera fov === #
        plt.plot(cam_pos[0], cam_pos[2], 'o', color='gray', markersize=10)
        ray_1 = np.array([[cam_pos[0], cam_pos[2]],
                        [np.min(pt_clouds[:, 0]), 0]])
        ray_2 = np.array([[cam_pos[0], cam_pos[2]],
                        [np.max(pt_clouds[:, 0]), 0]])

        plt.plot(ray_1[:, 0], ray_1[:, 1], color='gray', linewidth=2, linestyle='--')
        plt.plot(ray_2[:, 0], ray_2[:, 1], color='gray', linewidth=2, linestyle='--')

        # === draw planar rectangle centered at z = 0 === #
        rect = plt.Rectangle((-2, -0.1), 2.5, 0.1, fill=True, color='k', linewidth=3)
        plt.gca().add_patch(rect)

        # === Invert axes to match previous plot === #
        # plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()

        # === Set title, xlabel, ylabel === #
        plt.title(f'Frame {frame_num+1}', fontsize=title_fontsize)
        plt.xlabel('X (m)', fontsize=fontsize)
        plt.ylabel('Z (m)', fontsize=fontsize)

        # === increase tick font size === #
        plt.tick_params(axis='both', which='major', labelsize=fontsize)

        # === Set plot spacing === #
        plt.tight_layout()
        
        # === Extract plot as array === #
        fig.canvas.draw()
        rgba_buf = fig.canvas.buffer_rgba()
        (w, h) = fig.canvas.get_width_height()
        
        rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8)
        
        if reduce(operator.mul, rgba_arr.shape) == 16*h*w:
            rgba_arr = rgba_arr.reshape((2*h, 2*w, 4))
        else:
            rgba_arr = rgba_arr.reshape((h, w, 4))

        # === Convert rgba_arr to bgr === #
        bgr_arr = rgba_arr[:, :, [2, 1, 0, 3]][..., :3]

        # === Save image and close plot === #
        imgs.append(bgr_arr)
        plt.close()

    return imgs



# def convert_particles_to_image(particles : List[np.array], 
#                                scores : List[np.array],
#                                vol_center : list, 
#                                vol_size : float,
#                                gt_pos : List[np.array] = None, 
#                         ):
#     """
#     Plot particle locations and save figures as image arrays.

#     Parameters:
#     -----------
#     particles  : list of particle locations, each entry contains (num_particles, 3*num_objects)
#     gt_pos     : list containing ground truth position of each object, 
#                     each entry has shape (num_frames, 3)
#     vol_center : center of the volume
#     vol_size   : size of the volume

#     Returns:
#     --------
#     imgs : list of image arrays of length num_frames
#     """
#     num_objects = particles[0].shape[1] // 3

#     # === Extract particle bounds === #
#     x_min = vol_center[0] - vol_size / 2
#     x_max = vol_center[0] + vol_size / 2
#     y_min = vol_center[1] - vol_size / 2
#     y_max = vol_center[1] + vol_size / 2
#     z_min = vol_center[2] - vol_size / 2
#     z_max = vol_center[2] + vol_size / 2

#     # === Compute images === #
#     imgs = []
#     num_frames = len(particles)
#     est_traj = np.zeros((num_frames, 3*num_objects))
#     for i in tqdm(range(num_frames), desc="Creating plots"):
#         # === Extract mean location === #
#         pred_loc = np.mean(particles[i], axis=0) # (3*num_objects,)
#         # pred_loc = particles[i][np.argmax(scores[i])] # (3*num_objects,)
#         est_traj[i] = pred_loc

#         # === Create figure for (i+1)-th frame === #
#         num_rows = num_objects
#         num_cols = 3
#         plot_height = 5
#         fig = plt.figure(figsize=(plot_height * num_cols / num_rows, plot_height))
#         plt.suptitle(f'Frame {i+1}')

#         idx = 1
#         for j in range(num_objects):
#             # === Plot x-y projection === #
#             plt.subplot(num_rows, num_cols, idx); idx += 1
#             plt.plot(particles[i][:, 3*j], particles[i][:, 3*j+1], 'o')
#             plt.xlim([x_min, x_max])
#             plt.ylim([y_min, y_max])
#             plt.gca().set_aspect('equal', adjustable='box')
#             plt.xlabel('X')
#             plt.ylabel('Y')
#             plt.gca().invert_xaxis()

#             # plt.plot(gt_pos[j][i, 0], gt_pos[j][i, 1], 'go', ms=10) #  plot gt position
#             # plt.plot(gt_pos[:i+1, 0], gt_pos[:i+1, 1], 'g-')
#             if gt_pos is not None:
#                 plt.plot(gt_pos[j][:, 0], gt_pos[j][:, 1], 'g-') # plot gt trajectory


#             plt.plot(pred_loc[3*j], pred_loc[3*j+1], 'ro', ms=10) # plot estimated position
#             plt.plot(est_traj[:i+1, 3*j], est_traj[:i+1, 3*j+1], 'r-') # plot estimated trajectory

#             # === Plot x-z projection === #
#             plt.subplot(num_rows, num_cols, idx); idx += 1
#             plt.plot(particles[i][:, 3*j], particles[i][:, 3*j+2], 'o')
#             plt.xlim([x_min, x_max])
#             plt.ylim([z_min, z_max])
#             plt.gca().set_aspect('equal', adjustable='box')
#             plt.xlabel('X')
#             plt.ylabel('Z')
#             plt.gca().invert_xaxis()

#             if gt_pos is not None:
#                 # plt.plot(gt_pos[j][i, 0], gt_pos[j][i, 2], 'go', ms=10) #  plot gt position
#                 # plt.plot(gt_pos[:i+1, 0], gt_pos[:i+1, 2], 'g-')
#                 plt.plot(gt_pos[j][:, 0], gt_pos[j][:, 2], 'g-') # plot gt trajectory

#             plt.plot(pred_loc[3*j], pred_loc[3*j+2], 'ro', ms=10) # plot estimated position
#             plt.plot(est_traj[:i+1, 3*j], est_traj[:i+1, 3*j+2], 'r-') # plot estimated trajectory

#             # === Create plot title === #
#             plt.title(f'Object {j+1} Positions', fontweight='bold')

#             # === Plot y-z projection === #
#             plt.subplot(num_rows, num_cols, idx); idx += 1
#             plt.plot(particles[i][:, 3*j+1], particles[i][:, 3*j+2], 'o')
#             plt.xlim([y_min, y_max])
#             plt.ylim([z_min, z_max])
#             plt.gca().set_aspect('equal', adjustable='box')
#             plt.xlabel('Y')
#             plt.ylabel('Z')

#             if gt_pos is not None:
#                 # plt.plot(gt_pos[j][i, 1], gt_pos[j][i, 2], 'go', ms=10) # plot gt position
#                 # plt.plot(gt_pos[:i+1, 1], gt_pos[:i+1, 2], 'g-') # plot gt trajectory
#                 plt.plot(gt_pos[j][:, 1], gt_pos[j][:, 2], 'g-') # plot gt trajectory

#             plt.plot(pred_loc[3*j+1], pred_loc[3*j+2], 'ro', ms=10) # plot estimated position
#             plt.plot(est_traj[:i+1, 3*j+1], est_traj[:i+1, 3*j+2], 'r-') # plot estimated trajectory

#         # === Set plot spacing === #
#         plt.tight_layout()
        
#         # === Extract plot as array === #
#         fig.canvas.draw()
#         rgba_buf = fig.canvas.buffer_rgba()
#         (w, h) = fig.canvas.get_width_height()
        
#         rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8)
        
#         if reduce(operator.mul, rgba_arr.shape) == 16*h*w:
#             rgba_arr = rgba_arr.reshape((2*h, 2*w, 4))
#         else:
#             rgba_arr = rgba_arr.reshape((h, w, 4))

#         # === Convert rgba_arr to bgr === #
#         bgr_arr = rgba_arr[:, :, [2, 1, 0, 3]][..., :3]

#         # === Save image and close plot === #
#         imgs.append(bgr_arr)
#         plt.close()
    
#     return imgs

