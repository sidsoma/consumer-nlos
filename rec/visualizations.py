import matplotlib.pyplot as plt
from math import ceil
import numpy as np
import os

def plot_hist_images(hists : list, 
                     log_dir : str, 
                     norm : str ='linear',
                     lct: bool = False,
                ) -> None:
    """
    Plot measured histograms in each frame.

    Parameters:
    -----------
    hists   : List of num_frames histograms (..., num_bins)
    log_dir : Directory to save the plot
    norm    : Normalization for the colormap

    Returns:
    --------
    None
    """
    num_bins = hists[0].shape[-1]   
    num_imgs = len(hists)
    num_cols = 5
    num_rows = ceil(num_imgs / num_cols)

    # === Plot histograms === #
    plt.figure(figsize=(3*num_cols, 3*num_rows))
    for idx, hist in enumerate(hists):
        hist = hist.reshape(-1, num_bins)
        plt.subplot(num_rows, num_cols, idx+1)
        plt.imshow(hist, norm=norm, cmap='hot')
        plt.colorbar()

    # === Save image === #
    plt.title('3-Bounce Histograms')
    label = '_lct' if lct else ''
    plt.savefig(os.path.join(log_dir, f'hists{label}.png'))
    plt.close()


def plot_point_clouds(pt_clouds : list, 
                      log_dir : str,
                      fig_title : str,
                      filename : str) -> None:
    """
    Plot locations of sampled points on wall.

    Parameters:
    -----------
    pt_clouds : List of num_frames point clouds (num_pixels, 3)
    log_dir   : Directory to save the plot

    Returns:
    --------
    None

    """
    fig = plt.figure(figsize=(16, 4), facecolor='white')
    fig.suptitle(fig_title)
    num_pixels = pt_clouds[0].shape[0]

    # === Create 3D plot === #
    ax = fig.add_subplot(141, projection='3d')
    for idx, pt_cloud in enumerate(pt_clouds):
        pt_cloud = pt_cloud.reshape(num_pixels, 3)
        ax.plot3D(pt_cloud[:, 0], pt_cloud[:, 2], pt_cloud[:, 1], 'o')

    # Label plot 
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    

    # Flip x and z axis 
    ax.invert_xaxis()
    ax.set_ylim([-0.5, 0.5])
    ax.invert_yaxis()

    # === Plot x-y projection === #
    ax = fig.add_subplot(142)
    for idx, pt_cloud in enumerate(pt_clouds):
        pt_cloud = pt_cloud.reshape(num_pixels, 3)
        ax.scatter(pt_cloud[:, 0], pt_cloud[:, 1])
    
    # Label plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.invert_xaxis()

    # === Plot x-z projection === #
    ax = fig.add_subplot(143)
    for idx, pt_cloud in enumerate(pt_clouds):
        pt_cloud = pt_cloud.reshape(num_pixels, 3)
        ax.scatter(pt_cloud[:, 0], pt_cloud[:, 2])
    
    # Label plot
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.invert_xaxis()

    # === Plot y-z projection === #

    ax = fig.add_subplot(144)
    for idx, pt_cloud in enumerate(pt_clouds):
        pt_cloud = pt_cloud.reshape(num_pixels, 3)
        ax.scatter(pt_cloud[:, 1], pt_cloud[:, 2])
    
    # Label plot
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    
    # === Remove border around figure and add spacing === #
    plt.subplots_adjust(hspace=0.6)
    plt.subplots_adjust(wspace=0.4)

    for i, ax in enumerate(fig.get_axes()):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i == 0:
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

    # === Save plot === #
    plt.savefig(os.path.join(log_dir, f'{filename}.png'))
    plt.close()


def plot_obj_trajectories(object_shifts: np.array, log_dir: str, plot_title: str, save_file: str) -> None:
    """
    Plot object trajectory and save to log directory.

    Parameters:
    -----------
    object_shifts : Object shifts in meters (num_frames, 3)
    log_dir       : Directory to save the plot
    plot_title    : Title of the plot
    save_file     : Filename to save the plot

    Returns:
    -–------
    None
    """

    num_objects = len(object_shifts)
    num_frames = object_shifts[0].shape[0]

    fig = plt.figure(figsize=(16, 4), facecolor='white')
    fig.suptitle(plot_title)

    # === Plot 3D object trajectory === #
    ax = fig.add_subplot(141, projection='3d')
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(num_objects):
        ax.plot3D(object_shifts[i][:, 0], object_shifts[i][:, 2], object_shifts[i][:, 1], c=colors[i], label=f'Object {i+1}')
    
    # === Place plane at z = 0 === #
    x_min = min([np.min(object_shifts[i][:, 0]) for i in range(num_objects)])
    x_max = max([np.max(object_shifts[i][:, 0]) for i in range(num_objects)])
    y_min = min([np.min(object_shifts[i][:, 1]) for i in range(num_objects)])
    y_max = max([np.max(object_shifts[i][:, 1]) for i in range(num_objects)])
    
    if np.abs(x_max - x_min) < 0.1:
        x_min -= 0.5
        x_max += 0.5
    if np.abs(y_max - y_min) < 0.1:
        y_min -= 0.5
        y_max += 0.5

    X = np.linspace(x_min, x_max, 10)
    Y = np.linspace(y_min, y_max, 10)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Z, Y, alpha=0.5, color='green', label='Wall')
    
    # === Label plot === #
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.legend()

    # === Flip x axis === #
    ax.invert_xaxis()
    ax.invert_yaxis()

    # === set z bounds === #
    z_min = min([np.min(object_shifts[i][:, 2]) for i in range(num_objects)])
    z_max = min([np.max(object_shifts[i][:, 2]) for i in range(num_objects)])

    if np.abs(z_max - z_min) < 0.1:
        z_min -= 0.5
        z_max += 0.5
    ax.set_ylim([z_max, z_min])

    # === Plot x-y slice === #
    ax = fig.add_subplot(142)
    for i in range(num_objects):
        ax.scatter(object_shifts[i][:, 0], object_shifts[i][:, 1])
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # === Plot x-z slice === #
        ax = fig.add_subplot(143)
        ax.scatter(object_shifts[i][:, 0], object_shifts[i][:, 2])
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([z_min, z_max])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')

        # === Plot y-z slice === #
        ax = fig.add_subplot(144)
        ax.scatter(object_shifts[i][:, 1], object_shifts[i][:, 2])
        ax.set_xlim([y_min, y_max])
        ax.set_ylim([z_min, z_max])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')

        plt.legend(f'Object {i+1}')

    plt.savefig(os.path.join(log_dir, f'{save_file}.png'))
    plt.close()

def plot_cam_trajectories(cam_shifts: np.array, log_dir: str, plot_title: str, save_file: str) -> None:
    """
    Plot object trajectory and save to log directory.

    Parameters:
    -----------
    object_shifts : Object shifts in meters (num_frames, 3)
    log_dir       : Directory to save the plot
    plot_title    : Title of the plot
    save_file     : Filename to save the plot

    Returns:
    -–------
    None
    """

    num_frames = cam_shifts.shape[0]

    fig = plt.figure(figsize=(16, 4), facecolor='white')
    fig.suptitle(plot_title)

    # === Plot 3D object trajectory === #
    ax = fig.add_subplot(141, projection='3d')
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    ax.plot3D(cam_shifts[:, 0], cam_shifts[:, 2], cam_shifts[:, 1], c=colors[0])
    
    # === Place plane at z = 0 === #
    x_min = np.min(cam_shifts[:, 0])
    x_max = np.max(cam_shifts[:, 0]) 
    y_min = np.min(cam_shifts[:, 1])
    y_max = np.max(cam_shifts[:, 1]) 
    
    if np.abs(x_max - x_min) < 0.1:
        x_min -= 0.5
        x_max += 0.5
    if np.abs(y_max - y_min) < 0.1:
        y_min -= 0.5
        y_max += 0.5

    X = np.linspace(x_min, x_max, 10)
    Y = np.linspace(y_min, y_max, 10)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Z, Y, alpha=0.5, color='green', label='Wall')
    
    # === Label plot === #
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.legend()

    # === Flip x axis === #
    ax.invert_xaxis()
    ax.invert_yaxis()

    # === set z bounds === #
    z_min = np.min(cam_shifts[:, 2])
    z_max = np.max(cam_shifts[:, 2]) 

    if np.abs(z_max - z_min) < 0.1:
        z_min -= 0.5
        z_max += 0.5
    ax.set_ylim([z_max, z_min])

    # === Plot x-y slice === #
    ax = fig.add_subplot(142)
    ax.scatter(cam_shifts[:, 0], cam_shifts[:, 1])
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # === Plot x-z slice === #
    ax = fig.add_subplot(143)
    ax.scatter(cam_shifts[:, 0], cam_shifts[:, 2])
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([z_min, z_max])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')

    # === Plot y-z slice === #
    ax = fig.add_subplot(144)
    ax.scatter(cam_shifts[:, 1], cam_shifts[:, 2])
    ax.set_xlim([y_min, y_max])
    ax.set_ylim([z_min, z_max])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')

    plt.savefig(os.path.join(log_dir, f'{save_file}.png'))
    plt.close()

    
def plot_all_motion(object_shifts: np.array, cam_positions: np.array, pt_clouds: list, log_dir: str) -> None:
    """
    Plot object trajectory, camera trajectory, and sampled points on wall.

    Parameters:
    -----------
    object_shifts : Object shifts in meters (num_frames, 3)
    cam_positions : Camera positions in meters (num_frames, 3)
    pt_clouds     : List of num_frames point clouds (num_pixels, 3)
    log_dir       : Directory to save the plot

    Returns:
    --------
    None
    """
    num_objects = len(object_shifts)

    plt.figure()
    ax = plt.axes(projection='3d')
    
    # === Plot sampled points on wall === #
    for idx, pt_cloud in enumerate(pt_clouds):
        pt_cloud = pt_cloud.reshape(144, 3)

        ax.scatter3D(pt_cloud[:, 0], 
                     pt_cloud[:, 2], 
                     pt_cloud[:, 1])

    # === Place square at z = 0 === #
    x_min = min([np.min(object_shifts[i][:, 0]) for i in range(num_objects)])
    x_max = max([np.max(object_shifts[i][:, 0]) for i in range(num_objects)])
    y_min = min([np.min(object_shifts[i][:, 1]) for i in range(num_objects)])
    y_max = max([np.max(object_shifts[i][:, 1]) for i in range(num_objects)])


    X = np.linspace(x_min, x_max, 10)
    Y = np.linspace(y_min, y_max, 10)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Z, Y, alpha=0.2, color='grey')

    
    # === Plot 3D object trajectories === #
    num_points = object_shifts[0].shape[0]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for j in range(num_objects):
        ax.scatter3D(object_shifts[j][:, 0], 
                     object_shifts[j][:, 2], 
                     object_shifts[j][:, 1], 
                     color=colors[j],
                     label=f'Object {j+1}')

    
    # === Plot 3D camera trajectory === #
    for i in range(1, num_points):
        ax.scatter3D(cam_positions[i, 0], 
                  cam_positions[i, 2], 
                  cam_positions[i, 1], 
                  color='g', 
                  alpha=np.clip(i/(num_points-2), 0, 1))
    ax.scatter3D(cam_positions[0, 0], 
                 cam_positions[0, 2], 
                 cam_positions[0, 1], 
                 c='g')
    ax.scatter3D(cam_positions[-1, 0], 
                 cam_positions[-1, 2], 
                 cam_positions[-1, 1], 
                 c='g',
                 label='Cam Trajectory')
    
    # === Label plot === #
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.legend()
    plt.title('Handheld NLOS Setup')

    # === Flip x axis === #
    ax.invert_xaxis()
    ax.invert_yaxis()

    plt.savefig(os.path.join(log_dir, 'combined.png'))
    plt.close()

    # === Plot 3D
# def plot_cam_positions(cam_positions, pt_clouds, num_cam_positions):
#     numPositions = cam_positions.shape[0]
#     cam_positions_cm = 100 * cam_positions
#     plt.figure()
#     plt.subplot(1, 2, 1)
#     for i in range(numPositions):
#         plt.plot(cam_positions_cm[i, 0], cam_positions_cm[i, 1], 'o')
#         if i != 0:
#             x = cam_positions_cm[i-1, 0]
#             y = cam_positions_cm[i-1, 1]
#             dx = cam_positions_cm[i, 0] - x
#             dy = cam_positions_cm[i, 1] - y
#             plt.arrow(x, y, 0.8*dx, 0.8*dy, head_width=0.3, head_length=0.9, length_includes_head=True)
    
#     plt.xlim([np.min(cam_positions_cm[:, 0])-5, np.max(cam_positions_cm[:, 0])+5])
#     plt.ylim([np.min(cam_positions_cm[:, 1])-5, np.max(cam_positions_cm[:, 1])+5])
#     plt.xlabel('X (cm)')
#     plt.ylabel('Y (cm)')
#     plt.gca().invert_xaxis()

#     # === Visualize all camera pixel locations === #
#     plt.subplot(1, 2, 2)
#     for i in range(num_cam_positions):
#         x_pos = np.ndarray.flatten(pt_clouds[i][:, :, 0])
#         y_pos = np.ndarray.flatten(pt_clouds[i][:, :, 1])
#         plt.scatter(x_pos, y_pos)
#     plt.gca().invert_xaxis() 
#     plt.xlabel('X (m)')
#     plt.ylabel('Y (m)')