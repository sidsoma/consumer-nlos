from rec.backprojection import backprojection
from rec.visualizations import plot_hist_images, plot_point_clouds
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

t0 = 13
bin_width = 88E-12

gates = [30, 90]


# === Load data === #
data_dir = 'captured_data/ams_U_reconstruction'
log_dir = 'results/reconstruction'

os.makedirs(log_dir, exist_ok=True)

hists = []; pt_clouds = []
num_hists = len(glob(os.path.join(data_dir, '*.npy'))) - 1
for i in range(num_hists):
    filename = f'iter_{i+1}.npy'  
    data = np.load(os.path.join(data_dir, filename), allow_pickle=True).item()

    hist = data['histogram'].reshape(-1, 128)
    pt_cloud = data['point_cloud']

    # === Compute point cloud based on ray direction and tof === #
    ray_dirs = pt_cloud / np.linalg.norm(pt_cloud, axis=-1, keepdims=True)
    tof = (np.argmax(hist, axis=-1) - t0) * bin_width
    distance_along_ray = 3E8 * tof / 2 # (N, )
    pt_cloud = ray_dirs * distance_along_ray.reshape(-1, 1)

    # === Fix point cloud === #
    # Coordinate frame after correction: 
    #       x-axis points to to right (left is -x) 
    #       y-axis points up
    #       z-axis points away from SPAD

    x = pt_cloud[:, 0].copy() 
    y = pt_cloud[:, 1].copy()
    pt_cloud[:, 0] = y
    pt_cloud[:, 1] = -x

    hists.append(hist)
    pt_clouds.append(pt_cloud)

# === Remove 1b signal === #
num_pixels = hists[0].shape[0]
hists_bp = []
for i in range(len(hists)):
    hist = hists[i]
    # === Remove 1b signal === #
    hist_1b_crop = np.zeros_like(hist)
    for j in range(num_pixels):
        peak_1b = np.argmax(hist[j, :])
        hist_crop = hist[j, peak_1b:]
        hist_1b_crop[j, :hist_crop.shape[0]] = hist_crop

    hist_1b_crop[:, :gates[0]] = 0
    hist_1b_crop[:, gates[1]:] = 0

    hists_bp.append(hist_1b_crop)

# === Compute camera position === #
num_x = 6
num_y = 6
x_range = [0, 128] # in cm
y_range = [32, 96] # in cm
x_pos = np.linspace(x_range[0], x_range[1], num_x) / 100
y_pos = np.linspace(y_range[0], y_range[1], num_y) / 100
x_pos, y_pos = np.meshgrid(x_pos, y_pos)

cam_position = np.zeros((num_x * num_y, 3))
cam_position[:, 0] = x_pos.flatten()
cam_position[:, 1] = y_pos.flatten()

cam_position = np.flip(cam_position, axis=0)
cam_position[:, 0] = 1.28 - cam_position[:, 0]

for i in range(num_y):
    if i % 2 == 1:
        cam_position[i * num_x:(i + 1) * num_x, :] = np.flip(cam_position[i * num_x:(i + 1) * num_x, :], axis=0)

cam_position = cam_position[:len(hists)]

# === Compute point cloud based on camera position === #
pt_clouds_bp = []
for i in range(len(pt_clouds)):
    pt_cloud_bp = pt_clouds[i] + cam_position[i, :].reshape(1, 3)
    pt_clouds_bp.append(pt_cloud_bp)

plot_hist_images(hists_bp , log_dir, norm='linear', lct=False)
plot_point_clouds(pt_clouds_bp, log_dir, 'point_clouds', f'point_clouds')

# === Define voxel grid === #
def create_voxel_grid(xlim, ylim, zlim, num_x, num_y, num_z):
    x = np.linspace(*xlim, num_x)
    y = np.linspace(*ylim, num_y)
    z = np.linspace(*zlim, num_z)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return np.vstack([X.reshape(-1, ), 
                    Y.reshape(-1, ), 
                    Z.reshape(-1, )]).T

xlim = [0, 1]
ylim = [0.3, 1.3]
# zlim = [-1.5, -0.55]
zlim = [-0.5, -0.2]

num_x = 40
num_y = 40
num_z = 20

voxel_params = [num_x, num_y, num_z]
voxel_grid = create_voxel_grid(xlim, ylim, zlim, num_x, num_y, num_z)

bp_params = {
    'pt_clouds': pt_clouds_bp,
    'hists': hists_bp,
    'voxel_grid': voxel_grid,
    'gates': gates,
    'bin_width': bin_width,
    'voxel_params': voxel_params,
}

vol = backprojection(**bp_params, return_indiv=False)

# phasor_reconst = []
# for hist, pt_cloud in tqdm(zip(hists_bp, pt_clouds_bp), desc="Computing phasor baseline", total=len(hists_bp)):
#     reconst = phasor_nlos(
#         hists=hist, 
#         pt_clouds=pt_cloud, 
#         bin_width=bin_width, 
#         voxel_grid=voxel_grid,
#         low_cutoff_Hz=10*(1/bin_width) / 104,
#         # high_cutoff_Hz= 12 * (1/bin_width) / 104,
#         eps=0.0,
#     )
#     phasor_reconst.append(reconst)

# vol = sum(phasor_reconst)
# vol = vol.reshape(num_x, num_y, num_z, order="C")

# === Plot backprojection === #
gamma = 2.3

# note: axes pointed so x is left, right is up, z is away from wall
plt.figure(figsize=(15, 5))

# x-y projection
plt.subplot(1, 3, 1)
plt.title('Front view (x-y)')
plt.imshow(np.flip(np.max(vol, axis=2).T, axis=(0, ))**gamma, 
            cmap='hot', 
            aspect='auto',
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_xaxis()


# x-z projection (z axis pointing down)
plt.subplot(1, 3, 2)
plt.title('Top view (x-z)')
plt.imshow(np.flip(np.max(vol, axis=1).T, axis=(0, ))**gamma, 
            cmap='hot', 
            aspect='auto',
            extent=[xlim[0], xlim[1], zlim[0], zlim[1]])

plt.xlabel('X (m)')
plt.ylabel('Z (m)')
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_xaxis()

# y-z projection (y axis pointing up)
plt.subplot(1, 3, 3)
plt.title('Side view (y-z)')
plt.imshow(np.flip(np.max(vol, axis=0), axis=(0, 1))**gamma, 
            cmap='hot', 
            aspect='auto',
            extent=[zlim[1], zlim[0], ylim[0], ylim[1]])

plt.xlabel('Z (m)')
plt.ylabel('Y (m)')
plt.gca().set_aspect('equal', adjustable='box')

# add spacing between subplots
plt.subplots_adjust(wspace=0.5)

plt.savefig(os.path.join(log_dir, f'backprojection.png'))
plt.close()


# # === Plot z slices of backprojection === #
# num_cols = 5
# num_rows = np.ceil(num_z / num_cols).astype(int)
# plt.figure(figsize=(15, 5))
# for i in range(num_z):
#     plt.subplot(num_rows, num_cols, i+1)
#     plt.imshow(np.flip(vol[i, :, :], axis=(0, )), cmap='hot', aspect='auto', extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
#     plt.xlabel('X (m)')
#     plt.ylabel('Y (m)')
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.gca().invert_xaxis()
# plt.savefig(os.path.join(log_dir, f'z_slices.png'))

# === Save volume === #
save_dict = {
    'vol': vol,
    'xlim': xlim,
    'ylim': ylim,
    'zlim': zlim,
    'num_x': num_x,
    'num_y': num_y,
    'num_z': num_z,
}

os.makedirs(log_dir, exist_ok=True)
np.savez(os.path.join(log_dir, f'volume.npz'), **save_dict)
        