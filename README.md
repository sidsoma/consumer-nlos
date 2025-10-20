# Imaging Hidden Objects with Consumer LiDAR


To run the tracking algorithm on data captured with a ST VL853L8 device, run the following python command. 

python main.py

The captured data is in the directory captured_data/st_spad_person_tracking. The script is performing tracking 
using the proposed particle filtering formulation with the motion-induced aperture sampling (MAS) model. The parameters of the particle filtering algorithm and the dataset are specified in config.yaml. 


The outputs of this algorithms will be stored in /results/tracking_result.mp4. The individual frames are also available
in /results/reconstructed_frames. This output should match the result shown in Fig. 3 of the Supplementary Material and the 
"Plug-and-Play NLOS" result shown in the supplementary video.

While the particle filtering code is fast, the code to compute and plot the kernal density estimation (KDE) of the particles is slow, 
therefore we just plot the particles in the position. 


If you wish to visualize the code with the KDE (similar to the result visualization), go to config.yaml and toggle "plot_KDE = True". 



