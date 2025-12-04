# Imaging Hidden Objects with Consumer LiDAR



### Install dependencies
After cloning the repository, install all dependencies by running the following text block.

```cd consumer-nlos
conda create --name consumer-nlos python=3.11
conda activate consumer-nlos
pip install -r requirements.txt
```


### Dataset
We provide sample data in the folder captured data/st_spad_person_tracking to test this code on. This data was collected with a ST VL853L8 device. To use your own data, add your own implementation of the `load_data` function in `data/dataloader.py` and update `main.py` as needed. 



### Running the script


The script is performing tracking using the proposed particle filtering formulation with the motion-induced aperture sampling (MAS) model. The parameters of the particle filtering algorithm and the dataset are specified in `config.yaml`. To run the code, execute the following python command. 

```
python tracking.py
```

or 

```
python cam_localization.py
```

The outputs of this algorithms will be stored in `/results/`. 




