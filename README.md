# GraspDiffusion_Comparison
A comparison of different grasp generation models based on diffusion

# Requirements
1) Ubuntu 20.04
2) Cuda Toolkit 11.8

# Setup
### Install Conda Environment
conda env create -f evaluation.yml

### Install right pytorch
first uninstall the available torch version using <br />
pip uninstall torch torchvision torchaudio <br />
then use the following line to install the required version <br />
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 <br />
NOTE: IT IS IMPORTANT THAT CUDA COMPATIBLE TORCH IS INSTALLED OTHERWISE ISAACGYM WONT WORK

### Run the setup script
pip install -e.

### Install mesh_to_sdf 
git clone https://github.com/robotgradient/mesh_to_sdf.git <br />
cd mesh_to_sdf && pip install -e.

### Install isaac gym
https://developer.nvidia.com/isaac-gym <br />
cd isaacgym <br />
cd python <br />
pip install -e.

### Download Data
mkdir data <br />
cd data <br />
https://drive.google.com/drive/folders/1ULWuYZYyFncIBqBhRMNrVOrosGGRITZU

# Errors that may occur and how to solve them
### Failed to build scikit-sparse
use the following command: <br />
conda install -c conda-forge scikit-sparse <br />
then use the following command to install the rest of the dependencies: <br />
pip install -r requirements.txt
