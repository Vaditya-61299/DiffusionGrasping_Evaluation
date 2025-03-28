# GraspDiffusion_Comparison
A comparison of different grasp generation models based on diffusion

# Requirements
1) Ubuntu 20.04
2) Cuda Toolkit 11.8

# Setup
### Install Conda Environment

```
conda env create -f evaluation.yml
```

### Install right pytorch
first uninstall the available torch version using 
```
pip uninstall torch torchvision torchaudio 
```
then use the following line to install the required version 
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 
```
NOTE: IT IS IMPORTANT THAT CUDA COMPATIBLE TORCH IS INSTALLED OTHERWISE ISAACGYM WONT WORK

### Run the setup script
```
pip install -e.
```

### Install mesh_to_sdf 
```
git clone https://github.com/robotgradient/mesh_to_sdf.git 
cd mesh_to_sdf && pip install -e.
```

### Install isaac gym
use the following link to download the isaac gym
```
https://developer.nvidia.com/isaac-gym
```

Follow the follwoing commands to install the package in the environment
```
cd isaacgym 
cd python 
pip install -e.
```
### Download Data
```
mkdir data 
cd data 
mkdir splits 
```
Download the data from the following link
```
https://drive.google.com/drive/folders/1ULWuYZYyFncIBqBhRMNrVOrosGGRITZU
```
to prepare evaluation of graspLDM, run new_splits.py file in utils. test split will be generated in the new_splits folder. copy these in /data/splits

### Download models
Prepare DiffusionField models:
```
cd data
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/camusean/grasp_diffusion
```
After cloning the link above, rename the "grasp_diffusion" folder to "models".
Then, use the following commands
```
cd models
mkdir cgdf_v1
```
In the cgdf_v1 folder, download the model path from the following link
```
https://drive.google.com/drive/folders/1A9oGhv-gNkyMjSOfjKA4p6NcLVelIA-b?usp=sharing
```
### Install dependencies
```
pip install pykdtree
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```
# Structure of the directory after the Setup
```
DiffusionGrasping_Evaluation (repository) 
├── isaacgym 
├── mesh_to_sdf 
├── data 
│   ├── grasps 
│   │   ├── meshes 
│   │   ├── sdf 
│   │   ├── splits 
│   │   ├── models 
│   │   │   ├── model_1 
│   │   │   ├── model_2 
│   │   │   ├── model_3 
├── (other files of the repository)
```

# Instructions on getting the results
1) Open the Evaluate_models.py script.
2) Choose the model that you want to check. MAKE SURE TO COMMENT OUT THE OTHER MODELS
3) After running, the result will be saved in the LOGS folder.
4) Run utils/corrector.py to make result file work.
5) Run utils/results_extractor.py to check the results.
# Errors that may occur and how to solve them
### Failed to build scikit-sparse
use the following command: 
```
conda install -c conda-forge scikit-sparse 
```
then use the following command to install the rest of the dependencies: 
```
pip install -r requirements.txt
```

# Acknowledgment

This repository is heavily based on SE3-DiffusionFields(https://github.com/robotgradient/grasp_diffusion) , CGDF(https://github.com/constrained-grasp-diffusion/constrained-grasp-diffusion) and GraspLDM(https://github.com/kuldeepbrd1/graspLDM). The datset used is ACRONYM(https://github.com/NVlabs/acronym)
