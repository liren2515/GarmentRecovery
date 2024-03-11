# Installation

## Setup:
Download checkpoints from [here](https://drive.google.com/file/d/1Zhr93ejWGobqDnJjE-P95ssNTDYSFNXS/view?usp=sharing), and put `*.pth` at `./checkpoints`.

Download and extract the SMPL model from http://smplify.is.tue.mpg.de/, and place `basicModel_f_lbs_10_207_0_v1.0.0.pkl` and `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` in the folder of `./smpl_pytorch`.

## Installing All Modules

- The basic installation
  ```
  conda create --name py38 python=3.8 -y
  conda activate py38
  conda install pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y && conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y && conda install -c bottler nvidiacub -y && conda install -c conda-forge igl -y && conda install pytorch3d -c pytorch3d -y

  # Install pip dependencies
  pip install numpy==1.23 chumpy opencv-python torchgeometry rtree h5py plyfile tensorboardX cycler einops kornia mediapipe
  pip install hydra-core matplotlib scikit-image pyglet==1.5.7 shapely open3d

  ```

- Install Pymesh
  - This is required for hand motion capture. You can skip this if you need only body module
  - If you followed the versions mentioned above (pytorch 1.6.0, CUDA 10.1, on Linux), you may try the following:
  ```
    python -m pip install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
  ```
  - If it doesn't work, follow the instruction of [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)
  
- Install pytorch3d (optional, for pytorch3d renderering)
  - We use pytorch3d for an alternative rendering option. We provide other options (opengl by default) and you may skip this process.
  - You may try the following (pytorch 1.6.0, on Linux and Mac).
    ```
    pip install pytorch3d
    ```
  - If it doesn't work, follow the instruction of [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)

- Install other third-party libraries + download pretrained models and sample data
  - Run the following script
  ```
  sh scripts/install_frankmocap.sh
  ```

- Setting SMPL/SMPL-X Models
    - We use SMPL and SMPL-X model as 3D pose estimation output. You have to download them from the original website.
    - Download SMPL Model (Neutral model: basicModel_neutral_lbs_10_207_0_v1.0.0.pkl):    
        - Download in the original [website](http://smplify.is.tue.mpg.de/login). You need to register to download the SMPL data.
        - Put the file in: ./extra_data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
        - This is only for body module

    - Download SMPLX Model (Neutral model: SMPLX_NEUTRAL.pkl):
        - You can use SMPL-X model for body mocap instead of SMPL model. 
        - Download ```SMPLX_NEUTRAL.pkl``` in the original [SMPL website](https://smpl-x.is.tue.mpg.de/). You need to register to download the SMPLX data.
        - Put the ```SMPLX_NEUTRAL.pkl`` file in: ./extra_data/smpl/SMPLX_NEUTRAL.pkl
        - This is for hand module and whole body module

## Folder hierarchy
- Once you sucessfully installed and downloaded all, you should have the following files in your directory:
    ```
    ./extra_data/
    ├── hand_module
    │   └── mean_mano_params.pkl
    │   └── SMPLX_HAND_INFO.pkl
    |   └── pretrained_weights
    |   |   └── pose_shape_best.pth
    │   └── hand_detector
    │       └── faster_rcnn_1_8_132028.pth  
    │       └── model_0529999.pth
    ├── body_module
    |   └──body_pose_estimator
    |       └── checkpoint_iter_370000.pth     
    └── smpl
        └── basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
        └── SMPLX_NEUTRAL.pkl
        
    ./detectors/
    ├── body_pose_estimator
    ├── hand_object_detector
    └── hand_only_detector
    ```

## Installing Body Module Only

- The basic installation
    ```
    conda create -n venv_frankmocap python=3.7
    conda activate venv_frankmocap

    # Install cuda 
    # Choose versions based on your system. For example:
    # conda install cudatoolkit=10.1 cudnn=7.6.0

    # Install pytorch and torchvision 
    conda install -c pytorch pytorch==1.6.0 torchvision cudatoolkit=10.1

    # Install other required libraries
    pip install -r docs/requirements.txt
    ```

- Install pytorch3d (optional, for pytorch3d renderering)
    - We use pytorch3d for an alternative rendering option. We provide other options (opengl by default) and you may skip this process.
    - You may try the following (pytorch 1.6.0, on Linux and Mac).
        ```
        pip install pytorch3d
        ```
    - If it doesn't work, follow the instruction of [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)

- Install 2D pose detector and download pretrained models and sample data
    - Install [2D keypoint detector](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch): 
    ```
    sh scripts/install_pose2d.sh
    ```
    - Download pretrained model and other extra data
    ```
    sh scripts/download_data_body_module.sh
    ```
    - Download sample data
    ```
    sh scripts/download_sample_data.sh
    ```
- Setting SMPL/SMPL-X Models
    - You only need SMPL model. See above
