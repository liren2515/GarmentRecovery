# Installation

## Setup:
 - Download checkpoints/extra-data/sample-data from [here](https://drive.google.com/file/d/1hkhW7RGmlDviH2bZ4P5s5b-h4lqkrsNJ/view?usp=sharing), and unzip it to the root path of the repo.

 - Download and extract the SMPL model from http://smplify.is.tue.mpg.de/, and place `basicModel_f_lbs_10_207_0_v1.0.0.pkl` and `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` in the folder of `./smpl_pytorch`.

## Installing

- The basic installation
  ```
  conda create --name py38 python=3.8 -y
  conda activate py38
  conda install pytorch==2.0.1 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y && conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y && conda install -c bottler nvidiacub -y &&  conda install pytorch3d -c pytorch3d -y

  # Install pip dependencies
  pip install numpy==1.23 chumpy opencv-python torchgeometry rtree plyfile cycler einops kornia mediapipe
  pip install hydra-core matplotlib scikit-image pyglet==1.5.7 shapely

  ```

- Install Pymesh
  - Following the [instruction](https://pymesh.readthedocs.io/en/latest/installation.html) to install Pymesh.


## Folder hierarchy
- Once you sucessfully installed and downloaded all, you should have the following files in your directory:
    ```
    .
    └── checkpoints
    └── data_prepare
    └── doc
    └── figs
    └── fit
    ├── fitting-data
    │   └── jacket
    │   └── shirt
    |   └── skirt
    │   └── trousers
    ├── networks    
    └── smpl_pytorch
    └── smplx_econ
    └── snug
    └── utils
    ```
