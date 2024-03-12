##########################
# step 1
##########################
# infer smpl-x body mesh
import numpy as np
import torch
import trimesh
import os, sys
sys.path.append('../')
from utils.mesh import rotate_pose
from econ.pixielib.utils.config import cfg as pixie_cfg
from econ.pixielib.models.SMPLX import SMPLX as PIXIE_SMPLX
from econ.pixielib.utils.geometry import batch_rodrigues

image_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/open-shirt/images/'
econ_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/open-shirt/processed/econ'
output_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/open-shirt/processed/bodys/smplx'
image_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/skirt/images/'
econ_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/skirt/processed/econ'
output_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/skirt/processed/bodys/smplx'

image_dir = '/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video/images/'
econ_dir = '/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video/processed/econ'
output_dir = '/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video/processed/bodys/smplx'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
images = os.listdir(image_dir)

smpl_model = PIXIE_SMPLX(pixie_cfg.model).cuda()
for i in range(len(images)):

    smpl_np = np.load(os.path.join(econ_dir, 'obj/%s_smpl_00.npy'%images[i].split('.')[0]), allow_pickle=True)[()]

    shape_params = smpl_np['betas']
    expression_params = smpl_np['expression']
    body_pose = smpl_np['body_pose']
    global_pose = smpl_np['global_orient']
    jaw_pose = smpl_np['jaw_pose']
    left_hand_pose = smpl_np['left_hand_pose']
    right_hand_pose = smpl_np['right_hand_pose']

    global_pose = rotate_pose(global_pose.view(-1, 3), np.pi, which_axis='x').reshape(-1)

    N_body, N_pose = body_pose.shape[:2]
    N_body, N_hand_pose = left_hand_pose.shape[:2]
    orient_mat = batch_rodrigues(global_pose.view(-1, 3)).view(N_body, 1, 3, 3)
    pose_mat = batch_rodrigues(body_pose.view(-1,  3)).view(N_body, N_pose, 3, 3)
    left_hand_pose = batch_rodrigues(left_hand_pose.view(-1,  3)).view(N_body, N_hand_pose, 3, 3)
    right_hand_pose = batch_rodrigues(right_hand_pose.view(-1,  3)).view(N_body, N_hand_pose, 3, 3)
    jaw_pose = batch_rodrigues(jaw_pose.view(-1,  3)).view(1, -1, 3, 3)

    verts, _, joints = smpl_model(
                    shape_params=shape_params.cuda(),
                    expression_params=expression_params.cuda(),
                    body_pose=pose_mat.cuda(),
                    global_pose=orient_mat.cuda(),
                    jaw_pose=jaw_pose.cuda(),
                    left_hand_pose=left_hand_pose.cuda(),
                    right_hand_pose=right_hand_pose.cuda(),
                )

    faces_tensor = smpl_model.faces_tensor
    root_joint = joints[0, 0].detach().cpu().numpy()
    body = trimesh.Trimesh(verts.detach().squeeze().cpu().numpy(), faces_tensor.squeeze().detach().cpu().numpy())

    '''
    body.vertices -= root_joint
    body = apply_rotation(np.pi, body, 'x')
    body.vertices += root_joint
    '''

    body.export(os.path.join(output_dir, '%s_smplx.obj'%images[i].split('.')[0]))

##########################
# step 2
##########################
# run convertor: smpl-x to smpl
'''
# Use default pytorch3d conda
# use ic-registry.epfl.ch/cvlab/lis/lab-python-ml:cuda11
# update 'data_folder' and 'output_folder' in /cvlabdata2/home/ren/smpl-x/smplx/config_files/smplx2smpl.yaml 
pip install typing-extensions --upgrade 
pip install markupsafe jinja2 loguru omegaconf
pip install -U scikit-learn
pip install open3d chumpy trimesh
cd /cvlabdata2/home/ren/smpl-x/torch-trust-ncg
python setup.py install --user
cd ../smplx
sudo apt-get update
sudo apt-get install libopenblas-dev libx11-6 libgl1
python -m transfer_model --exp-cfg config_files/smplx2smpl.yaml
'''