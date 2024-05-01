import os, sys
import numpy as np 
import torch
import trimesh
import cv2

sys.path.append('../')
from utils.readfile import load_pkl
from utils import renderer
from smpl_pytorch.body_models import SMPL
from smplx_econ.pixielib.utils.geometry import rotation_matrix_to_angle_axis

def infer_smpl(pose, beta, smpl_server, return_joints=False):
    with torch.no_grad():
        output = smpl_server.forward_custom(betas=beta,
                                    body_pose=pose[:, 3:],
                                    global_orient=pose[:, :3],
                                    return_verts=True,
                                    return_full_pose=True,
                                    v_template=smpl_server.v_template, rectify_root=False)
    w = output.weights
    tfs = output.T
    verts = output.vertices
    pose_offsets = output.pose_offsets
    shape_offsets = output.shape_offsets
    joints = output.joints
    root_J = output.joints[:,[0]]

    if return_joints:
        return verts, joints
    else:
        return verts

def init_smpl_sever(gender='f'):
    smpl_server = SMPL(model_path='../smpl_pytorch',
                            gender=gender,
                            use_hands=False,
                            use_feet_keypoints=False,
                            dtype=torch.float32).cuda()
    return smpl_server

def get_scale_trans(points_src, points_tgt):
    points_src_stack = points_src.reshape(-1)
    stack0 = np.zeros((len(points_src_stack)))
    stack1 = np.zeros((len(points_src_stack)))
    stack0[::2] = 1
    stack1[1::2] = 1
    points_src_stack = np.stack((points_src_stack, stack0, stack1), axis=-1)
    points_tgt_stack = points_tgt.reshape(-1, 1)

    x = np.linalg.inv(points_src_stack.T@points_src_stack)@(points_src_stack.T)@points_tgt_stack
    warp_mat = np.float32([[x[0],0,x[1]],[0,x[0],x[2]]])
    return warp_mat

def align_image(image, warp_mat, size=256):
    image_new = np.zeros((size, size, 3)).astype(np.uint8)
    rows, cols, ch = image_new.shape
    image_new = cv2.warpAffine(image, warp_mat, (cols, rows))
    return image_new

#######################################
# shirt: 60
# jacket: 120
# trousers: 180
# skirt: 240
#######################################
target_label = 180
image_dir = '/fitting-data/garment/images'
econ_dir = './fitting-data/garment/processed/econ'
seg_dir = './fitting-data/garment/processed/segmentation'
smpl_dir = './fitting-data/garment/processed/bodys/smpl'
output_dir = './fitting-data/garment/processed/align'

images = os.listdir(image_dir)

smpl_server = init_smpl_sever()
transform = renderer.get_transform(scale=80)
transform_100 = renderer.get_transform(scale=100)

for i in range(len(images)):
    if not os.path.isfile(os.path.join(econ_dir, 'vid/%s_in_tensor.pt'%images[i].split('.')[0])):
        continue
    vid = torch.load(os.path.join(econ_dir, 'vid/%s_in_tensor.pt'%images[i].split('.')[0]))
    normal_econ = vid['normal_F'][0].permute(1,2,0).cpu().numpy()
    normal_econ = ((normal_econ*0.5+0.5)*255).astype(np.uint8)
    smpl_np = np.load(os.path.join(econ_dir, 'obj/%s_smpl_00.npy'%images[i].split('.')[0]), allow_pickle=True)[()]
    mask_sam = cv2.imread(os.path.join(seg_dir, '%s-sam-labeled.png'%images[i].split('.')[0]))
    mask_sam = mask_sam == target_label

    smpl_data = load_pkl(os.path.join(smpl_dir, '%s_smplx.pkl'%images[i].split('.')[0]))
    pose_smpl = smpl_data['full_pose'].reshape(24,3,3)
    pose_smpl = rotation_matrix_to_angle_axis(pose_smpl).unsqueeze(0).cuda().reshape(1, 72).detach()
    beta_smpl = smpl_data['betas'].cuda().detach()

    verts, joints = infer_smpl(pose_smpl, beta_smpl, smpl_server, return_joints=True)
    root_joint = joints[0, 0].detach().cpu().numpy()
    body_smpl = trimesh.Trimesh(verts.detach().squeeze().cpu().numpy(), smpl_server.faces)
    body_smpl.vertices -= root_joint

    body_align_target = trimesh.load(os.path.join(econ_dir, 'obj/%s_smpl_00.obj'%images[i].split('.')[0]))

    v_body_align_target = body_align_target.vertices
    v_body_smpl = body_smpl.vertices
    a = v_body_align_target.max(axis=0) - v_body_align_target.min(axis=0)
    b = v_body_smpl.max(axis=0) - v_body_smpl.min(axis=0)
    scale = a/b
    v_body_smpl *= scale[None]
    center_body_align_target = (v_body_align_target.max(axis=0) + v_body_align_target.min(axis=0))/2
    center_body_smpl = (v_body_smpl.max(axis=0) + v_body_smpl.min(axis=0))/2 
    trans = center_body_align_target - center_body_smpl
    trans = torch.FloatTensor(trans).cuda().unsqueeze(0)
    scale = torch.FloatTensor(scale).cuda().unsqueeze(0)

    joints = joints - joints[:,[0]]
    joints_adjust = joints*scale[None] + trans[None]
    joints_adjust_2d = transform_100.transform_points(joints_adjust)
    joints_2d = transform_80.transform_points(joints)
    joints_adjust_2d = joints_adjust_2d[0,:,:2].cpu().numpy()*(-255)+256
    joints_2d = joints_2d[0,:,:2].cpu().numpy()*(-128)+128

    warp_mat = get_scale_trans(joints_adjust_2d, joints_2d)

    normal_econ = normal_econ*mask_sam
    normal_econ_align = align_image(normal_econ.copy(), warp_mat)
    cv2.imwrite(os.path.join(output_dir, '%s_normal_align.png'%images[i].split('.')[0]), normal_econ_align)

    np.savez(os.path.join(output_dir, '%s_bni_v2'%images[i].split('.')[0]), 
                          trans=trans.cpu().numpy(), scale=scale.cpu().numpy(),
                          pose=pose_smpl.cpu().numpy(), beta=beta_smpl.cpu().numpy(), )
    #sys.exit()

