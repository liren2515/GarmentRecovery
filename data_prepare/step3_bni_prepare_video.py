import os, sys
import numpy as np 
import torch
import trimesh
import random
import cv2
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

sys.path.append('../')
from utils.readfile import load_pkl
from utils.mesh import apply_rotation, rotate_pose
from utils.render_normal import get_render
from utils.rasterize import get_raster, get_pix_to_face
from smpl_pytorch.body_models import SMPL
from econ.pixielib.utils.geometry import rotation_matrix_to_angle_axis

def collect_projected_vertices(mesh, transform, silh, res=512):
    vertices = torch.FloatTensor(mesh.vertices).cuda().unsqueeze(0)
    vertices_2d = transform.transform_points(vertices)
    vertices_2d = torch.round(vertices_2d[0,:,:2]*(-255.5) + 255.5).cpu().numpy().astype(int)
    keep = silh[vertices_2d[:,1], vertices_2d[:,0], 0] == 1
    
    vertices_keep = mesh.vertices[keep]
    
    C_g = np.array([[0,0,0]]*mesh.vertices.shape[0], np.uint8)
    C_g[keep] = 255
    mesh.visual.vertex_colors = C_g
    #mesh.export('collect_projected_vertices.obj')
    
    return vertices_keep, mesh


def get_pix_to_face(verts, faces, raster, silh):
    mesh_py3d = Meshes(
            verts=[verts],   
            faces=[faces],
            textures=TexturesVertex(verts_features=torch.ones_like(verts[None]))
        )

    #image = renderer(mesh_py3d)[0,:,:,:3]
    #cv2.imwrite('../tmp/image.png', (image*255).detach().cpu().numpy().astype(np.uint8))
    #sys.exit()
    Fragments = raster(mesh_py3d)
    pix_to_face = Fragments.pix_to_face.squeeze()
    #pix_to_face[::2,::2] = -1
    #silh[::2,::2] = 0
    valid_pixel = torch.logical_and(pix_to_face > -1, silh == 1)

    valid_faces = pix_to_face[valid_pixel]

    valid_faces = torch.unique(valid_faces.flatten())
    valid_vertices = torch.unique(faces[valid_faces].flatten())
    #print(len(valid_vertices),pix_to_face.max(), silh.max())
    return valid_faces, valid_vertices

def collect_projected_vertices_v2(mesh, raster, silh, res=512):
    vertices = torch.FloatTensor(mesh.vertices).cuda()
    faces = torch.LongTensor(mesh.faces).cuda()
    _, idx_v = get_pix_to_face(vertices, faces, raster, silh)
    idx_v = idx_v.detach().cpu().numpy()

    vertices_keep = mesh.vertices[idx_v]
    
    C_g = np.array([[0,0,0]]*mesh.vertices.shape[0], np.uint8)
    C_g[idx_v] = 255
    mesh.visual.vertex_colors = C_g
    #mesh.export('collect_projected_vertices.obj')
    
    return vertices_keep, mesh
'''
def split_front_back(mesh):
    meshes = mesh.split(only_watertight=False)
    z_max = -100
    keep = -1
    for i in range(len(meshes)):
        if meshes[i].vertices[:, -1].mean() > z_max:
            z_max = meshes[i].vertices[:, -1].mean()
            keep = i
    if keep == 0:
        i_back = 1
    else:
        i_back = 0

    return meshes[keep], meshes[i_back]
'''
def split_front_back(mesh):
    meshes = mesh.split(only_watertight=False)
    z_max = -100
    keep = -1

    num_v = []
    for mesh in meshes:
        num_v.append(len(mesh.vertices))
    idx = sorted(range(len(num_v)), key=lambda k: num_v[k])[::-1][:2]
    meshes = meshes[idx]

    for i in range(len(meshes)):
        if meshes[i].vertices[:, -1].mean() > z_max:
            z_max = meshes[i].vertices[:, -1].mean()
            keep = i
    if keep == 0:
        i_back = 1
    else:
        i_back = 0

    return meshes[keep], meshes[i_back]

def infer_smpl(pose, beta, smpl_server, return_joints=False):
    with torch.no_grad():
        output = smpl_server.forward_custom(betas=beta,
                                    #transl=transl,
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
    smpl_server = SMPL(model_path='/scratch/cvlab/home/ren/code/snug-pytorch/smpl_pytorch',
                            gender=gender,
                            #batch_size=1,
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

target_label = 255 # skirt
image_dir = '/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video/images/'
econ_dir = '/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video/processed/econ'
seg_dir = '/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video/processed/segmentation'
smpl_dir = '/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video/processed/bodys/smpl'
output_dir = '/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video/processed/align'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

images = os.listdir(image_dir)

smpl_server = init_smpl_sever()
_, _, transform_80 = get_render(render_res=512, scale=80, return_transform=True, sigma=1e-7)
_, _, transform_100 = get_render(render_res=512, scale=100, return_transform=True, sigma=1e-7)

raster = get_raster(render_res=512, scale=100, faces_per_pixel=1)

for i in range(len(images)):
    print(i)
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

    mesh_bni = trimesh.load(os.path.join(econ_dir, 'obj/%s_0_BNI.obj'%images[i].split('.')[0]))
    mesh_full = trimesh.load(os.path.join(econ_dir, 'obj/%s_0_full.obj'%images[i].split('.')[0]))
    vertices_keep, mesh = collect_projected_vertices_v2(mesh_bni, raster, torch.from_numpy(mask_sam[:,:,0]).cuda(), res=512)
    vertices_all = mesh_full.vertices

    mesh.export(os.path.join(output_dir, '%s_collect_projected_vertices_v2.obj'%images[i].split('.')[0]))
    np.savez(os.path.join(output_dir, '%s_bni_v2'%images[i].split('.')[0]), 
                          vertices_keep=vertices_keep, vertices_all=vertices_all,
                          trans=trans.cpu().numpy(), scale=scale.cpu().numpy(),
                          pose=pose_smpl.cpu().numpy(), beta=beta_smpl.cpu().numpy(), )
    #sys.exit()

