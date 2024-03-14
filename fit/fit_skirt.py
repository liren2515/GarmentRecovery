import os,sys
import numpy as np 
import torch
import trimesh
import time
import random
import cv2
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from functools import reduce

sys.path.append('../')
from utils import mesh_reader, renderer
from utils.init_utils import init_uv_mesh, init_smpl_sever, load_model
from utils.img_utils import prepare_input
from utils.mesh_utils import concatenate_mesh, flip_mesh, repair_pattern, barycentric_faces, project_waist
from utils.isp_cut import select_boundary, connect_2_way, one_ring_neighour, create_uv_mesh, get_connected_paths_skirt
from utils.smpl_utils import infer_smpl, skinning_diffuse
from utils.fit_utils import infer_model, reconstruct_pattern_with_label, rescale_cloth
from smpl_pytorch.body_models import SMPL
from utils.optimization_skirt import optimize_lat_code, optimize_prior, optimize_vertices


from snug.snug_class import Body, Cloth_from_NP, Material
try:
    import pymesh
except:
    pass

def sewing_front_back(mesh_pattern_f, mesh_pattern_b, mesh_atlas_f, mesh_atlas_b, labels_f, labels_b, z_offset=0.001):

    idx_boundary_v_f, boundary_edges_f = select_boundary(mesh_pattern_f)
    idx_boundary_v_b, boundary_edges_b = select_boundary(mesh_pattern_b)
    boundary_edges_f = set([tuple(sorted(e)) for e in boundary_edges_f.tolist()])
    boundary_edges_b = set([tuple(sorted(e)) for e in boundary_edges_b.tolist()])
    label_boundary_v_f = labels_f[idx_boundary_v_f]
    label_boundary_v_b = labels_b[idx_boundary_v_b]

    indicator_seam1_f = label_boundary_v_f == 1
    indicator_seam1_b = label_boundary_v_b == 1
    indicator_seam2_f = label_boundary_v_f == 2
    indicator_seam2_b = label_boundary_v_b == 2

    idx_seam1_v_f = idx_boundary_v_f[indicator_seam1_f]
    idx_seam1_v_b = idx_boundary_v_b[indicator_seam1_b]
    idx_seam2_v_f = idx_boundary_v_f[indicator_seam2_f]
    idx_seam2_v_b = idx_boundary_v_b[indicator_seam2_b]
    
    one_rings_seam1_f = one_ring_neighour(idx_seam1_v_f, mesh_pattern_f, is_dic=True, mask_set=set(idx_seam1_v_f))
    one_rings_seam1_b = one_ring_neighour(idx_seam1_v_b, mesh_pattern_b, is_dic=True, mask_set=set(idx_seam1_v_b))
    one_rings_seam2_f = one_ring_neighour(idx_seam2_v_f, mesh_pattern_f, is_dic=True, mask_set=set(idx_seam2_v_f))
    one_rings_seam2_b = one_ring_neighour(idx_seam2_v_b, mesh_pattern_b, is_dic=True, mask_set=set(idx_seam2_v_b))

    path_seam1_f, _ = connect_2_way(set(idx_seam1_v_f), one_rings_seam1_f, boundary_edges_f)
    path_seam1_b, _ = connect_2_way(set(idx_seam1_v_b), one_rings_seam1_b, boundary_edges_b)
    path_seam2_f, _ = connect_2_way(set(idx_seam2_v_f), one_rings_seam2_f, boundary_edges_f)
    path_seam2_b, _ = connect_2_way(set(idx_seam2_v_b), one_rings_seam2_b, boundary_edges_b)

    if mesh_pattern_f.vertices[path_seam2_f[0], 1] < mesh_pattern_f.vertices[path_seam2_f[-1], 1]: # high to low
        path_seam2_f = path_seam2_f[::-1]
    if mesh_pattern_b.vertices[path_seam2_b[0], 1] < mesh_pattern_b.vertices[path_seam2_b[-1], 1]:
        path_seam2_b = path_seam2_b[::-1]

    if mesh_pattern_f.vertices[path_seam1_f[0], 1] < mesh_pattern_f.vertices[path_seam1_f[-1], 1]:
        path_seam1_f = path_seam1_f[::-1]
    if mesh_pattern_b.vertices[path_seam1_b[0], 1] < mesh_pattern_b.vertices[path_seam1_b[-1], 1]:
        path_seam1_b = path_seam1_b[::-1]

    idx_offset = len(mesh_pattern_f.vertices)

    faces_seam1 = mesh_reader.triangulation_seam_v2(mesh_atlas_f, mesh_atlas_b, path_seam1_f, path_seam1_b, idx_offset, reverse=False)
    faces_seam2 = mesh_reader.triangulation_seam_v2(mesh_atlas_f, mesh_atlas_b, path_seam2_f, path_seam2_b, idx_offset, reverse=False)

    faces_sewing = [mesh_atlas_f.faces, mesh_atlas_b.faces + idx_offset, faces_seam1, faces_seam2[:,::-1]]
    #sys.exit()

    mesh_atlas_f.vertices[:, -1] += z_offset
    verts_sewing = np.concatenate((mesh_atlas_f.vertices, mesh_atlas_b.vertices), axis=0)
    faces_sewing = np.concatenate(faces_sewing, axis=0)
    mesh_sewing = trimesh.Trimesh(verts_sewing, faces_sewing, validate=False, process=False)

    labels_sewing = np.concatenate((labels_f, labels_b), axis=0)

    return mesh_sewing, labels_sewing


def reconstruct_v3(cnn_regressor, model_isp, latent_codes, images, pose, beta, uv, Rot_rest, pose_offsets_rest, verts_uv_cano_f_mean, verts_uv_cano_b_mean, weight_f, weight_b, smpl_server, trans, scale, save_path):
    #######################################################
    ''' step 1: infer '''
    #######################################################

    with torch.no_grad():
        #images, images_body = images
        #extractor, extractorBody, featuror, field, diffusion_skin = cnn_regressor
        mesh_uv, uv_vertices, uv_faces, edges = uv

        verts_pose_f, verts_pose_b, v_indicator_f, v_indicator_b = infer_model(images, cnn_regressor, uv_vertices)
        
        verts_pose_f_infer = verts_pose_f.cpu().numpy()
        verts_pose_b_infer = verts_pose_b.cpu().numpy()

        '''
        verts_pose_f_pred = verts_pose_f[v_indicator_f]
        verts_pose_b_pred = verts_pose_b[v_indicator_b]
        verts_pose = torch.cat((verts_pose_f_pred, verts_pose_b_pred), dim=0).cpu().numpy()*scale.cpu().numpy() + trans.cpu().numpy()
        verts_pred_pc = trimesh.PointCloud(verts_pose)
        verts_pred_pc.export(os.path.join(save_path, 'verts_pred_pc.obj'))
        mesh_uv_f = trimesh.Trimesh(uv_vertices.cpu().numpy(), uv_faces, validate=False, process=False)
        mesh_uv_b = trimesh.Trimesh(uv_vertices.cpu().numpy(), uv_faces, validate=False, process=False)
        C_g_f = np.array([[0,0,0]]*mesh_uv_f.vertices.shape[0], np.uint8)
        C_g_b = np.array([[0,0,0]]*mesh_uv_f.vertices.shape[0], np.uint8)
        C_g_f[v_indicator_f.cpu().numpy()] = 255
        C_g_b[v_indicator_b.cpu().numpy()] = 255
        mesh_uv_f.visual.vertex_colors = C_g_f
        mesh_uv_b.visual.vertex_colors = C_g_b
        mesh_uv_f.export(os.path.join(save_path, 'mesh_uv_f_infer.obj'))
        mesh_uv_b.export(os.path.join(save_path, 'mesh_uv_b_infer.obj'))
        '''
    #######################################################
    ''' step 2: optimize latent code with ISP '''
    #######################################################
    lat_code_opt, v_indicator_f_target, v_indicator_b_target = optimize_lat_code(model_isp, latent_codes, uv_vertices, uv_faces, v_indicator_f, v_indicator_b, iters=1000)
    
    with torch.no_grad():
        uv_vertices_256, uv_faces_256 = create_uv_mesh(256, 256, debug=False)
        mesh_uv_256 = trimesh.Trimesh(uv_vertices_256, uv_faces_256, validate=False, process=False)
        uv_vertices_256 = torch.FloatTensor(uv_vertices_256).cuda()
        edges_256 = torch.LongTensor(mesh_uv_256.edges).cuda()

        mesh_atlas_f, mesh_atlas_b, mesh_pattern_f, mesh_pattern_b, label_f, label_b = reconstruct_pattern_with_label(model_isp, lat_code_opt, uv_vertices_256, uv_faces_256, edges_256, using_repair=True)
        num_v_f = len(mesh_atlas_f.vertices)
        num_f_f = len(mesh_atlas_f.faces)
        
        z_offset = 0.005
        mesh_atlas_f.vertices[:, -1] += z_offset
        #mesh_atlas = concatenate_mesh(mesh_atlas_f, mesh_atlas_b)
        mesh_pattern_f.export(os.path.join(save_path, 'mesh_pattern_f.obj'))
        mesh_pattern_b.export(os.path.join(save_path, 'mesh_pattern_b.obj'))
        mesh_atlas_f.export(os.path.join(save_path, 'mesh_atlas_f.obj'))
        mesh_atlas_b.export(os.path.join(save_path, 'mesh_atlas_b.obj'))

        mesh_atlas_sewing, labels = sewing_front_back(mesh_pattern_f, mesh_pattern_b, mesh_atlas_f, mesh_atlas_b, label_f, label_b, z_offset=z_offset)
        mesh_atlas_sewing.export(os.path.join(save_path, 'mesh_atlas_sewing.obj'))


    #######################################################
    ''' step 3: optimize deformation prior '''
    #######################################################
    idx_boundary_v, boundary_edges = select_boundary(mesh_atlas_sewing)
    path_waist = get_connected_paths_skirt(mesh_atlas_sewing, idx_boundary_v, boundary_edges)[0]
    waist_edges = np.array([path_waist[:-1], path_waist[1:]])

    # make it a waistband
    idx_waist_v_propogate = np.unique(waist_edges.flatten()).tolist()
    for _iter in range(3):
        idx_waist_v_propogate = one_ring_neighour(idx_waist_v_propogate, mesh_atlas_sewing, is_dic=False, mask_set=None)
        idx_waist_v_propogate = set(reduce(lambda a,b: list(a)+list(b), idx_waist_v_propogate))
        idx_waist_v_propogate = list(idx_waist_v_propogate)
    idx_waist_v_propogate = np.array(idx_waist_v_propogate)

    '''
    verts_pose_raw = torch.FloatTensor(mesh_atlas_sewing.vertices).cuda()
    weight_skin = diffusion_skin(verts_pose_raw*10)
    weight_skin = F.softmax(weight_skin, dim=-1)
    verts_pose_raw = skinning_diffuse(verts_pose_raw.unsqueeze(0), w_smpl, tfs, pose_offsets, shape_offsets, weight_skin, Rot_rest, pose_offsets_rest)
    verts_pose_raw -= root_J
    verts_pose_raw = verts_pose_raw.squeeze().detach()
    mesh_pose_raw = trimesh.Trimesh((verts_pose_raw*scale + trans).cpu().detach().numpy(), mesh_atlas_sewing.faces)
    mesh_pose_raw.export(os.path.join(save_path, 'mesh_pose_raw.obj'))
    '''

    v_barycentric_f, closest_face_idx_f = barycentric_faces(mesh_pattern_f, mesh_uv)
    v_barycentric_b, closest_face_idx_b = barycentric_faces(mesh_pattern_b, mesh_uv)
    ##删了 idx_v！！！！！
    #idx_v = np.arange(len(mesh_atlas_sewing.vertices))

    cloth_scale = rescale_cloth(beta, body_zero, smpl_server)
    material = Material()
    cloth = Cloth_from_NP(mesh_atlas_sewing.vertices*cloth_scale[None], mesh_atlas_sewing.faces, material)

    uv_faces_cuda = torch.from_numpy(uv_faces).cuda()
    cloth_related = [cloth, torch.FloatTensor(v_barycentric_f).cuda(), torch.from_numpy(closest_face_idx_f).cuda(), torch.FloatTensor(v_barycentric_b).cuda(), torch.from_numpy(closest_face_idx_b).cuda(), torch.from_numpy(waist_edges).cuda()]
    #cloth_related = [cloth, torch.FloatTensor(v_barycentric_f).cuda(), torch.from_numpy(closest_face_idx_f).cuda(), torch.FloatTensor(v_barycentric_b).cuda(), torch.from_numpy(closest_face_idx_b).cuda(), torch.from_numpy(idx_v).cuda(), torch.from_numpy(waist_edges).cuda()]

    v_indicator_f, v_indicator_b, verts_pose_f, verts_pose_b = optimize_prior(images, images_body, extractor, extractorBody, featuror, field, uv_features_f.detach(), uv_features_b.detach(), verts_pose_f.unsqueeze(0).detach(), verts_pose_b.unsqueeze(0).detach(), verts_pose_raw, vertices_keep, vertices_all, mask_normal, v_indicator_f_target, v_indicator_b_target, transform, transform_100, trans, scale, body_collision, cloth_related, uv_faces_cuda, iters=300, file_loss_path=file_loss_path, weight_strain=5)
    verts_pose_f = verts_pose_f.cpu().numpy()
    verts_pose_b = verts_pose_b.cpu().numpy()

    tri_f = verts_pose_f[mesh_uv.faces[closest_face_idx_f]]
    tri_b = verts_pose_b[mesh_uv.faces[closest_face_idx_b]]
    verts_f = trimesh.triangles.barycentric_to_points(tri_f, v_barycentric_f)
    verts_b = trimesh.triangles.barycentric_to_points(tri_b, v_barycentric_b)
    faces_pattern_f = mesh_pattern_f.faces
    faces_pattern_b = mesh_pattern_b.faces
    

    trimesh_cloth_f = trimesh.Trimesh(verts_f, faces_pattern_f, validate=False, process=False)
    trimesh_cloth_b = trimesh.Trimesh(verts_b, faces_pattern_b, validate=False, process=False)
    trimesh_cloth_f.export(os.path.join(save_path, 'trimesh_cloth_f.obj'))
    trimesh_cloth_b.export(os.path.join(save_path, 'trimesh_cloth_b.obj'))
    trimesh_cloth = concatenate_mesh(trimesh_cloth_f, trimesh_cloth_b)

    verts_sewing = trimesh_cloth.vertices#[idx_v]
    mesh_atlas_sewing_posed = mesh_atlas_sewing.copy()
    mesh_atlas_sewing_posed_trans = mesh_atlas_sewing.copy()
    mesh_atlas_sewing_posed.vertices = verts_sewing
    mesh_atlas_sewing_posed_trans.vertices = verts_sewing*scale.cpu().numpy() + trans.cpu().numpy()
    mesh_atlas_sewing_posed_trans.export(os.path.join(save_path, 'mesh_atlas_sewing_posed.obj'))
    mesh_atlas_sewing_posed_trans.vertices[:,1] += 0.05
    mesh_atlas_sewing_posed_trans.export(os.path.join(save_path, 'mesh_atlas_sewing_posed_offset.obj'))
    
    tri_f_infer = verts_pose_f_infer[mesh_uv.faces[closest_face_idx_f]]
    tri_b_infer = verts_pose_b_infer[mesh_uv.faces[closest_face_idx_b]]
    verts_f_infer = trimesh.triangles.barycentric_to_points(tri_f_infer, v_barycentric_f)
    verts_b_infer = trimesh.triangles.barycentric_to_points(tri_b_infer, v_barycentric_b)

    mesh_atlas_sewing_posed_trans_infer = mesh_atlas_sewing.copy()
    mesh_atlas_sewing_posed_trans_infer.vertices = np.concatenate((verts_f_infer, verts_b_infer), axis=0)[idx_v]
    mesh_atlas_sewing_posed_trans_infer.vertices = mesh_atlas_sewing_posed_trans_infer.vertices*scale.cpu().numpy() + trans.cpu().numpy()
    mesh_atlas_sewing_posed_trans_infer.export(os.path.join(save_path, 'mesh_atlas_sewing_posed_infer.obj'))
    #sys.exit()
    
    
    mesh_atlas_sewing_posed = trimesh.load(os.path.join(save_path, 'mesh_atlas_sewing_posed.obj'))
    mesh_atlas_sewing_posed.vertices = (mesh_atlas_sewing_posed.vertices - trans.cpu().numpy())/scale.cpu().numpy()

    v_waist = project_waist(mesh_atlas_sewing_posed.vertices[idx_waist_v_propogate], body_smpl)
    mesh_atlas_sewing_posed.vertices[idx_waist_v_propogate] = v_waist
    mesh_atlas_sewing_posed.export(os.path.join(save_path, 'mesh_atlas_sewing_posed_projected.obj'))

    mesh_atlas_sewing_posed_post = optimize_vertices(mesh_atlas_sewing_posed, vertices_keep, vertices_all, trans, scale, body_collision, cloth, torch.from_numpy(idx_waist_v_propogate).cuda(), mask_normal, transform_100, iters=400, smooth_boundary=True, file_loss_path=file_loss_path)
    mesh_atlas_sewing_posed_post.export(os.path.join(save_path, 'mesh_atlas_sewing_posed_post.obj'))
    
    
    return mesh_atlas_sewing_posed, mesh_atlas_sewing_posed_post


def sewing_left_right(mesh_atlas_l, idx_seam3_v):
    mesh_atlas_r = flip_mesh(mesh_atlas_l)

    map_r_v_new = np.arange(len(mesh_atlas_r.vertices))
    num_v_l = len(mesh_atlas_l.vertices)
    num_v_r_new = num_v_l - len(idx_seam3_v)
    flag_v_r_new = np.ones((num_v_l)).astype(bool)
    flag_v_r_new[idx_seam3_v] = 0
    map_r_v_new[flag_v_r_new] = np.arange(num_v_r_new) + num_v_l

    v_r_new = mesh_atlas_r.vertices[flag_v_r_new]
    f_r_new = mesh_atlas_r.faces
    num_f_r = len(f_r_new)
    f_r_new = map_r_v_new[f_r_new.flatten()].reshape(num_f_r, 3)

    mesh_atlas_l.vertices[idx_seam3_v, 0] = 0
    verts_sewing = np.concatenate((mesh_atlas_l.vertices, v_r_new), axis=0)
    faces_sewing = np.concatenate((mesh_atlas_l.faces, f_r_new), axis=0)
    mesh_sewing = trimesh.Trimesh(verts_sewing, faces_sewing, validate=False, process=False)
    
    return mesh_sewing

using_repair = True
ROOT_PATH = '/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video'
mesh_uv, uv_vertices, uv_faces, edges, verts_uv_cano_f_mean, verts_uv_cano_b_mean, weight_f, weight_b = init_uv_mesh(x_res=128, y_res=128, garment='Skirt')
uv = [mesh_uv, uv_vertices, uv_faces, edges]
smpl_server, Rot_rest, pose_offsets_rest = init_smpl_sever()
body_zero = trimesh.load('../extra-data/body-0-root.obj')

_, _, transform = renderer.get_transform(scale=80)
_, _, transform_100 = renderer.get_transform(scale=100)
body_renderer = renderer.get_render()

image_dir = os.path.join(ROOT_PATH, 'images')
seg_dir = os.path.join(ROOT_PATH, 'segmentation')
econ_dir = os.path.join(ROOT_PATH, 'econ')
align_dir = os.path.join(ROOT_PATH, 'align')
output_dir = os.path.join(ROOT_PATH, 'result')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

images_list = sorted(os.listdir(image_dir))

for i in range(0, len(images_list)):
    if not os.path.isfile(os.path.join(econ_dir, 'vid/%s_in_tensor.pt'%images_list[i].split('.')[0])):
        continue

    result_path = os.path.join(output_dir, images_list[i].split('.')[0])
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    file_loss_path = os.path.join(result_path, 'loss.txt')
    file_loss = open(file_loss_path, 'w')
    file_loss.close()
        
    model_isp, latent_codes, model_cnn_regressor = load_model(numG=400, garment='Skirt')

    vid = torch.load(os.path.join(econ_dir, 'vid/%s_in_tensor.pt'%images_list[i].split('.')[0]))
    normal_econ_512 = vid['normal_F'][0].permute(1,2,0).cpu().numpy()
    mask_sam = cv2.imread(os.path.join(seg_dir, '%s-sam-labeled.png'%images_list[i].split('.')[0]))[:,:,0] == 240

    #改名！！！！！
    normal_econ = cv2.imread(os.path.join(align_dir, '%s_normal_align.png'%images_list[i].split('.')[0]))
    data = np.load(os.path.join(align_dir, '%s_bni_v2.npz'%images_list[i].split('.')[0]))
    pose = torch.FloatTensor(data['pose']).cuda()
    beta = torch.FloatTensor(data['beta']).cuda()
    trans = torch.FloatTensor(data['trans']).cuda()
    scale = torch.FloatTensor(data['scale']).cuda()

    _, _, verts, _, _, root_joint = infer_smpl(pose, beta, smpl_server, return_root=True)
    root_joint = root_joint.squeeze().detach().cpu().numpy()
    verts = verts.squeeze().detach().cpu().numpy()
    verts -= root_joint

    body_smpl_trans = trimesh.Trimesh(verts*scale.cpu().numpy() + trans.cpu().numpy(), smpl_server.faces)
    body_smpl_trans.export(os.path.join(result_path, 'body_smpl.obj'))

    body_collision = Body(body_smpl_trans.faces)
    body_collision.update_body(torch.FloatTensor(body_smpl_trans.vertices).unsqueeze(0).cuda())

    images, images_body = prepare_input(normal_econ_512, mask_sam, verts, smpl_server.faces, body_renderer, result_path, save_img=True)
    
    mesh_atlas_sewing_posed, mesh_atlas_sewing_posed_post = reconstruct_v3(model_cnn_regressor, model_isp, latent_codes, [images, images_body], pose, beta, uv, Rot_rest, pose_offsets_rest, verts_uv_cano_f_mean, verts_uv_cano_b_mean, weight_f, weight_b, smpl_server, trans, scale, result_path)
    
    sys.exit()
    