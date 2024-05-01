import os,sys
import numpy as np 
import torch
import trimesh
import cv2
import torch.nn.functional as F
from functools import reduce

sys.path.append('../')
from utils import mesh_reader, renderer
from utils.init_utils import init_uv_mesh, init_smpl_sever, load_model
from utils.img_utils import prepare_input
from utils.mesh_utils import concatenate_mesh, flip_mesh, repair_pattern, barycentric_faces
from utils.isp_cut import select_boundary, connect_2_way, one_ring_neighour, create_uv_mesh, get_connected_paths_trousers
from utils.smpl_utils import infer_smpl, skinning_diffuse
from utils.fit_utils import infer_model, reconstruct_pattern_with_label, rescale_cloth, pseudo_e
from utils.optimization_trousers import clean_mask, optimize_lat_code, optimize_prior, optimize_vertices
from snug.snug_class import Body, Cloth_from_NP, Material


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

    if mesh_pattern_f.vertices[path_seam2_f[0], 1] < mesh_pattern_f.vertices[path_seam2_f[-1], 1]:
        path_seam2_f = path_seam2_f[::-1]
    if mesh_pattern_b.vertices[path_seam2_b[0], 1] < mesh_pattern_b.vertices[path_seam2_b[-1], 1]:
        path_seam2_b = path_seam2_b[::-1]

    if mesh_pattern_f.vertices[path_seam1_f[0], 1] < mesh_pattern_f.vertices[path_seam1_f[-1], 1]:
        path_seam1_f = path_seam1_f[::-1]
    if mesh_pattern_b.vertices[path_seam1_b[0], 1] < mesh_pattern_b.vertices[path_seam1_b[-1], 1]:
        path_seam1_b = path_seam1_b[::-1]

    idx_offset = len(mesh_pattern_f.vertices)
    faces_seam1 = mesh_reader.triangulation_seam_v2(mesh_atlas_f, mesh_atlas_b, path_seam1_f, path_seam1_b, idx_offset, reverse=False)
    faces_seam2 = mesh_reader.triangulation_seam_v2(mesh_atlas_f, mesh_atlas_b, path_seam2_f, path_seam2_b, idx_offset, reverse=True)
    faces_sewing = [mesh_atlas_f.faces, mesh_atlas_b.faces + idx_offset, faces_seam1, faces_seam2]

    
    indicator_seam3_f = label_boundary_v_f == 3
    indicator_seam3_b = label_boundary_v_b == 3
    

    idx_seam3_v_f = idx_boundary_v_f[indicator_seam3_f]
    idx_seam3_v_b = idx_boundary_v_b[indicator_seam3_b] #+ idx_offset
    idx_seam3_v = np.concatenate((idx_seam3_v_f, idx_seam3_v_b+idx_offset), axis=0)

    
    one_rings_seam3_f = one_ring_neighour(idx_seam3_v_f, mesh_pattern_f, is_dic=True, mask_set=set(idx_seam3_v_f))
    one_rings_seam3_b = one_ring_neighour(idx_seam3_v_b, mesh_pattern_b, is_dic=True, mask_set=set(idx_seam3_v_b))
    path_seam3_f, _ = connect_2_way(set(idx_seam3_v_f), one_rings_seam3_f, boundary_edges_f)
    path_seam3_b, _ = connect_2_way(set(idx_seam3_v_b), one_rings_seam3_b, boundary_edges_b)
    
    if mesh_pattern_f.vertices[path_seam3_f[0], 1] < mesh_pattern_f.vertices[path_seam3_f[-1], 1]:
        path_seam3_f = path_seam3_f[::-1]
    if mesh_pattern_b.vertices[path_seam3_b[0], 1] < mesh_pattern_b.vertices[path_seam3_b[-1], 1]:
        path_seam3_b = path_seam3_b[::-1]
        
    faces_sewing += [np.array([[path_seam3_f[-1], path_seam1_f[0], path_seam1_b[0]+idx_offset], [path_seam3_b[-1]+idx_offset, path_seam3_f[-1], path_seam1_b[0]+idx_offset]])]

    #z_offset = 0.2
    mesh_atlas_f.vertices[:, -1] += z_offset
    verts_sewing = np.concatenate((mesh_atlas_f.vertices, mesh_atlas_b.vertices), axis=0)
    faces_sewing = np.concatenate(faces_sewing, axis=0)
    mesh_sewing = trimesh.Trimesh(verts_sewing, faces_sewing, validate=False, process=False)

    labels_sewing = np.concatenate((labels_f, labels_b), axis=0)

    return  mesh_sewing, labels_sewing, idx_seam3_v

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


def reconstruct(cnn_regressor, model_isp, latent_codes, images_input, mask_normal, smpl_related, uv, verts_uv_cano_mean, trans, scale, transforms, body_smpl, save_path):   
    #######################################################
    ''' step 1: infer '''
    #######################################################
    with torch.no_grad():
        mesh_uv, uv_vertices, uv_faces, edges = uv
        transform, transform_100 = transforms

        verts_pose_f, verts_pose_b, v_indicator_f, v_indicator_b, uv_features_f, uv_features_b = infer_model(images_input, cnn_regressor, uv_vertices, smpl_related, verts_uv_cano_mean, transform)
        
        verts_pose_f_infer = verts_pose_f.cpu().numpy()
        verts_pose_b_infer = verts_pose_b.cpu().numpy()

        verts_pose = torch.cat((verts_pose_f, verts_pose_b), dim=0).cpu().numpy()*scale.cpu().numpy() + trans.cpu().numpy()
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
        mesh_uv_f.export(os.path.join(save_path, 'mesh_uv_f.obj'))
        mesh_uv_b.export(os.path.join(save_path, 'mesh_uv_b.obj'))

    
    #######################################################
    ''' step 2: optimize latent code with ISP '''
    #######################################################
    lat_code_opt, v_indicator_f_target, v_indicator_b_target = optimize_lat_code(model_isp, latent_codes, uv_vertices, uv_faces, v_indicator_f, v_indicator_b, iters=1000)

    with torch.no_grad():
        uv_vertices_256, uv_faces_256 = create_uv_mesh(256, 256, debug=False)
        mesh_uv_256 = trimesh.Trimesh(uv_vertices_256, uv_faces_256, validate=False, process=False)
        uv_vertices_256 = torch.FloatTensor(uv_vertices_256).cuda()
        edges_256 = torch.LongTensor(mesh_uv_256.edges).cuda()

        mesh_atlas_fl, mesh_atlas_bl, mesh_pattern_fl, mesh_pattern_bl, label_f, label_b = reconstruct_pattern_with_label(model_isp, lat_code_opt, uv_vertices_256, uv_faces_256, edges_256)
        mesh_pattern_fr = flip_mesh(mesh_pattern_fl)
        mesh_pattern_br = flip_mesh(mesh_pattern_bl)
        mesh_atlas_fr = flip_mesh(mesh_atlas_fl)
        mesh_atlas_br = flip_mesh(mesh_atlas_bl)
        mesh_pattern_f = concatenate_mesh(mesh_pattern_fl, mesh_pattern_fr)
        mesh_pattern_b = concatenate_mesh(mesh_pattern_bl, mesh_pattern_br)
        mesh_atlas_f = concatenate_mesh(mesh_atlas_fl, mesh_atlas_fr)
        mesh_atlas_b = concatenate_mesh(mesh_atlas_bl, mesh_atlas_br)
        num_v_f = len(mesh_atlas_f.vertices)
        num_f_f = len(mesh_atlas_f.faces)
        
        z_offset = 0.005
        mesh_atlas_f.vertices[:, -1] += z_offset
        mesh_atlas = concatenate_mesh(mesh_atlas_f, mesh_atlas_b)
        mesh_pattern_fl.export(os.path.join(save_path, 'mesh_pattern_fl.obj'))
        mesh_pattern_bl.export(os.path.join(save_path, 'mesh_pattern_bl.obj'))
        mesh_atlas_fl.export(os.path.join(save_path, 'mesh_atlas_fl.obj'))
        mesh_atlas_bl.export(os.path.join(save_path, 'mesh_atlas_bl.obj'))

        mesh_atlas_l, labels_l, idx_seam3_v = sewing_front_back(mesh_pattern_fl, mesh_pattern_bl, mesh_atlas_fl, mesh_atlas_bl, label_f, label_b, z_offset=z_offset)
        mesh_atlas_sewing = sewing_left_right(mesh_atlas_l, idx_seam3_v)
        
        mesh_atlas_sewing.export(os.path.join(save_path, 'mesh_atlas_sewing.obj'))


    #######################################################
    ''' step 3: optimize deformation prior '''
    #######################################################
    idx_boundary_v, boundary_edges = select_boundary(mesh_atlas_sewing)
    path_waist = get_connected_paths_trousers(mesh_atlas_sewing, idx_boundary_v, boundary_edges)[0]
    waist_edges = np.array([path_waist[:-1], path_waist[1:]])

    # make it a waistband
    idx_waist_v_propogate = np.unique(waist_edges.flatten()).tolist()
    for _iter in range(3):
        idx_waist_v_propogate = one_ring_neighour(idx_waist_v_propogate, mesh_atlas_sewing, is_dic=False, mask_set=None)
        idx_waist_v_propogate = set(reduce(lambda a,b: list(a)+list(b), idx_waist_v_propogate))
        idx_waist_v_propogate = list(idx_waist_v_propogate)
    idx_waist_v_propogate = np.array(idx_waist_v_propogate)
    
    # compute pseudo edges for waist
    verts_rest = torch.FloatTensor(mesh_atlas_sewing.vertices).cuda()
    edges_pseudo = pseudo_e(smpl_related, cnn_regressor[-1], verts_rest, waist_edges, scale, trans)

    v_barycentric_f, closest_face_idx_f = barycentric_faces(mesh_pattern_f, mesh_uv)
    v_barycentric_b, closest_face_idx_b = barycentric_faces(mesh_pattern_b, mesh_uv)
    v_barycentric, closest_face_idx = barycentric_faces(mesh_atlas_sewing, mesh_atlas)

    smpl_server  = smpl_related[2]
    cloth_scale = rescale_cloth(beta, body_zero, smpl_server)
    material = Material()
    cloth = Cloth_from_NP(mesh_atlas_sewing.vertices*cloth_scale[None], mesh_atlas_sewing.faces, material)
    
    
    cloth_related = [cloth, torch.FloatTensor(v_barycentric_f).cuda(), torch.from_numpy(closest_face_idx_f).cuda(), torch.FloatTensor(v_barycentric_b).cuda(), torch.from_numpy(closest_face_idx_b).cuda(), torch.FloatTensor(v_barycentric).cuda(), torch.from_numpy(closest_face_idx).cuda(), torch.from_numpy(waist_edges).cuda(), torch.LongTensor(mesh_atlas.faces).cuda()]
    
    uv_faces_cuda = torch.from_numpy(uv_faces).cuda()
    verts_pose_f, verts_pose_b = optimize_prior(images_input, cnn_regressor[:4], uv_features_f.detach(), uv_features_b.detach(), verts_pose_f.unsqueeze(0).detach(), verts_pose_b.unsqueeze(0).detach(), edges_pseudo, mask_normal, mask_top, v_indicator_f_target, v_indicator_b_target, transforms, trans, scale, body_collision, cloth_related, uv_faces_cuda, iters=300, file_loss_path=file_loss_path)
    verts_pose_f = verts_pose_f.cpu().numpy()
    verts_pose_b = verts_pose_b.cpu().numpy()

    tri_f = verts_pose_f[mesh_uv.faces[closest_face_idx_f]]
    tri_b = verts_pose_b[mesh_uv.faces[closest_face_idx_b]]
    verts_f = trimesh.triangles.barycentric_to_points(tri_f, v_barycentric_f)
    verts_b = trimesh.triangles.barycentric_to_points(tri_b, v_barycentric_b)
    faces_pattern_f = mesh_pattern_f.faces
    faces_pattern_b = mesh_pattern_b.faces
    
    verts_prior_opt = np.concatenate((verts_f, verts_b), axis=0)
    tri = verts_prior_opt[mesh_atlas.faces[closest_face_idx]]
    verts_prior_opt = trimesh.triangles.barycentric_to_points(tri, v_barycentric)

    verts_prior_opt_trans = verts_prior_opt*scale.cpu().numpy() + trans.cpu().numpy()
    mesh_prior_opt = trimesh.Trimesh(verts_prior_opt, mesh_atlas_sewing.faces, validate=False, process=False)
    mesh_prior_opt_trans = trimesh.Trimesh(verts_prior_opt_trans, mesh_atlas_sewing.faces, validate=False, process=False)
    mesh_prior_opt_trans.export(os.path.join(save_path, 'mesh_prior_opt.obj'))

    mesh_verts_opt = optimize_vertices(mesh_prior_opt, trans, scale, body_collision, cloth, torch.from_numpy(idx_waist_v_propogate).cuda(), mask_normal, mask_top, transform_100, iters=400, smooth_boundary=True, file_loss_path=file_loss_path)
    mesh_verts_opt.export(os.path.join(save_path, 'mesh_verts_opt.obj'))
    
    return mesh_verts_opt


if __name__ == "__main__":
    ROOT_PATH = '../fitting-data/trousers'
    mesh_uv, uv_vertices, uv_faces, edges, verts_uv_cano_mean, weight_f, weight_b = init_uv_mesh(x_res=128, y_res=128, garment='Trousers')
    uv = [mesh_uv, uv_vertices, uv_faces, edges]
    smpl_server, Rot_rest, pose_offsets_rest = init_smpl_sever()
    body_zero = trimesh.load('../extra-data/body-0-root.obj')

    transform = renderer.get_transform(scale=80)
    transform_100 = renderer.get_transform(scale=100)
    body_renderer = renderer.get_render()

    image_dir = os.path.join(ROOT_PATH, 'images')
    seg_dir = os.path.join(ROOT_PATH, 'processed', 'segmentation')
    econ_dir = os.path.join(ROOT_PATH, 'processed', 'econ')
    align_dir = os.path.join(ROOT_PATH, 'processed', 'align')
    pose_dir = os.path.join(ROOT_PATH, 'processed', 'pose-refine')
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
            
        model_isp, latent_codes, model_cnn_regressor = load_model(numG=200, garment='Trousers')

        vid = torch.load(os.path.join(econ_dir, 'vid/%s_in_tensor.pt'%images_list[i].split('.')[0]))
        normal_econ_512 = vid['normal_F'][0].permute(1,2,0).cpu().numpy()
        mask_sam = cv2.imread(os.path.join(seg_dir, '%s-sam-labeled.png'%images_list[i].split('.')[0]))[:,:,0] == 180

        mask_top = cv2.imread(os.path.join(seg_dir, '%s-sam-labeled.png'%images_list[i].split('.')[0]))[:,:,0]
        mask_top = np.logical_or(mask_top == 60, mask_top == 120)
        mask_top = clean_mask(mask_top.astype(np.uint8)).astype(bool)

        normal_input = cv2.imread(os.path.join(align_dir, '%s_normal_align.png'%images_list[i].split('.')[0]))
        data = np.load(os.path.join(align_dir, '%s_bni_v2.npz'%images_list[i].split('.')[0]))
        pose = torch.FloatTensor(data['pose']).cuda()
        beta = torch.FloatTensor(data['beta']).cuda()
        trans = torch.FloatTensor(data['trans']).cuda()
        scale = torch.FloatTensor(data['scale']).cuda()

        if os.path.isfile(os.path.join(pose_dir, '%s_smpl_opt.npz'%images_list[i].split('.')[0])):
            data_pose_opt = np.load(os.path.join(pose_dir, '%s_smpl_opt.npz'%images_list[i].split('.')[0]))
            pose = torch.FloatTensor(data_pose_opt['pose']).cuda().unsqueeze(0)
            beta = torch.FloatTensor(data_pose_opt['beta']).cuda().unsqueeze(0)

        _, _, verts, _, _, root_joint = infer_smpl(pose, beta, smpl_server, return_root=True)
        root_joint = root_joint.squeeze().detach().cpu().numpy()
        verts = verts.squeeze().detach().cpu().numpy()
        verts -= root_joint

        body_smpl = trimesh.Trimesh(verts, smpl_server.faces)
        body_smpl_trans = trimesh.Trimesh(verts*scale.cpu().numpy() + trans.cpu().numpy(), smpl_server.faces)
        body_smpl_trans.export(os.path.join(result_path, 'body.obj'))

        body_collision = Body(body_smpl_trans.faces)
        body_collision.update_body(torch.FloatTensor(body_smpl_trans.vertices).unsqueeze(0).cuda())

        images, images_body, mask_normal = prepare_input(normal_input, normal_econ_512, mask_sam, verts, smpl_server.faces.astype(int), body_renderer, result_path, save_img=True)
        
        smpl_related =[pose, beta, smpl_server, weight_f, weight_b, Rot_rest, pose_offsets_rest]
        transforms = [transform, transform_100]
        mesh_verts_opt = reconstruct(model_cnn_regressor, model_isp, latent_codes, [images, images_body], mask_normal, smpl_related, uv, verts_uv_cano_mean, trans, scale, transforms, body_smpl, result_path)

        sys.exit()
        