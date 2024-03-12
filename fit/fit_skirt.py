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
from networks import SDF, convnext, diffusion
from utils import mesh_reader, renderer
from utils.img_utils import prepare_input
from utils.mesh_utils import concatenate_mesh, flip_mesh, repair_pattern, barycentric_faces, project_waist
from utils.isp_cut import select_boundary, connect_2_way, one_ring_neighour, create_uv_mesh, get_connected_paths_skirt
from smpl_pytorch.body_models import SMPL
from utils.optimization_skirt import optimize_lat_code, optimize_prior, optimize_vertices


from snug.snug_class import Body, Cloth_from_NP, Material
try:
    import pymesh
except:
    pass

def init_uv_mesh(x_res=128, y_res=128):
    uv_vertices, uv_faces = create_uv_mesh(x_res, y_res, debug=False)
    mesh_uv = trimesh.Trimesh(uv_vertices, uv_faces, validate=False, process=False)
    uv_vertices = torch.FloatTensor(uv_vertices).cuda()
    edges = torch.LongTensor(mesh_uv.edges).cuda()

    pattern_mean = np.load('/scratch/cvlab/datasets/common/CLOTH3D-Grading/ISP-cut-flatten/pattern-mean-Skirt.npz')
    verts_uv_cano_f_mean = torch.FloatTensor(pattern_mean['verts_uv_cano_f_mean']).cuda()
    verts_uv_cano_b_mean = torch.FloatTensor(pattern_mean['verts_uv_cano_b_mean']).cuda()
    verts_uv_cano_f_mean[:,1] += 0.1
    verts_uv_cano_b_mean[:,1] += 0.1

    diffusion_pattern_mean = np.load('/scratch/cvlab/datasets/common/CLOTH3D-Grading/ISP-cut-flatten/diffusion-pattern-mean-Skirt.npz')
    weight_f = torch.FloatTensor(diffusion_pattern_mean['weight_f']).cuda()
    weight_b = torch.FloatTensor(diffusion_pattern_mean['weight_b']).cuda()

    return mesh_uv, uv_vertices, uv_faces, edges, verts_uv_cano_f_mean, verts_uv_cano_b_mean, weight_f, weight_b


def infer_smpl(pose, beta, smpl_server, return_root=False, return_joints=False):
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

    if return_root:
        return w, tfs, verts, pose_offsets, shape_offsets, root_J
    elif return_joints:
        return w, tfs, verts, pose_offsets, shape_offsets, joints
    else:
        return w, tfs, verts, pose_offsets, shape_offsets

def init_smpl_sever(gender='f'):
    smpl_server = SMPL(model_path='/scratch/cvlab/home/ren/code/snug-pytorch/smpl_pytorch',
                            gender=gender,
                            use_hands=False,
                            use_feet_keypoints=False,
                            dtype=torch.float32).cuda()

    pose = torch.zeros(1, 72).cuda()
    beta = torch.zeros(1, 10).cuda()
    pose = pose.reshape(24,3)
    pose[1, 2] = .15
    pose[2, 2] = -.15
    pose = pose.reshape(-1).unsqueeze(0)
    w, tfs, verts_zero, pose_offsets, shape_offsets = infer_smpl(pose, beta, smpl_server)
    Rot_rest = torch.einsum('nk,kij->nij', w.squeeze(), tfs.squeeze()) 
    pose_offsets_rest = pose_offsets.squeeze()
    return smpl_server, Rot_rest, pose_offsets_rest

def skinning_diffuse(points, w_smpl, tfs, pose_offsets, shape_offsets, weights, Rot_rest, pose_offsets_rest):
    
    Rot_rest_weighted_inv = torch.einsum('pn,nij->pij', weights, Rot_rest)
    Rot_rest_weighted_inv = torch.linalg.inv(Rot_rest_weighted_inv)
    offsets_rest_weighted = torch.einsum('pn,ni->pi', weights, pose_offsets_rest)

    Rot = torch.einsum('bnj,bjef->bnef', w_smpl, tfs)
    Rot_weighted = torch.einsum('pn,bnij->bpij', weights, Rot)

    offsets = pose_offsets + shape_offsets
    offsets_weighted = torch.einsum('pn,bni->bpi', weights, offsets)


    points_h = F.pad(points, (0, 1), value=1.0)
    points = torch.einsum('pij,bpj->bpi', Rot_rest_weighted_inv, points_h)
    points = points[:,:,:3]
    points = points - offsets_rest_weighted[None]

    points = points + offsets_weighted
    points_h = F.pad(points, (0, 1), value=1.0)
    points_new = torch.einsum('bpij,bpj->bpi', Rot_weighted, points_h)
    points_new = points_new[:,:,:3]
    
    return points_new
        

def load_model(numG=400):
    rep_size = 32
    model_sdf_f = SDF.SDF2branch_deepSDF(d_in=2+rep_size, d_out=1+3, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
    model_sdf_b = SDF.SDF2branch_deepSDF(d_in=2+rep_size, d_out=1+3, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
    model_rep = SDF.learnt_representations(rep_size=rep_size, samples=numG).cuda()
    model_atlas_f = SDF.SDF(d_in=2+rep_size, d_out=3, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
    model_atlas_b = SDF.SDF(d_in=2+rep_size, d_out=3, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()

    
    model_sdf_f.load_state_dict(torch.load('/scratch/cvlab/home/ren/code/cloth-from-image/checkpoints/isp-skirt/net_epoch_11999_id_sdf_f.pth'))
    model_sdf_b.load_state_dict(torch.load('/scratch/cvlab/home/ren/code/cloth-from-image/checkpoints/isp-skirt/net_epoch_11999_id_sdf_b.pth'))
    model_rep.load_state_dict(torch.load('/scratch/cvlab/home/ren/code/cloth-from-image/checkpoints/isp-skirt/net_epoch_11999_id_rep.pth'))
    model_atlas_f.load_state_dict(torch.load('/scratch/cvlab/home/ren/code/cloth-from-image/checkpoints/isp-skirt/net_epoch_11999_id_atlas_f.pth'))
    model_atlas_b.load_state_dict(torch.load('/scratch/cvlab/home/ren/code/cloth-from-image/checkpoints/isp-skirt/net_epoch_11999_id_atlas_b.pth'))
    latent_codes = model_rep.weights.detach()
    
    extractor = convnext.ConvNeXtExtractor(n_stages=4).cuda()
    extractorBody = convnext.ConvNeXtExtractorCustom(in_channel=4, n_stages=4).cuda()
    featuror = convnext.FeatureNetwork_xyz(context_dims=(96, 192, 384, 768), ave=False, cat_xyz=True).cuda()
    field = convnext.MLP(d_in=featuror.feature_dim*4, d_out=4, width=400, depth=9, gaussian=True, skip_layer=[5]).cuda()

    extractor.load_state_dict(torch.load('/scratch/cvlab/home/ren/code/cloth-from-image/checkpoints/xyz-fb-collision-10-aug-smooth-skirt/net_epoch_44_id_extractor.pth'))
    extractorBody.load_state_dict(torch.load('/scratch/cvlab/home/ren/code/cloth-from-image/checkpoints/xyz-fb-collision-10-aug-smooth-skirt/net_epoch_44_id_extractorBody.pth'))
    featuror.load_state_dict(torch.load('/scratch/cvlab/home/ren/code/cloth-from-image/checkpoints/xyz-fb-collision-10-aug-smooth-skirt/net_epoch_44_id_featuror.pth'))
    field.load_state_dict(torch.load('/scratch/cvlab/home/ren/code/cloth-from-image/checkpoints/xyz-fb-collision-10-aug-smooth-skirt/net_epoch_44_id_field.pth'))

    extractor.eval()
    extractorBody.eval()
    featuror.eval()
    field.eval()

    diffusion_skin = diffusion.skip_connection(d_in=3, width=512, depth=8, d_out=6890, skip_layer=[]).cuda()
    diffusion_skin.load_state_dict(torch.load('/scratch/cvlab/home/ren/code/cloth-from-image/checkpoints/diffusion-0-root/net_epoch_2399_id_diffusion.pth'))
    diffusion_skin.eval()
    
    model_cnn_regressor = [extractor, extractorBody, featuror, field, diffusion_skin]
    model_isp = [model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b]
    return model_isp, latent_codes, model_cnn_regressor


def reconstruct_pattern_with_label(model_isp, latent_code, uv_vertices, uv_faces, edges):
    model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b = model_isp
    with torch.no_grad():
        uv_faces_torch_f = torch.LongTensor(uv_faces).cuda()
        uv_faces_torch_b = torch.LongTensor(uv_faces[:,[0,2,1]]).cuda()
        vertices_new_f = uv_vertices[:,:2].clone()
        vertices_new_b = uv_vertices[:,:2].clone()

        uv_input = uv_vertices[:,:2]*10
        num_points = len(uv_vertices)
        latent_code = latent_code.repeat(num_points, 1)
        tic = time.time()
        pred_f = model_sdf_f(uv_input, latent_code)
        pred_b = model_sdf_b(uv_input, latent_code)
        sdf_pred_f = pred_f[:, 0]
        sdf_pred_b = pred_b[:, 0]
        label_f = pred_f[:, 1:]
        label_b = pred_b[:, 1:]
        label_f = torch.argmax(label_f, dim=-1)
        label_b = torch.argmax(label_b, dim=-1)

        sdf_pred = torch.stack((sdf_pred_f, sdf_pred_b), dim=0)
        uv_vertices_batch = torch.stack((uv_vertices[:,:2], uv_vertices[:,:2]), dim=0)
        label_pred = torch.stack((label_f, label_b), dim=0)
        vertices_new, faces_list, labels_list = mesh_reader.read_mesh_from_sdf_test_batch_v2_with_label(uv_vertices_batch, uv_faces_torch_f, sdf_pred, label_pred, edges, reorder=True)
        vertices_new_f = vertices_new[0]
        vertices_new_b = vertices_new[1]
        faces_new_f = faces_list[0]
        faces_new_b = faces_list[1][:,[0,2,1]]
        label_new_f = labels_list[0]
        label_new_b = labels_list[1]
        #print(label_new_f.shape)
        #sys.exit()

        v_f = np.zeros((len(vertices_new_f), 3))
        v_b = np.zeros((len(vertices_new_b), 3))
        v_f[:, :2] = vertices_new_f
        v_b[:, :2] = vertices_new_b
        mesh_pattern_f = trimesh.Trimesh(v_f, faces_new_f, validate=False, process=False)
        mesh_pattern_b = trimesh.Trimesh(v_b, faces_new_b, validate=False, process=False)
        if using_repair:
            print('repair mesh_pattern_f')
            mesh_pattern_f = repair_pattern(mesh_pattern_f, res=256)
            print('repair mesh_pattern_b')
            mesh_pattern_b = repair_pattern(mesh_pattern_b, res=256)
            
        
        pattern_vertices_f = torch.FloatTensor(mesh_pattern_f.vertices).cuda()[:,:2]
        pattern_vertices_b = torch.FloatTensor(mesh_pattern_b.vertices).cuda()[:,:2]

        pred_f = model_sdf_f(pattern_vertices_f*10, latent_code[:len(pattern_vertices_f)])
        pred_b = model_sdf_b(pattern_vertices_b*10, latent_code[:len(pattern_vertices_b)])
        label_new_f = pred_f[:, 1:]
        label_new_b = pred_b[:, 1:]
        label_new_f = torch.argmax(label_new_f, dim=-1).cpu().numpy()
        label_new_b = torch.argmax(label_new_b, dim=-1).cpu().numpy()

        pred_atlas_f = model_atlas_f(pattern_vertices_f*10, latent_code[:len(pattern_vertices_f)])/10
        pred_atlas_b = model_atlas_b(pattern_vertices_b*10, latent_code[:len(pattern_vertices_b)])/10

        mesh_atlas_f = trimesh.Trimesh(pred_atlas_f.cpu().numpy(), mesh_pattern_f.faces, process=False, valid=False)
        mesh_atlas_b = trimesh.Trimesh(pred_atlas_b.cpu().numpy(), mesh_pattern_b.faces, process=False, valid=False)

        idx_boundary_v_f, boundary_edges_f = select_boundary(mesh_pattern_f)
        idx_boundary_v_b, boundary_edges_b = select_boundary(mesh_pattern_b)
        boundary_edges_f = set([tuple(sorted(e)) for e in boundary_edges_f.tolist()])
        boundary_edges_b = set([tuple(sorted(e)) for e in boundary_edges_b.tolist()])
        label_boundary_v_f = label_new_f[idx_boundary_v_f]
        label_boundary_v_b = label_new_b[idx_boundary_v_b]

    return mesh_atlas_f, mesh_atlas_b, mesh_pattern_f, mesh_pattern_b, label_new_f, label_new_b

def rescale_cloth(beta, body_zero, smpl_server):
    pose = torch.zeros(1,72).cuda()
    _, _, verts_body, _, _ = infer_smpl(pose, beta, smpl_server)
    verts_body = verts_body.squeeze().cpu().numpy()

    len_v = verts_body.max(axis=0) - verts_body.min(axis=0)
    len_v_zero = body_zero.vertices.max(axis=0) - body_zero.vertices.min(axis=0)
    scale = len_v/len_v_zero
    return scale

def reconstruct_v3(cnn_regressor, model_isp, latent_codes, images, pose, beta, uv, Rot_rest, pose_offsets_rest, verts_uv_cano_f_mean, verts_uv_cano_b_mean, weight_f, weight_b, smpl_server, trans, scale, save_path):
    with torch.no_grad():
        images, images_body = images
        extractor, extractorBody, featuror, field, diffusion_skin = cnn_regressor
        mesh_uv, uv_vertices, uv_faces, edges = uv
        
        uv_vertices_f = uv_vertices.clone()
        uv_vertices_b = uv_vertices.clone()
        uv_vertices_f[:,-1] = 1
        uv_vertices_b[:,-1] = -1

        w_smpl, tfs, verts_body, pose_offsets, shape_offsets, root_J = infer_smpl(pose, beta, smpl_server, return_root=True)

        verts_cano_f = verts_uv_cano_f_mean.unsqueeze(0)
        verts_cano_b = verts_uv_cano_b_mean.unsqueeze(0)

        features = extractor(images)
        featuresBody_f = extractorBody(images_body[:,:4])
        featuresBody_b = extractorBody(images_body[:,4:])
        
        verts_pose_f = skinning_diffuse(verts_cano_f, w_smpl, tfs, pose_offsets, shape_offsets, weight_f, Rot_rest, pose_offsets_rest)
        verts_pose_b = skinning_diffuse(verts_cano_b, w_smpl, tfs, pose_offsets, shape_offsets, weight_b, Rot_rest, pose_offsets_rest)
        verts_pose_f -= root_J
        verts_pose_b -= root_J

        verts_pose_f_2D = transform.transform_points(verts_pose_f)[:,:,:2]*(-1)
        verts_pose_b_2D = transform.transform_points(verts_pose_b)[:,:,:2]*(-1)

        uv_features_f = featuror.uv_embed(uv_vertices_f.unsqueeze(0))
        uv_features_b = featuror.uv_embed(uv_vertices_b.unsqueeze(0))
        feature_f = featuror.forward_embeding(uv_features_f, verts_pose_f, verts_pose_f_2D, features, featuresBody_f, featuresBody_b)
        feature_b = featuror.forward_embeding(uv_features_b, verts_pose_b, verts_pose_b_2D, features, featuresBody_f, featuresBody_b)

        delta_x_f = field(feature_f)
        delta_x_b = field(feature_b)

        verts_pose_f = verts_pose_f + delta_x_f[:,:,:3]/10
        verts_pose_b = verts_pose_b + delta_x_b[:,:,:3]/10
        v_indicator_f, v_indicator_b = F.sigmoid(delta_x_f[:,:, -1]), F.sigmoid(delta_x_b[:,:, -1])

        verts_pose_f = verts_pose_f.squeeze().detach()
        verts_pose_b = verts_pose_b.squeeze().detach()
        v_indicator_f, v_indicator_b = v_indicator_f.detach().squeeze().reshape(-1) > 0.5, v_indicator_b.detach().squeeze().reshape(-1) > 0.5
        verts_pose_f_infer = verts_pose_f.cpu().numpy()
        verts_pose_b_infer = verts_pose_b.cpu().numpy()

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
        mesh_uv_f.export(os.path.join(save_path, 'mesh_uv_f.obj'))
        mesh_uv_b.export(os.path.join(save_path, 'mesh_uv_b.obj'))

    
    lat_code_opt,  mesh_uv_f_new, mesh_uv_b_new, v_indicator_f_target, v_indicator_b_target = optimize_lat_code(model_isp, latent_codes, uv_vertices, uv_faces, v_indicator_f, v_indicator_b, iters=1000)
    dic = {'lat_code_opt':lat_code_opt, 'mesh_uv_f_new':mesh_uv_f_new, 'mesh_uv_b_new':mesh_uv_b_new, 'v_indicator_f_target':v_indicator_f_target, 'v_indicator_b_target':v_indicator_b_target}
    dump_pkl(dic, os.path.join(save_path, 'status.pkl'))
    mesh_uv_f_new.export(os.path.join(save_path, 'mesh_uv_f_new.obj'))
    mesh_uv_b_new.export(os.path.join(save_path, 'mesh_uv_b_new.obj'))
    
    '''
    #dic = load_pkl(os.path.join(save_path, 'status.pkl'))
    dic = load_pkl(os.path.join('/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video/processed/result-/frame_117', 'status.pkl'))
    lat_code_opt = dic['lat_code_opt']
    mesh_uv_f_new = dic['mesh_uv_f_new']
    mesh_uv_b_new = dic['mesh_uv_b_new']
    v_indicator_f_target = dic['v_indicator_f_target']
    v_indicator_b_target = dic['v_indicator_b_target']
    '''
    
    with torch.no_grad():
        uv_vertices_256, uv_faces_256 = create_uv_mesh(256, 256, debug=False)
        mesh_uv_256 = trimesh.Trimesh(uv_vertices_256, uv_faces_256, validate=False, process=False)
        uv_vertices_256 = torch.FloatTensor(uv_vertices_256).cuda()
        edges_256 = torch.LongTensor(mesh_uv_256.edges).cuda()

        mesh_atlas_f, mesh_atlas_b, mesh_pattern_f, mesh_pattern_b, label_f, label_b = reconstruct_pattern_with_label(model_isp, lat_code_opt, uv_vertices_256, uv_faces_256, edges_256)
        num_v_f = len(mesh_atlas_f.vertices)
        num_f_f = len(mesh_atlas_f.faces)
        
        z_offset = 0.005
        mesh_atlas_f.vertices[:, -1] += z_offset
        mesh_atlas = concatenate_mesh(mesh_atlas_f, mesh_atlas_b)

        mesh_pattern_f.export(os.path.join(save_path, 'mesh_pattern_f.obj'))
        mesh_pattern_b.export(os.path.join(save_path, 'mesh_pattern_b.obj'))
        mesh_atlas_f.export(os.path.join(save_path, 'mesh_atlas_f.obj'))
        mesh_atlas_b.export(os.path.join(save_path, 'mesh_atlas_b.obj'))

        mesh_atlas_sewing, labels = sewing_front_back(mesh_pattern_f, mesh_pattern_b, mesh_atlas_f, mesh_atlas_b, label_f, label_b, z_offset=z_offset)
        
        mesh_atlas_sewing.export(os.path.join(save_path, 'mesh_atlas_sewing.obj'))

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
        
        #C_g_f = np.array([[0,0,0]]*mesh_atlas_sewing.vertices.shape[0], np.uint8)
        #C_g_f[idx_waist_v_propogate] = 255
        #mesh_atlas_sewing.visual.vertex_colors = C_g_f
        #mesh_atlas_sewing.export(os.path.join('../tmp', 'waist.obj'))
        

        verts_pose_raw = torch.FloatTensor(mesh_atlas_sewing.vertices).cuda()
        weight_skin = diffusion_skin(verts_pose_raw*10)
        weight_skin = F.softmax(weight_skin, dim=-1)
        verts_pose_raw = skinning_diffuse(verts_pose_raw.unsqueeze(0), w_smpl, tfs, pose_offsets, shape_offsets, weight_skin, Rot_rest, pose_offsets_rest)
        verts_pose_raw -= root_J
        verts_pose_raw = verts_pose_raw.squeeze().detach()

        mesh_pose_raw = trimesh.Trimesh((verts_pose_raw*scale + trans).cpu().detach().numpy(), mesh_atlas_sewing.faces)
        mesh_pose_raw.export(os.path.join(save_path, 'mesh_pose_raw.obj'))

        faces_pattern_f = mesh_pattern_f.faces
        faces_pattern_b = mesh_pattern_b.faces
        v_barycentric_f, closest_face_idx_f = barycentric_faces(mesh_pattern_f, mesh_uv)
        v_barycentric_b, closest_face_idx_b = barycentric_faces(mesh_pattern_b, mesh_uv)
        #query = trimesh.proximity.ProximityQuery(mesh_atlas)
        #_, idx_v = query.vertex(mesh_atlas_sewing.vertices)
        #print(len(mesh_atlas_sewing.vertices), len(mesh_atlas.vertices), len(set(idx_v.flatten().tolist())))
        #sys.exit()
        idx_v = np.arange(len(mesh_atlas_sewing.vertices))


    
    cloth_scale = rescale_cloth(beta, body_zero, smpl_server)
    print(cloth_scale)
    material = Material()
    #cloth = Cloth_from_NP(mesh_atlas_sewing.vertices, mesh_atlas_sewing.faces, material)
    cloth = Cloth_from_NP(mesh_atlas_sewing.vertices*cloth_scale[None], mesh_atlas_sewing.faces, material)

    dic = {'cloth':cloth, 'verts_pose_raw':verts_pose_raw, 'v_barycentric_f':v_barycentric_f, 'closest_face_idx_f':closest_face_idx_f, 'v_barycentric_b':v_barycentric_b, 'closest_face_idx_b':closest_face_idx_b, 'idx_v':idx_v, 'idx_waist_v_propogate':idx_waist_v_propogate}
    dump_pkl(dic, os.path.join(save_path, 'cloth.pkl'))
    
    '''
    dic = load_pkl(os.path.join(save_path, 'cloth.pkl'))
    cloth = dic['cloth']
    verts_pose_raw = dic['verts_pose_raw']
    v_barycentric_f = dic['v_barycentric_f']
    closest_face_idx_f = dic['closest_face_idx_f']
    v_barycentric_b = dic['v_barycentric_b']
    closest_face_idx_b = dic['closest_face_idx_b']
    idx_v = dic['idx_v']
    '''
    
    uv_faces_cuda = torch.from_numpy(uv_faces).cuda()
    cloth_related = [cloth, torch.FloatTensor(v_barycentric_f).cuda(), torch.from_numpy(closest_face_idx_f).cuda(), torch.FloatTensor(v_barycentric_b).cuda(), torch.from_numpy(closest_face_idx_b).cuda(), torch.from_numpy(idx_v).cuda(), torch.from_numpy(waist_edges).cuda()]

    v_indicator_f, v_indicator_b, verts_pose_f, verts_pose_b = optimize_prior(images, images_body, extractor, extractorBody, featuror, field, uv_features_f.detach(), uv_features_b.detach(), verts_pose_f.unsqueeze(0).detach(), verts_pose_b.unsqueeze(0).detach(), verts_pose_raw, vertices_keep, vertices_all, mask_normal, v_indicator_f_target, v_indicator_b_target, transform, transform_100, trans, scale, body_collision, cloth_related, uv_faces_cuda, iters=300, file_loss_path=file_loss_path, weight_strain=5)
    verts_pose_f = verts_pose_f.cpu().numpy()
    verts_pose_b = verts_pose_b.cpu().numpy()

    tri_f = verts_pose_f[mesh_uv.faces[closest_face_idx_f]]
    tri_b = verts_pose_b[mesh_uv.faces[closest_face_idx_b]]
    verts_f = trimesh.triangles.barycentric_to_points(tri_f, v_barycentric_f)
    verts_b = trimesh.triangles.barycentric_to_points(tri_b, v_barycentric_b)
    

    trimesh_cloth_f = trimesh.Trimesh(verts_f, faces_pattern_f, validate=False, process=False)
    trimesh_cloth_b = trimesh.Trimesh(verts_b, faces_pattern_b, validate=False, process=False)
    trimesh_cloth_f.export(os.path.join(save_path, 'trimesh_cloth_f.obj'))
    trimesh_cloth_b.export(os.path.join(save_path, 'trimesh_cloth_b.obj'))
    trimesh_cloth = concatenate_mesh(trimesh_cloth_f, trimesh_cloth_b)

    verts_sewing = trimesh_cloth.vertices[idx_v]
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
mesh_uv, uv_vertices, uv_faces, edges, verts_uv_cano_f_mean, verts_uv_cano_b_mean, weight_f, weight_b = init_uv_mesh(x_res=128, y_res=128)
uv = [mesh_uv, uv_vertices, uv_faces, edges]
smpl_server, Rot_rest, pose_offsets_rest = init_smpl_sever()
body_zero = trimesh.load('/scratch/cvlab/home/ren/code/cloth-from-image/mesh-repo/body-0-root.obj')

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
        
    model_isp, latent_codes, model_cnn_regressor = load_model()

    vid = torch.load(os.path.join(econ_dir, 'vid/%s_in_tensor.pt'%images_list[i].split('.')[0]))
    mask_sam = cv2.imread(os.path.join(seg_dir, '%s-sam-labeled.png'%images_list[i].split('.')[0]))[:,:,0] == 240
    normal_econ = cv2.imread(os.path.join(align_dir, '%s_normal_align.png'%images_list[i].split('.')[0]))
    data = np.load(os.path.join(align_dir, '%s_bni_v2.npz'%images_list[i].split('.')[0]))
    pose = torch.FloatTensor(data['pose']).cuda()
    beta = torch.FloatTensor(data['beta']).cuda()
    trans = torch.FloatTensor(data['trans']).cuda()
    scale = torch.FloatTensor(data['scale']).cuda()

    _, _, verts, _, _, joints = infer_smpl(pose, beta, smpl_server, return_joints=True)
    root_joint = joints[0, 0].detach().cpu().numpy()
    body_smpl = trimesh.Trimesh(verts.detach().squeeze().cpu().numpy(), smpl_server.faces)
    body_smpl.vertices -= root_joint
    body_smpl_trans = trimesh.Trimesh(body_smpl.vertices*scale.cpu().numpy() + trans.cpu().numpy(), body_smpl.faces)
    body_smpl_trans.export(os.path.join(result_path, 'body_smpl.obj'))

    body_collision = Body(body_smpl_trans.faces)
    vertices_body = torch.FloatTensor(body_smpl_trans.vertices).unsqueeze(0).cuda()
    body_collision.update_body(vertices_body)

    images, images_body = prepare_input(normal_econ_512, mask_sam, body_smpl, body_renderer, result_path, save_img=True)
    
    mesh_atlas_sewing_posed, mesh_atlas_sewing_posed_post = reconstruct_v3(model_cnn_regressor, model_isp, latent_codes, [images, images_body], pose, beta, uv, Rot_rest, pose_offsets_rest, verts_uv_cano_f_mean, verts_uv_cano_b_mean, weight_f, weight_b, smpl_server, trans, scale, result_path)
    
    sys.exit()
    