import torch
import numpy as np
from utils.smpl_utils import infer_smpl
from utils.mesh_utils import repair_pattern

def infer_model(images, cnn_regressor, uv_vertices):
    images, images_body = images
    extractor, extractorBody, featuror, field, diffusion_skin = cnn_regressor
    
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

    return verts_pose_f, verts_pose_b, v_indicator_f, v_indicator_b


def reconstruct_pattern_with_label(model_isp, latent_code, uv_vertices, uv_faces, edges, using_repair=True):
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