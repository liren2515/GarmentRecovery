import torch
import torch.nn.functional as F
import numpy as np 
import os, sys
import trimesh
import cv2

from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance

from snug.snug_helper import stretching_energy, bending_energy, gravitational_energy, collision_penalty
from utils.isp_cut import select_boundary
from utils.rasterize import get_raster, get_pix_to_face

def clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    mask = cv2.erode(mask, kernel)
    return mask

def reconstruct_pattern(model_isp, latent_code, uv_vertices, uv_faces):
    model_sdf_f, model_sdf_b, _, _ = model_isp
    with torch.no_grad():
        uv_vertices_flip = uv_vertices.clone()
        uv_vertices_flip[:, 0] *= -1

        uv_input = uv_vertices[:,:2]*10
        uv_input_flip = uv_vertices_flip[:,:2]*10
        num_points = len(uv_vertices)
        latent_code = latent_code.repeat(num_points, 1)
        pred_f = model_sdf_f(uv_input, latent_code)
        pred_f_flip = model_sdf_f(uv_input_flip, latent_code)
        pred_b = model_sdf_b(uv_input, latent_code)
        pred_b_flip = model_sdf_b(uv_input_flip, latent_code)
        sdf_pred_f = pred_f[:, 0] < 0
        sdf_pred_f_flip = pred_f_flip[:, 0] < 0
        sdf_pred_b = pred_b[:, 0] < 0
        sdf_pred_b_flip = pred_b_flip[:, 0] < 0

        sdf_pred_f = sdf_pred_f.detach().squeeze().cpu().numpy()
        sdf_pred_b = sdf_pred_b.detach().squeeze().cpu().numpy()
        sdf_pred_f_flip = sdf_pred_f_flip.detach().squeeze().cpu().numpy()
        sdf_pred_b_flip = sdf_pred_b_flip.detach().squeeze().cpu().numpy()

        sdf_pred_f = (sdf_pred_f + sdf_pred_f_flip) > 0
        sdf_pred_b = (sdf_pred_b + sdf_pred_b_flip) > 0

        mesh_uv_f = trimesh.Trimesh(uv_vertices.squeeze().cpu().numpy(), uv_faces, process=False, valid=False)
        mesh_uv_b = trimesh.Trimesh(uv_vertices.squeeze().cpu().numpy(), uv_faces, process=False, valid=False)
        C_g_f = np.array([[0,0,0]]*mesh_uv_f.vertices.shape[0], np.uint8)
        C_g_b = np.array([[0,0,0]]*mesh_uv_b.vertices.shape[0], np.uint8)
        C_g_f[sdf_pred_f] = 255
        C_g_b[sdf_pred_b] = 255
        mesh_uv_f.visual.vertex_colors = C_g_f
        mesh_uv_b.visual.vertex_colors = C_g_b

    return mesh_uv_f, mesh_uv_b, sdf_pred_f.astype(int), sdf_pred_b.astype(int)


def erode_indicator(v_indicator, res=128):
    v_indicator_img = v_indicator.cpu().numpy().reshape(res,res).astype(np.uint8)*255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    v_indicator_img = cv2.erode(v_indicator_img, kernel)

    v_indicator_img = torch.BoolTensor(v_indicator_img.astype(int)/255).cuda().reshape(-1)

    return v_indicator_img


def optimize_lat_code(model_isp, latent_codes, uv_vertices, uv_faces, v_indicator_f, v_indicator_b, iters=1000):
    model_sdf_f, model_sdf_b, _, _ = model_isp
    
    for param in model_sdf_f.parameters():
        param.requires_grad = False
    for param in model_sdf_b.parameters():
        param.requires_grad = False

    v_indicator_f = erode_indicator(v_indicator_f, res=128)
    v_indicator_b = erode_indicator(v_indicator_b, res=128)

    uv_vertices_f_in = uv_vertices[v_indicator_f]
    uv_vertices_b_in = uv_vertices[v_indicator_b]
    idx_right_f_in = uv_vertices_f_in[:, 0] > 0
    idx_right_b_in = uv_vertices_b_in[:, 0] > 0
    uv_vertices_f_in[idx_right_f_in, 0] *= -1
    uv_vertices_b_in[idx_right_b_in, 0] *= -1

    idx_right = uv_vertices[:, 0] > 0

    uv_vertices_f_out = uv_vertices[torch.logical_or(~v_indicator_f, idx_right)]
    uv_vertices_b_out = uv_vertices[torch.logical_or(~v_indicator_b, idx_right)]

    lat_code = latent_codes.mean(dim=0, keepdim=True)
    lat_code.requires_grad = True
    
    lr = 1e-3
    eps = 0#-1e-3
    optimizer = torch.optim.Adam([{'params': lat_code, 'lr': lr},])

    uv_vertices_f_in_input = uv_vertices_f_in[:,:2]*10
    uv_vertices_b_in_input = uv_vertices_b_in[:,:2]*10
    uv_vertices_f_out_input = uv_vertices_f_out[:,:2]*10
    uv_vertices_b_out_input = uv_vertices_b_out[:,:2]*10
    uv_vertices_f_in_input.requires_grad = False
    uv_vertices_b_in_input.requires_grad = False
    uv_vertices_f_out_input.requires_grad = False
    uv_vertices_b_out_input.requires_grad = False

    for i in range(iters):
        latent_code_f_in = lat_code.repeat(len(uv_vertices_f_in), 1)
        latent_code_b_in = lat_code.repeat(len(uv_vertices_b_in), 1)
        latent_code_f_out = lat_code.repeat(len(uv_vertices_f_out), 1)
        latent_code_b_out = lat_code.repeat(len(uv_vertices_b_out), 1)

        pred_f_in = model_sdf_f(uv_vertices_f_in_input, latent_code_f_in)
        pred_b_in = model_sdf_b(uv_vertices_b_in_input, latent_code_b_in)
        pred_f_out = model_sdf_f(uv_vertices_f_out_input, latent_code_f_out)
        pred_b_out = model_sdf_b(uv_vertices_b_out_input, latent_code_b_out)

        sdf_pred_f_in = pred_f_in[:, 0]
        sdf_pred_b_in = pred_b_in[:, 0]
        sdf_pred_f_out = pred_f_out[:, 0]
        sdf_pred_b_out = pred_b_out[:, 0]

        loss_sdf = F.relu(eps + sdf_pred_f_in).mean() + F.relu(eps + sdf_pred_b_in).mean() + F.relu(-eps -sdf_pred_f_out).mean() + F.relu(-eps -sdf_pred_b_out).mean()
        loss_rep = lat_code.norm(dim=-1).mean()
        loss = loss_sdf/4 + loss_rep/100

        if i%100 == 0 or i == iters-1:
            print('iter: %3d, loss: %0.5f, loss_sdf: %0.5f, loss_rep: %0.5f'%(i, loss.item(), loss_sdf.item(), loss_rep.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mesh_uv_f, mesh_uv_b, v_indicator_f_new, v_indicator_b_new = reconstruct_pattern(model_isp, lat_code, uv_vertices, uv_faces)

    v_indicator_f_new = torch.from_numpy(v_indicator_f_new).cuda()
    v_indicator_b_new = torch.from_numpy(v_indicator_b_new).cuda()
    
    v_indicator_f_new = v_indicator_f_new > 0.5
    v_indicator_b_new = v_indicator_b_new > 0.5

    return lat_code.detach(), v_indicator_f_new, v_indicator_b_new
    

def uv_to_3D(pattern_deform, uv_faces, barycentric_uv, closest_face_idx_uv):
    uv_faces_id = uv_faces[closest_face_idx_uv]
    uv_faces_id = uv_faces_id.reshape(-1)

    pattern_deform_triangles = pattern_deform[uv_faces_id].reshape(-1, 3, 3)
    pattern_deform_bary = (pattern_deform_triangles * barycentric_uv[:, :, None]).sum(dim=-2)
    return pattern_deform_bary


def collision_penalty_body(va, vb, nb, eps=2e-3, kcollision=2500):
    with torch.no_grad():
        vec = va[:, None] - vb[None]
        dist = torch.sum(vec**2, dim=-1)
        closest_vertices = torch.argmin(dist, dim=-1)
    
    vb_collect = vb[closest_vertices]
    nb_collect = nb[closest_vertices]

    distance = (nb_collect*(va - vb_collect)).sum(dim=-1) 
    interpenetration = torch.nn.functional.relu(eps - distance)

    return (interpenetration**3).sum() * kcollision

def optimize_body(gar_mesh, pose, beta, smpl_server, body, trans, scale, iters=800, tune_pose=False, lr=1e-3):

    vertices_gar = torch.FloatTensor(gar_mesh.vertices).cuda()
    vertices_gar.requires_grad = False


    with torch.no_grad():
        _, joints_2D_gt, _, _, _ = smpl_server.forward_verts(betas=beta,
                                        body_pose=pose[:, 3:],
                                        global_orient=pose[:, :3],
                                        return_verts=True,
                                        return_full_pose=True,
                                        v_template=smpl_server.v_template, rectify_root=False)
        joints_2D_gt = joints_2D_gt[0,:,:2].detach()

    pose = pose.detach()
    beta = beta.detach()
    #lr = 1e-3
    eps = 5e-4
    if tune_pose:
        pose.requires_grad = True
        beta.requires_grad = True
        optimizer = torch.optim.Adam([{'params': pose, 'lr': lr},
                                    {'params': beta, 'lr': lr},]
        )
    else:
        pose.requires_grad = False
        beta.requires_grad = True
        optimizer = torch.optim.Adam([#{'params': pose, 'lr': lr},
                                    {'params': beta, 'lr': lr},]
        )


    for i in range(iters):
        verts, joints, _, _, _ = smpl_server.forward_verts(betas=beta,
                                        body_pose=pose[:, 3:],
                                        global_orient=pose[:, :3],
                                        return_verts=True,
                                        return_full_pose=True,
                                        v_template=smpl_server.v_template, rectify_root=False)
        
        verts = verts.squeeze()
        joints = joints.squeeze()
        root_joint = joints[[0]]
        verts = verts - root_joint
        verts = verts*scale + trans
        
        with torch.no_grad():
            body.update_body(verts.clone().unsqueeze(0).detach())
            nb = body.nb.detach().squeeze()
        
            
        loss_collision = collision_penalty_body(vertices_gar, verts, nb, eps=eps)
        if tune_pose:
            loss_2D = (joints[:,:2] - joints_2D_gt).norm(p=2, dim=-1).mean() #+ joints[[21,23],-1].sum()
            loss = loss_collision + loss_2D
            print('iter: %3d, loss: %0.4f, loss_collision: %0.4f, loss_2D: %0.4f'%(i, loss.item(), loss_collision.item(), loss_2D.item()))
        else:
             loss = loss_collision
             print('iter: %3d, loss: %0.4f'%(i, loss.item()))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        verts, joints, _, _, _ = smpl_server.forward_verts(betas=beta,
                                        body_pose=pose[:, 3:],
                                        global_orient=pose[:, :3],
                                        return_verts=True,
                                        return_full_pose=True,
                                        v_template=smpl_server.v_template, rectify_root=False)         
        verts = verts.squeeze()
        joints = joints.squeeze()
        root_joint = joints[[0]]
        verts = verts - root_joint
    body_mesh = trimesh.Trimesh((verts*scale + trans).squeeze().cpu().detach().numpy(), smpl_server.faces)
    return pose.detach(), beta.detach(), body_mesh


def optimize_prior(images_input, models_prior, uv_features_f, uv_features_b, verts_pose_f, verts_pose_b, edges_pseudo, mask_normal, mask_top, v_indicator_f_target, v_indicator_b_target, transforms, trans, scale, body, cloth_related, uv_faces, iters=350, file_loss_path=None):

    images, images_body = images_input
    transform, transform_100 = transforms
    extractor, extractorBody, featuror, field = models_prior

    cloth, v_barycentric_f, closest_face_idx_f, v_barycentric_b, closest_face_idx_b, v_barycentric, closest_face_idx, waist_edges, atlas_faces = cloth_related
    v_barycentric_f.requires_grad = False
    closest_face_idx_f.requires_grad = False
    v_barycentric_b.requires_grad = False
    closest_face_idx_b.requires_grad = False

    verts_pose_f_2D = (transform.transform_points(verts_pose_f)[:,:,:2]*(-1)).detach()
    verts_pose_b_2D = (transform.transform_points(verts_pose_b)[:,:,:2]*(-1)).detach()

    for param in extractor.parameters():
        param.requires_grad = False
    for param in extractorBody.parameters():
        param.requires_grad = False
    for param in featuror.parameters():
        param.requires_grad = False
    for param in field.parameters():
        param.requires_grad = True

    lr = 1e-4
    optimizer = torch.optim.Adam(list(field.parameters()), lr=lr)

    verts_pose_f.requires_grad = False
    verts_pose_b.requires_grad = False
    images.requires_grad = False
    images_body.requires_grad = False

    target_f = torch.zeros_like(v_indicator_f_target).float()
    target_b = torch.zeros_like(v_indicator_b_target).float()
    target_f[v_indicator_f_target] = 1
    target_b[v_indicator_b_target] = 1
    target_f = target_f.unsqueeze(0)
    target_b = target_b.unsqueeze(0)

    vb = body.vb
    nb = body.nb
    vb.requires_grad = False
    nb.requires_grad = False
    eps = 0
 
    mask_top = torch.BoolTensor(mask_top).cuda()
    idx_x, idx_y = np.where((mask_normal==0).sum(axis=-1)!=3)
    idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
    idx_mask = torch.FloatTensor(idx_mask).cuda()
    idx_mask_pend = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1)
    normal = (mask_normal[idx_x, idx_y].astype(float)/255*2) - 1
    normal = torch.FloatTensor(normal).cuda()
    normal_img = normal/normal.norm(p=2, dim=-1, keepdim=True)
    
    edges_oritation_gt = edges_pseudo[:, 0] - edges_pseudo[:, 1]

    raster = get_raster(render_res=512, scale=100)
    faces_body = torch.from_numpy(body.f).cuda()
    
    loss_min = 1000000000000000000000000
    delta_x_f_best = None
    delta_x_b_best = None
    if not (file_loss_path is None):
        file_loss = open(file_loss_path, 'a')
        
    #iters = 300
    for i in range(iters):
        features = extractor(images)
        featuresBody_f = extractorBody(images_body[:,:4])
        featuresBody_b = extractorBody(images_body[:,4:])
        feature_f = featuror.forward_embeding(uv_features_f, verts_pose_f, verts_pose_f_2D, features, featuresBody_f, featuresBody_b)
        feature_b = featuror.forward_embeding(uv_features_b, verts_pose_b, verts_pose_b_2D, features, featuresBody_f, featuresBody_b)

        delta_x_f = field(feature_f)
        delta_x_b = field(feature_b)

        verts_pose_f_new = verts_pose_f + delta_x_f[:,:,:3]/10
        verts_pose_b_new = verts_pose_b + delta_x_b[:,:,:3]/10
        v_indicator_f, v_indicator_b = F.sigmoid(delta_x_f[:,:, -1]), F.sigmoid(delta_x_b[:,:, -1])
        _indicator_f, _indicator_b = v_indicator_f > 0.5, v_indicator_b > 0.5

        verts_pose_f_new = verts_pose_f_new.squeeze()
        verts_pose_b_new = verts_pose_b_new.squeeze()

        verts_f = uv_to_3D(verts_pose_f_new, uv_faces, v_barycentric_f, closest_face_idx_f)
        verts_b = uv_to_3D(verts_pose_b_new, uv_faces, v_barycentric_b, closest_face_idx_b)
        verts = torch.cat((verts_f, verts_b), dim=0)
        verts_pose = uv_to_3D(verts, atlas_faces, v_barycentric, closest_face_idx)
        verts_pose = verts_pose*scale + trans

        with torch.no_grad():
            idx_faces, idx_vertices = get_pix_to_face(verts_pose, cloth.f, vb.squeeze(), faces_body, raster)
            mesh = trimesh.Trimesh(verts_pose.squeeze().detach().cpu().numpy(), cloth.f[idx_faces].squeeze().detach().cpu().numpy(), process=False, validate=False)
            faces = cloth.f[idx_faces]
        tri = verts_pose[faces.reshape(-1)].reshape(-1,3,3)
        tri_center = tri.mean(dim=1)
        vectors = tri[:,1:] - tri[:,:2]
        normal = torch.cross(vectors[:, 0], vectors[:, 1], dim=-1)
        normal = normal/normal.norm(p=2, dim=-1, keepdim=True)
        normal = torch.cat((normal[:,:2], normal[:,[-1]].abs()), dim=-1)
        normal = normal.unsqueeze(0)

        # loss to image observation
        verts_pose_2D = (transform_100.transform_points(tri_center.unsqueeze(0))[:,:,[1,0]]*(-255.5)) + 255.5 # x,y
        with torch.no_grad():
            verts_pose_2D_round = torch.clamp(torch.round(verts_pose_2D), 0, 511).int().squeeze()
            valid_f = (~mask_top[verts_pose_2D_round[:,0],verts_pose_2D_round[:,1]]).detach()
        normal = normal[:,valid_f]
        verts_pose_2D = verts_pose_2D[:,valid_f]
        
        verts_pose_2D_pend = torch.cat((verts_pose_2D, torch.zeros(verts_pose_2D.shape[0], verts_pose_2D.shape[1], 1).cuda()), dim=-1)
        loss_cd_keep, loss_normal = chamfer_distance(verts_pose_2D_pend, idx_mask_pend.unsqueeze(0), x_normals=normal, y_normals=normal_img.unsqueeze(0))
        loss_cd_keep = loss_cd_keep/5

        loss_reg = F.binary_cross_entropy(v_indicator_f, target_f) + F.binary_cross_entropy(v_indicator_b, target_b) 

        # physical loss
        loss_strain = stretching_energy(verts_pose.unsqueeze(0), cloth)
        loss_bending = bending_energy(verts_pose.unsqueeze(0), cloth)
        loss_gravity = gravitational_energy(verts_pose.unsqueeze(0), cloth.v_mass)
        loss_collision = collision_penalty(verts_pose.unsqueeze(0), vb, nb, eps=eps)

        edges_update = verts_pose[waist_edges.reshape(-1)].reshape(-1, 2, 3)
        edges_oritation_update = edges_update[:, 0] - edges_update[:, 1]
        loss_edge = (1 - F.cosine_similarity(edges_oritation_update, edges_oritation_gt, dim=-1)).mean()

        loss = loss_cd_keep + loss_normal + loss_reg + (loss_collision + loss_bending + loss_strain + loss_gravity) + loss_edge

        line = 'iter: %3d, loss: %0.4f, loss_cd_keep: %0.4f, loss_normal: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_collision: %0.4f, loss_gravity: %0.4f, loss_reg: %0.4f , loss_edge: %0.4f '%(i, loss.item(), loss_cd_keep.item(), loss_normal.item(), loss_strain.item(), loss_bending.item(), loss_collision.item(), loss_gravity.item(), loss_reg.item(), loss_edge.item())

        if i%50 == 0 or i == iters - 1:
            print(line)
        if not (file_loss_path is None):
            file_loss.write(line+'\n')

        if loss_min > loss.item():
            delta_x_f_best = delta_x_f.clone().detach()
            delta_x_b_best = delta_x_b.clone().detach()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    verts_pose_f_new = verts_pose_f + delta_x_f_best[:,:,:3]/10
    verts_pose_b_new = verts_pose_b + delta_x_b_best[:,:,:3]/10

    if not (file_loss_path is None):
        file_loss.close()

    return verts_pose_f_new.detach().squeeze(), verts_pose_b_new.detach().squeeze()


def optimize_vertices(mesh_prior_opt, trans, scale, body, cloth, idx_waist_v_propogate, mask_normal, mask_top, transform_100, iters=200, smooth_boundary=False, file_loss_path=None):

    if smooth_boundary:
        _, boundary_edges = select_boundary(mesh_prior_opt)
        edges = mesh_prior_opt.vertices[boundary_edges.reshape(-1)].reshape(-1, 2, 3)
        edges_oritation_gt = edges[:, 0] - edges[:, 1]
        edges_oritation_gt = torch.FloatTensor(edges_oritation_gt).cuda()
        boundary_edges = torch.from_numpy(boundary_edges).cuda()

    vertices = torch.FloatTensor(mesh_prior_opt.vertices).cuda()
    offset_update = torch.zeros(mesh_prior_opt.vertices.shape).cuda()
    scale_update = torch.ones(1,3).cuda()
    scale_update = torch.ones(mesh_prior_opt.vertices.shape).cuda()
    offset_update.requires_grad = True
    scale_update.requires_grad = True
    vertices.requires_grad = False

    lr = 1e-4
    optimizer = torch.optim.Adam([{'params': offset_update, 'lr': lr},
                                {'params': scale_update, 'lr': lr},]
    )

    vertices_all.requires_grad = False
    vertices_keep.requires_grad = False

    vb = body.vb
    nb = body.nb
    vb.requires_grad = False
    nb.requires_grad = False
    eps = 5e-4

    mask_top = torch.BoolTensor(mask_top).cuda()
    idx_x, idx_y = np.where((mask_normal==0).sum(axis=-1)!=3)
    idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
    idx_mask = torch.FloatTensor(idx_mask).cuda()
    idx_mask_pend = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1)
    normal = (mask_normal[idx_x, idx_y].astype(float)/255*2) - 1
    normal = torch.FloatTensor(normal).cuda()
    normal_img = normal/normal.norm(p=2, dim=-1, keepdim=True)

    raster = get_raster(render_res=512, scale=100)
    faces_body = torch.from_numpy(body.f).cuda()
    
    if not (file_loss_path is None):
        file_loss = open(file_loss_path, 'a')

    strain_record = 0
    for i in range(iters):
        garment_update = vertices*scale_update + offset_update
        garment_update = garment_update*scale + trans

        with torch.no_grad():
            idx_faces, idx_vertices = get_pix_to_face(garment_update, cloth.f, vb.squeeze(), faces_body, raster)
            faces = cloth.f[idx_faces]
        tri = garment_update[faces.reshape(-1)].reshape(-1,3,3)
        tri_center = tri.mean(dim=1)
        vectors = tri[:,1:] - tri[:,:2]
        normal = torch.cross(vectors[:, 0], vectors[:, 1], dim=-1)
        normal = normal/normal.norm(p=2, dim=-1, keepdim=True)
        normal = torch.cat((normal[:,:2], normal[:,[-1]].abs()), dim=-1)
        normal = normal.unsqueeze(0)

        verts_pose_2D = (transform_100.transform_points(tri_center.unsqueeze(0))[:,:,[1,0]]*(-255.5)) + 255.5 # x,y
        with torch.no_grad():
            verts_pose_2D_round = torch.clamp(torch.round(verts_pose_2D), 0, 511).int().squeeze()
            valid_f = (~mask_top[verts_pose_2D_round[:,0],verts_pose_2D_round[:,1]]).detach()
        normal = normal[:,valid_f]
        verts_pose_2D = verts_pose_2D[:,valid_f]
        
        verts_pose_2D_pend = torch.cat((verts_pose_2D, torch.zeros(verts_pose_2D.shape[0], verts_pose_2D.shape[1], 1).cuda()), dim=-1)
        loss_cd_keep, loss_normal = chamfer_distance(verts_pose_2D_pend, idx_mask_pend.unsqueeze(0), x_normals=normal, y_normals=normal_img.unsqueeze(0))
        loss_cd_keep = loss_cd_keep/5

        loss_strain = stretching_energy(garment_update.unsqueeze(0), cloth)/5
        loss_bending = bending_energy(garment_update.unsqueeze(0), cloth)*5
        loss_gravity = gravitational_energy(garment_update.unsqueeze(0), cloth.v_mass)
        loss_collision = collision_penalty(garment_update.unsqueeze(0), vb, nb, eps=eps)

        if smooth_boundary:
            edges_update = garment_update[boundary_edges.reshape(-1)].reshape(-1, 2, 3)
            edges_oritation_update = edges_update[:, 0] - edges_update[:, 1]
            loss_edge = (1 - F.cosine_similarity(edges_oritation_update, edges_oritation_gt, dim=-1)).mean()
        else:
            loss_edge = torch.zeros(1).cuda()

        loss = loss_cd_keep + loss_normal*2 + (loss_collision  + loss_bending + loss_strain + loss_gravity) + loss_edge


        if i%50 == 0 or i == iters - 1:
            print('iter: %3d, loss: %0.4f, loss_cd_keep: %0.4f, loss_normal: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_collision: %0.4f, loss_gravity: %0.4f , loss_edge: %0.4f'%(i, loss.item(), loss_cd_keep.item(), loss_normal.item(), loss_strain.item(), loss_bending.item(), loss_collision.item(), loss_gravity.item(), loss_edge.item()))


        if not (file_loss_path is None):
            line = 'iter: %3d, loss: %0.4f, loss_cd_keep: %0.4f, loss_normal: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_collision: %0.4f, loss_gravity: %0.4f , loss_edge: %0.4f\n'%(i, loss.item(), loss_cd_keep.item(), loss_normal.item(), loss_strain.item(), loss_bending.item(), loss_collision.item(), loss_gravity.item(), loss_edge.item())
            file_loss.write(line)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    garment_update = vertices*scale_update + offset_update
    garment_update = garment_update*scale + trans
    mesh_verts_opt = trimesh.Trimesh(garment_update.detach().squeeze().cpu().numpy(), mesh_prior_opt.faces)

    if not (file_loss_path is None):
        file_loss.close()

    return mesh_verts_opt
