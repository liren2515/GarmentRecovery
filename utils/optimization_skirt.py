import torch
import torch.nn.functional as F
import numpy as np 
import os, sys
import trimesh
import cv2

from utils.chamfer import chamfer_distance_single
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.loss import chamfer_distance
from snug.snug_helper import stretching_energy, bending_energy, gravitational_energy, collision_penalty, spring_energy, collision_penalty_skirt, shrink_penalty
from snug.snug_class import Cloth_from_NP
from utils.isp_cut import select_boundary
from utils.rasterize import get_raster, get_pix_to_face


def reconstruct_pattern(model_isp, latent_code, uv_vertices, uv_faces):
    model_sdf_f, model_sdf_b, _, _ = model_isp
    with torch.no_grad():

        uv_input = uv_vertices[:,:2]*10
        num_points = len(uv_vertices)
        latent_code = latent_code.repeat(num_points, 1)
        pred_f = model_sdf_f(uv_input, latent_code)
        pred_b = model_sdf_b(uv_input, latent_code)
        sdf_pred_f = pred_f[:, 0] < 0
        sdf_pred_b = pred_b[:, 0] < 0

        sdf_pred_f = sdf_pred_f.detach().squeeze().cpu().numpy()
        sdf_pred_b = sdf_pred_b.detach().squeeze().cpu().numpy()

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
    uv_vertices_f_out = uv_vertices[~v_indicator_f]
    uv_vertices_b_out = uv_vertices[~v_indicator_b]

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


def compute_normals_per_vertex(
        vs: torch.Tensor,
        fs: torch.Tensor,
        face_normals: torch.Tensor = None):

    device = vs.device

    # Get number of vertices and faces
    _, nvs, _ = vs.shape
    nfs, _ = fs.shape

    # For each vertex we want to know what faces it is in
    # We will build a tensor for this. Each row will be a vertex, and each column will be a face
    # The value will be 0. if the vertex is not in the face, and 1. if it is

    # First, we create the tensor with zeros
    vertex_in_face = torch.zeros(nvs, nfs, device=device)

    # Then we will need the indices where we will put ones
    # The following tensor has a sequence repeated 3 times
    # [[0,0,0], [1,1,1], [2,2,2], ...]
    w = torch.tensor(range(nfs), device=device)[None].repeat(3, 1).t()

    # We index the rows with fs, which contains the vertices, and pair it with w
    # Simple example:
    #
    # fs = [[0, 1, 2], [2, 3, 4]]
    # w = [[0, 0, 0], [1, 1, 1]]  <- in fs there are 2 faces, so 0,1 repeated
    #
    # That would make the following pairs:
    # 0,0   1,0   2,0   2,1   3,1   4,1
    # and then the following sentence will set ones where we wanted:
    # [[1, 0],  <- vertex 0 is in face 0, but not in face 1
    #  [1, 0],
    #  [1, 1],  <- vertex 2 is in both faces
    #  [0, 1],
    #  [0, 1]]  <- vertex 4 is in face 1, but not in face 0
    vertex_in_face[fs, w] = 1.0

    # Now, for each vertex, we want the sum of the normals from each of the faces it is part of
    # vertex_in_face x face_normals = (nvs x nfs) x (nfs x 3) = (nvs x 3)
    sum_normals = torch.einsum('ij,bjk->bik', vertex_in_face, face_normals)

    # The sum takes non-unit normals, which makes it able to take into account their importance. But at this point we
    # don't want the importance anymore, so we normalize them
    normalized_sum_normals = sum_normals / sum_normals.norm(dim=2, keepdim=True)

    return normalized_sum_normals

def optimize_prior(images, images_body, extractor, extractorBody, featuror, field, uv_features_f, uv_features_b, verts_pose_f, verts_pose_b, verts_pose_raw, mask_normal, v_indicator_f_target, v_indicator_b_target, transform, transform_100, trans, scale, body, cloth_related, uv_faces, iters=350, idx_collar_f=None, file_loss_path=None):

    cloth, v_barycentric_f, closest_face_idx_f, v_barycentric_b, closest_face_idx_b, waist_edges = cloth_related
    #cloth, v_barycentric_f, closest_face_idx_f, v_barycentric_b, closest_face_idx_b, idx_v, waist_edges = cloth_related
    v_barycentric_f.requires_grad = False
    closest_face_idx_f.requires_grad = False
    v_barycentric_b.requires_grad = False
    closest_face_idx_b.requires_grad = False
    #idx_v.requires_grad = False

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
     
    cv2.imwrite('../tmp/mask_normal.png', mask_normal)
    idx_x, idx_y = np.where((mask_normal==0).sum(axis=-1)!=3)
    idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
    idx_mask = torch.FloatTensor(idx_mask).cuda()
    idx_mask_pend = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1)
    normal = (mask_normal[idx_x, idx_y].astype(float)/255*2) - 1
    normal = torch.FloatTensor(normal).cuda()
    normal_img = normal/normal.norm(p=2, dim=-1, keepdim=True)
    verts_pose = verts_pose_raw*scale + trans

    edges = verts_pose[waist_edges.reshape(-1)].reshape(-1, 2, 3)
    edges_oritation_gt = edges[:, 0] - edges[:, 1]

    raster = get_raster(render_res=512, scale=100)
    faces_body = torch.from_numpy(body.f).cuda()
    
    loss_min = 1000000000000000000000000
    delta_x_f_best = None
    delta_x_b_best = None
    if not (file_loss_path is None):
        file_loss = open(file_loss_path, 'a')
        
    strain_record = 0
    iters = 300
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


        verts_f = uv_to_3D(verts_pose_f_new, uv_faces, v_barycentric_f, closest_face_idx_f).squeeze()
        verts_b = uv_to_3D(verts_pose_b_new, uv_faces, v_barycentric_b, closest_face_idx_b).squeeze()
        verts = torch.cat((verts_f, verts_b), dim=0)
        #verts_pose = verts[idx_v]*scale + trans
        verts_pose = verts*scale + trans

        with torch.no_grad():
            idx_faces, idx_vertices = get_pix_to_face(verts_pose, cloth.f, vb.squeeze(), faces_body, raster)
            mesh = trimesh.Trimesh(verts_pose.squeeze().detach().cpu().numpy(), cloth.f[idx_faces].squeeze().detach().cpu().numpy())
            faces = cloth.f[idx_faces]
        tri = verts_pose[faces.reshape(-1)].reshape(-1,3,3)
        tri_center = tri.mean(dim=1)
        vectors = tri[:,1:] - tri[:,:2]
        normal = torch.cross(vectors[:, 0], vectors[:, 1], dim=-1)
        normal = normal/normal.norm(p=2, dim=-1, keepdim=True)
        normal = torch.cat((normal[:,:2], normal[:,[-1]].abs()), dim=-1)
        normal = normal.unsqueeze(0)


        verts_pose_2D = (transform_100.transform_points(tri_center.unsqueeze(0))[:,:,[1,0]]*(-255.5)) + 255.5 # x,y
        
        verts_pose_2D_pend = torch.cat((verts_pose_2D, torch.zeros(verts_pose_2D.shape[0], verts_pose_2D.shape[1], 1).cuda()), dim=-1)
        loss_cd_keep, loss_normal = chamfer_distance(verts_pose_2D_pend, idx_mask_pend.unsqueeze(0), x_normals=normal, y_normals=normal_img.unsqueeze(0))
        loss_cd_keep = loss_cd_keep/5

        loss_reg = F.binary_cross_entropy(v_indicator_f, target_f) + F.binary_cross_entropy(v_indicator_b, target_b) 

        loss_strain = stretching_energy(verts_pose.unsqueeze(0), cloth)/5
        loss_bending = bending_energy(verts_pose.unsqueeze(0), cloth)#*10
        loss_gravity = gravitational_energy(verts_pose.unsqueeze(0), cloth.v_mass)#*50

        with torch.no_grad():
            tri = verts_pose[cloth.f.reshape(-1)].reshape(-1,3,3)
            vectors = tri[:,1:] - tri[:,:2]
            normal_faces = torch.cross(vectors[:, 0], vectors[:, 1], dim=-1)
            normal_faces = normal_faces/normal_faces.norm(p=2, dim=-1, keepdim=True)
            normal_vertices = compute_normals_per_vertex(verts_pose.unsqueeze(0), cloth.f, normal_faces.unsqueeze(0)).squeeze()
        loss_collision = collision_penalty_skirt(verts_pose, normal_vertices, vb.squeeze(), nb.squeeze(), eps=eps)#/5


        edges_update = verts_pose[waist_edges.reshape(-1)].reshape(-1, 2, 3)
        edges_oritation_update = edges_update[:, 0] - edges_update[:, 1]
        loss_edge = (1 - F.cosine_similarity(edges_oritation_update, edges_oritation_gt, dim=-1)).mean()

        loss = loss_cd_keep + loss_normal + loss_reg + (loss_collision + loss_bending + loss_strain + loss_gravity) + loss_edge
        if i%50 == 0 or i == iters - 1:
            print('iter: %3d, loss: %0.4f, loss_cd_keep: %0.4f, loss_normal: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_collision: %0.4f, loss_gravity: %0.4f, loss_reg: %0.4f , loss_edge: %0.4f '%(i, loss.item(), loss_cd_keep.item(), loss_normal.item(), loss_strain.item(), loss_bending.item(), loss_collision.item(), loss_gravity.item(), loss_reg.item(), loss_edge.item()))

        if not (file_loss_path is None):
            line = 'iter: %3d, loss: %0.4f, loss_cd_keep: %0.4f, loss_normal: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_collision: %0.4f, loss_gravity: %0.4f, loss_reg: %0.4f, loss_edge: %0.4f \n'%(i, loss.item(), loss_cd_keep.item(), loss_normal.item(), loss_strain.item(), loss_bending.item(), loss_collision.item(), loss_gravity.item(), loss_reg.item(), loss_edge.item())
            file_loss.write(line)

        if loss_min > loss.item():
            delta_x_f_best = delta_x_f.clone().detach()
            delta_x_b_best = delta_x_b.clone().detach()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    verts_pose_f_new = verts_pose_f + delta_x_f_best[:,:,:3]/10
    verts_pose_b_new = verts_pose_b + delta_x_b_best[:,:,:3]/10
    v_indicator_f, v_indicator_b = F.sigmoid(delta_x_f_best[:,:, -1]), F.sigmoid(delta_x_b_best[:,:, -1])
    v_indicator_f, v_indicator_b = v_indicator_f > 0.5, v_indicator_b > 0.5

    if not (file_loss_path is None):
        file_loss.close()

    return v_indicator_f.detach().squeeze().reshape(-1), v_indicator_b.detach().squeeze().reshape(-1), verts_pose_f_new.detach().squeeze(), verts_pose_b_new.detach().squeeze()



def optimize_vertices(gar_mesh, vertices_keep, vertices_all, trans, scale, body, cloth, idx_waist_v_propogate, mask_normal, transform_100, iters=200, smooth_boundary=False, idx_collar_f=None, file_loss_path=None):

    if smooth_boundary:
        _, boundary_edges = select_boundary(gar_mesh)
        
        edges = gar_mesh.vertices[boundary_edges.reshape(-1)].reshape(-1, 2, 3)
        edges_oritation_gt = edges[:, 0] - edges[:, 1]
        edges_oritation_gt = torch.FloatTensor(edges_oritation_gt).cuda()
        boundary_edges = torch.from_numpy(boundary_edges).cuda()

    vertices = torch.FloatTensor(gar_mesh.vertices).cuda()
    # add small perterbation can avoid NaN in backward.
    offset_update = torch.zeros(gar_mesh.vertices.shape).cuda() #+ torch.randn(gar_mesh.vertices.shape).cuda()*0.0001
    scale_update = torch.ones(gar_mesh.vertices.shape).cuda()
    offset_update.requires_grad = True
    scale_update.requires_grad = True
    vertices.requires_grad = False
    trans.requires_grad = False

    
    scale_y = torch.ones(1).cuda()
    trans_y = torch.zeros(1).cuda()
    scale_y.requires_grad = False
    trans_y.requires_grad = False

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


    idx_x, idx_y = np.where((mask_normal==0).sum(axis=-1)!=3)
    #idx_x, idx_y = np.where(mask_sam>0)
    idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
    idx_mask = torch.FloatTensor(idx_mask).cuda()
    idx_mask_pend = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1)
    normal = (mask_normal[idx_x, idx_y].astype(float)/255*2) - 1
    normal = torch.FloatTensor(normal).cuda()
    normal_img = normal/normal.norm(p=2, dim=-1, keepdim=True)

    raster = get_raster(render_res=512, scale=100)
    faces_body = torch.from_numpy(body.f).cuda()
    
    weight_faces = torch.ones(len(cloth.f)).cuda()
    if not (idx_collar_f is None):
        weight_faces[idx_collar_f] = 10
    
    if not (file_loss_path is None):
        file_loss = open(file_loss_path, 'a')

    strain_record = 0
    for i in range(iters):
        #garment_update = vertices + offset_update
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
        
        verts_pose_2D_pend = torch.cat((verts_pose_2D, torch.zeros(verts_pose_2D.shape[0], verts_pose_2D.shape[1], 1).cuda()), dim=-1)
        loss_cd_keep, loss_normal = chamfer_distance(verts_pose_2D_pend, idx_mask_pend.unsqueeze(0), x_normals=normal, y_normals=normal_img.unsqueeze(0))
        loss_cd_keep = loss_cd_keep/5#0
        loss_cd_rest, _ = chamfer_distance_single(vertices_keep, garment_update.unsqueeze(0))
        loss_cd_rest *= 100*50/10*0
        loss_cd_rest_all, _ = chamfer_distance_single(garment_update.unsqueeze(0), vertices_all)
        loss_cd_rest_all *= 100*50/10*0#0

        #sys.exit()
        loss_strain = stretching_energy(garment_update.unsqueeze(0), cloth)/10#5#/5#/10
        #loss_strain = spring_energy(garment_update, cloth_new)/100#/10
        loss_bending = bending_energy(garment_update.unsqueeze(0), cloth)*5#2#10#2
        #mesh = Meshes(verts=[torch.zeros_like(garment_update)], faces=[cloth.f])
        #new_mesh = mesh.offset_verts(garment_update)
        #loss_bending = mesh_laplacian_smoothing(new_mesh)*500
        loss_gravity = gravitational_energy(garment_update.unsqueeze(0), cloth.v_mass)#*50
        loss_collision = collision_penalty(garment_update.unsqueeze(0), vb, nb, eps=eps)#/10

        loss_shrink = shrink_penalty(garment_update[idx_waist_v_propogate], vb.squeeze())#*10
        
        loss = (loss_cd_keep + loss_normal*2) + (loss_collision  + loss_bending + loss_strain + loss_gravity) + loss_shrink

        if smooth_boundary:
            edges_update = garment_update[boundary_edges.reshape(-1)].reshape(-1, 2, 3)
            edges_oritation_update = edges_update[:, 0] - edges_update[:, 1]
            loss_edge = (1 - F.cosine_similarity(edges_oritation_update, edges_oritation_gt, dim=-1)).mean()*10
        else:
            loss_edge = torch.zeros(1).cuda()

        loss += loss_edge
        #loss *= 0
        print('iter: %3d, loss: %0.4f, loss_cd_keep: %0.4f, loss_normal: %0.4f, loss_cd_rest: %0.4f, loss_cd_rest_all: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_collision: %0.4f, loss_gravity: %0.4f , loss_edge: %0.4f, loss_shrink: %0.4f'%(i, loss.item(), loss_cd_keep.item(), loss_normal.item(), loss_cd_rest.item(), loss_cd_rest_all.item(), loss_strain.item(), loss_bending.item(), loss_collision.item(), loss_gravity.item(), loss_edge.item(), loss_shrink.item()))


        if not (file_loss_path is None):
            line = 'iter: %3d, loss: %0.4f, loss_cd_keep: %0.4f, loss_normal: %0.4f, loss_cd_rest: %0.4f, loss_cd_rest_all: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_collision: %0.4f, loss_gravity: %0.4f , loss_edge: %0.4f, loss_shrink: %0.4f\n'%(i, loss.item(), loss_cd_keep.item(), loss_normal.item(), loss_cd_rest.item(), loss_cd_rest_all.item(), loss_strain.item(), loss_bending.item(), loss_collision.item(), loss_gravity.item(), loss_edge.item(), loss_shrink.item())
            file_loss.write(line)

        
        optimizer.zero_grad()
        loss.backward()
        offset_update.grad = torch.nan_to_num(offset_update.grad, nan=0.0)
        scale_update.grad = torch.nan_to_num(scale_update.grad, nan=0.0)
        optimizer.step()

        #print(torch.isnan(offset_update).int().sum())
        #sys.exit()

    #print(scale_update)
    garment_update = vertices*scale_update + offset_update
    garment_update = garment_update*scale + trans
    cloth_mesh = trimesh.Trimesh(garment_update.detach().squeeze().cpu().numpy(), gar_mesh.faces)


    if not (file_loss_path is None):
        file_loss.close()


    return cloth_mesh


def optimize_ablation(gar_mesh, vertices_keep, vertices_all, trans, scale, body, cloth, verts_pose_raw, waist_edges, idx_waist_v_propogate, mask_normal, transform_100, iters=200, smooth_boundary=False, idx_collar_f=None, file_loss_path=None):

    if smooth_boundary:
        verts_pose = verts_pose_raw*scale + trans
        edges = verts_pose[waist_edges.reshape(-1)].reshape(-1, 2, 3)
        edges_oritation_gt = edges[:, 0] - edges[:, 1]

    vertices = torch.FloatTensor(gar_mesh.vertices).cuda()
    offset_update = torch.zeros(gar_mesh.vertices.shape).cuda()
    scale_update = torch.ones(gar_mesh.vertices.shape).cuda()
    offset_update.requires_grad = True
    scale_update.requires_grad = True
    vertices.requires_grad = False

    
    scale_y = torch.ones(1).cuda()
    trans_y = torch.zeros(1).cuda()
    scale_y.requires_grad = True
    trans_y.requires_grad = True

    lr = 1e-4
    optimizer = torch.optim.Adam([{'params': offset_update, 'lr': lr},
                                {'params': scale_update, 'lr': lr},
                                {'params': scale_y, 'lr': lr},
                                {'params': trans_y, 'lr': lr},]
    )

    vertices_all.requires_grad = False
    vertices_keep.requires_grad = False

    vb = body.vb
    nb = body.nb
    vb.requires_grad = False
    nb.requires_grad = False
    eps = 5e-4

    idx_x, idx_y = np.where((mask_normal==0).sum(axis=-1)!=3)
    idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
    idx_mask = torch.FloatTensor(idx_mask).cuda()
    idx_mask_pend = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1)
    normal = (mask_normal[idx_x, idx_y].astype(float)/255*2) - 1
    normal = torch.FloatTensor(normal).cuda()
    normal_img = normal/normal.norm(p=2, dim=-1, keepdim=True)

    raster = get_raster(render_res=512, scale=100)
    faces_body = torch.from_numpy(body.f).cuda()
    
    weight_faces = torch.ones(len(cloth.f)).cuda()
    if not (idx_collar_f is None):
        weight_faces[idx_collar_f] = 10
    
    if not (file_loss_path is None):
        file_loss = open(file_loss_path, 'a')

    strain_record = 0
    for i in range(iters):
        garment_update = vertices*scale_update + offset_update
        garment_update = garment_update*scale + trans
        
        with torch.no_grad():
            idx_faces, idx_vertices = get_pix_to_face(garment_update, cloth.f, vb.squeeze(), faces_body, raster)
            faces = cloth.f[idx_faces]
            #mesh = trimesh.Trimesh(garment_update.detach().cpu().numpy(), faces.detach().cpu().numpy())
            #mesh.export('../tmp/mesh.obj')
            #sys.exit()
        tri = garment_update[faces.reshape(-1)].reshape(-1,3,3)
        tri_center = tri.mean(dim=1)
        vectors = tri[:,1:] - tri[:,:2]
        normal = torch.cross(vectors[:, 0], vectors[:, 1], dim=-1)
        normal = normal/normal.norm(p=2, dim=-1, keepdim=True)
        normal = torch.cat((normal[:,:2], normal[:,[-1]].abs()), dim=-1)
        normal = normal.unsqueeze(0)

        verts_pose_2D = (transform_100.transform_points(tri_center.unsqueeze(0))[:,:,[1,0]]*(-255.5)) + 255.5 # x,y
        
        verts_pose_2D_pend = torch.cat((verts_pose_2D, torch.zeros(verts_pose_2D.shape[0], verts_pose_2D.shape[1], 1).cuda()), dim=-1)
        loss_cd_keep, loss_normal = chamfer_distance(verts_pose_2D_pend, idx_mask_pend.unsqueeze(0), x_normals=normal, y_normals=normal_img.unsqueeze(0))
        loss_cd_keep = loss_cd_keep/5#0
        loss_cd_rest, _ = chamfer_distance_single(vertices_keep, garment_update.unsqueeze(0))
        loss_cd_rest *= 100*50/10*0
        loss_cd_rest_all, _ = chamfer_distance_single(garment_update.unsqueeze(0), vertices_all)
        loss_cd_rest_all *= 100*50/10*0#0

        #sys.exit()
        loss_strain = stretching_energy(garment_update.unsqueeze(0), cloth)/10#5#/5#/10
        loss_bending = bending_energy(garment_update.unsqueeze(0), cloth)#*5#2#10#2
        loss_gravity = gravitational_energy(garment_update.unsqueeze(0), cloth.v_mass)
        
        with torch.no_grad():
            tri = verts_pose[cloth.f.reshape(-1)].reshape(-1,3,3)
            vectors = tri[:,1:] - tri[:,:2]
            normal_faces = torch.cross(vectors[:, 0], vectors[:, 1], dim=-1)
            normal_faces = normal_faces/normal_faces.norm(p=2, dim=-1, keepdim=True)
            normal_vertices = compute_normals_per_vertex(verts_pose.unsqueeze(0), cloth.f, normal_faces.unsqueeze(0)).squeeze()
            #print(verts_pose.shape, normal_vertices.shape)
        loss_collision = collision_penalty_skirt(verts_pose, normal_vertices, vb.squeeze(), nb.squeeze(), eps=eps)#/5

        loss_shrink = shrink_penalty(garment_update[idx_waist_v_propogate], vb.squeeze())#*10
        
        loss = loss_cd_keep + loss_normal*2 + loss_cd_rest + (loss_collision  + loss_bending + loss_strain + loss_gravity) + loss_shrink

        if smooth_boundary:
            edges_update = verts_pose[waist_edges.reshape(-1)].reshape(-1, 2, 3)
            edges_oritation_update = edges_update[:, 0] - edges_update[:, 1]
            loss_edge = (1 - F.cosine_similarity(edges_oritation_update, edges_oritation_gt, dim=-1)).mean()*10
        else:
            loss_edge = torch.zeros(1).cuda()

        loss += loss_edge
        print('iter: %3d, loss: %0.4f, loss_cd_keep: %0.4f, loss_normal: %0.4f, loss_cd_rest: %0.4f, loss_cd_rest_all: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_collision: %0.4f, loss_gravity: %0.4f , loss_edge: %0.4f, loss_shrink: %0.4f'%(i, loss.item(), loss_cd_keep.item(), loss_normal.item(), loss_cd_rest.item(), loss_cd_rest_all.item(), loss_strain.item(), loss_bending.item(), loss_collision.item(), loss_gravity.item(), loss_edge.item(), loss_shrink.item()))


        if not (file_loss_path is None):
            line = 'iter: %3d, loss: %0.4f, loss_cd_keep: %0.4f, loss_normal: %0.4f, loss_cd_rest: %0.4f, loss_cd_rest_all: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_collision: %0.4f, loss_gravity: %0.4f , loss_edge: %0.4f, loss_shrink: %0.4f\n'%(i, loss.item(), loss_cd_keep.item(), loss_normal.item(), loss_cd_rest.item(), loss_cd_rest_all.item(), loss_strain.item(), loss_bending.item(), loss_collision.item(), loss_gravity.item(), loss_edge.item(), loss_shrink.item())
            file_loss.write(line)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #print(scale_update)
    garment_update = vertices*scale_update + offset_update
    garment_update = garment_update*scale + trans
    cloth_mesh = trimesh.Trimesh(garment_update.detach().squeeze().cpu().numpy(), gar_mesh.faces)


    if not (file_loss_path is None):
        file_loss.close()


    return cloth_mesh
