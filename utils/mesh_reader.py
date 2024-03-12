import numpy as np 
import os, sys
import trimesh
import torch

def reorder_vertices_faces(vertices, faces_new, labels=None):
    faces_new_flatten = faces_new.reshape(-1)
    mask = np.zeros((len(vertices))).astype(bool)
    mask[faces_new_flatten] = True
    vertices_reorder = vertices[mask]
    if labels is not None:
        labels_reorder = labels[mask]

    re_id = np.zeros((len(vertices))).astype(int) - 1
    re_id[mask] = np.arange(len(vertices_reorder))
    faces_reorder = re_id[faces_new_flatten].reshape(-1, 3)

    if labels is not None:
        return vertices_reorder, faces_reorder, labels_reorder
    else:
        return vertices_reorder, faces_reorder

def read_mesh_from_sdf_test_batch_v2_with_label(vertices, faces, sdf, labels, edges, thresh=0, reorder=False):
    batch_size = len(sdf)
    
    edge_sdf = sdf[:,edges.reshape(-1)].reshape(batch_size, -1, 2) # (n, e, 2)
    
    edge_sdf_count = (edge_sdf<thresh).float().sum(dim=-1) == 1 # (n, e)
    idx_b, idx_e = torch.where(edge_sdf_count)
    
    v0 = vertices[idx_b, edges[idx_e, 0]]
    v1 = vertices[idx_b, edges[idx_e, 1]]
    v0_sdf = edge_sdf[idx_b, idx_e, 0][:, None]
    v1_sdf = edge_sdf[idx_b, idx_e, 1][:, None]

    v_border = ((v0*(v1_sdf-thresh)) + v1*(thresh-v0_sdf))/(v1_sdf - v0_sdf) # (n, e, 3)

    _, idx_v = torch.where(edge_sdf[idx_b,idx_e] >= thresh)
    idx_v = edges[idx_e, idx_v]

    vertices[idx_b, idx_v] = v_border

    tri_sdf = sdf[:, faces.reshape(-1)].reshape(batch_size, -1, 3)
    flag_in = (tri_sdf<thresh).sum(dim=-1)

    faces_list = []
    for b in range(batch_size):
        tri_in = faces[flag_in[b]>0]
        faces_list.append(tri_in)

    if reorder:
        vertices_list = []
        labels_list = []
        vertices = vertices.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        for b in range(batch_size):
            vertices_new, faces_new, labels_new = reorder_vertices_faces(vertices[b], faces_list[b].cpu().numpy(), labels=labels[b])
            faces_list[b] = faces_new
            vertices_list.append(vertices_new)
            labels_list.append(labels_new)

        return vertices_list, faces_list, labels_list

    return vertices, faces_list, labels

def check_closest_i_b(closest_i_b):
    for i in range(len(closest_i_b)-1):
        if closest_i_b[i] > closest_i_b[i+1]:
            closest_i_b[i] = closest_i_b[i+1]

    return closest_i_b

def triangulation_seam_v2(mesh_f, mesh_b, idx_seam_f, idx_seam_b, idx_offset, reverse=False):

    len_f = len(idx_seam_f)
    len_b = len(idx_seam_b)

    v_seam_f = mesh_f.vertices[idx_seam_f]
    v_seam_b = mesh_b.vertices[idx_seam_b]

    idx_f = idx_seam_f
    idx_b = [idx + idx_offset for idx in idx_seam_b]

    
    distance = ((v_seam_f[:, None] - v_seam_b[None])**2).sum(axis=-1)
    closest_i_b = np.argmin(distance, axis=-1)
    closest_i_b[0] = 0
    closest_i_b = check_closest_i_b(closest_i_b)
    #print(closest_i_b)

    i_seam_v0_f = 0 
    i_seam_v1_f = 1
    faces_new = []
    while i_seam_v0_f < len_f-1:
        if closest_i_b[i_seam_v0_f] == closest_i_b[i_seam_v1_f]:
            faces_new.append([idx_f[i_seam_v0_f], idx_f[i_seam_v1_f], idx_b[closest_i_b[i_seam_v1_f]]])
        else:
            faces_new.append([idx_f[i_seam_v0_f], idx_f[i_seam_v1_f], idx_b[closest_i_b[i_seam_v1_f]]])
            i_seam_v0_b = closest_i_b[i_seam_v0_f]
            while i_seam_v0_b < closest_i_b[i_seam_v1_f]:
                faces_new.append([idx_b[i_seam_v0_b], idx_f[i_seam_v0_f], idx_b[i_seam_v0_b+1]])
                i_seam_v0_b += 1

        i_seam_v0_f += 1
        i_seam_v1_f += 1

    if closest_i_b[-1] != len_b-1:
        i_seam_v0_b = closest_i_b[-1]
        while i_seam_v0_b < len_b-1:
            faces_new.append([idx_b[i_seam_v0_b], idx_f[-1], idx_b[i_seam_v0_b+1]])
            i_seam_v0_b += 1

    faces_new = np.array(faces_new).astype(int)

    if reverse:
        faces_new = faces_new[:, [0,2,1]]
    return faces_new
    
def catch_collar_vertices_v2(mesh, idx_v, mesh_body):
    #cannot be used for general purpose
    idx_v_right2left = idx_v[::-1]
    edges = [tuple(sorted([idx_v_right2left[i], idx_v_right2left[i+1]])) for i in range(len(idx_v_right2left)-1)]
    edges_set = set(edges)
    edges_faces = {}
    for i in range(len(mesh.faces)):
        f = mesh.faces[i]
        e1 = tuple(sorted([f[0], f[1]]))
        e2 = tuple(sorted([f[1], f[2]]))
        e3 = tuple(sorted([f[2], f[0]]))
        if e1 in edges_set:
            edges_faces[e1] = i
        if e2 in edges_set:
            edges_faces[e2] = i
        if e3 in edges_set:
            edges_faces[e3] = i

    face_idx = []
    for e in edges:
        face_i = edges_faces[e]
        face_idx.append(face_i)

    f_center = mesh.triangles_center[face_idx]
    f_normals = mesh.face_normals[face_idx]
        
    closest_face_points, _, face_body_id = trimesh.proximity.closest_point(mesh_body, f_center)

    normal_b2g = f_center - closest_face_points

    indicator_collar = (f_normals*normal_b2g).sum(axis=-1) < 0 
    indicator_collar = indicator_collar.tolist() 

    for i in range(len(indicator_collar)):
        if not indicator_collar[i]:
            break
    idx_collar = idx_v_right2left[:i+1][::-1]
    idx_sleeve = idx_v_right2left[i:][::-1]

    return idx_collar, idx_sleeve
