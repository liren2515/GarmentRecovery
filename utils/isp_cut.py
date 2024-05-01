import os, sys
import numpy as np 
import trimesh
import networkx as nx
from scipy.spatial import Delaunay
from functools import reduce
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from collections import defaultdict

def select_boundary(mesh):
    unique_edges = mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
    idx_boundary_v = np.unique(unique_edges.flatten())
    return idx_boundary_v, unique_edges

def connect_2_way(idx_boundary_v_set, one_rings, boundary_edges):
    path = [list(idx_boundary_v_set)[0]]
    idx_boundary_v_set.remove(path[0])
    # connect one way
    while len(idx_boundary_v_set):
        node = path[-1]
        neighbour = one_rings[node]
        for n in neighbour:
            if n in idx_boundary_v_set and tuple(sorted([node, n])) in boundary_edges:
                path.append(n)
                idx_boundary_v_set.remove(n)
                break

        if node == path[-1]:
            break


    # connect the other way
    while len(idx_boundary_v_set):
        node = path[0]
        neighbour = one_rings[node]
        for n in neighbour:
            if n in idx_boundary_v_set and tuple(sorted([node, n])) in boundary_edges:
                path.insert(0, n)
                idx_boundary_v_set.remove(n)
                break

        if node == path[0]:
            break

    return path, idx_boundary_v_set

def get_connected_paths_skirt(mesh, idx_boundary_v, boundary_edges):
    idx_boundary_v_set = set(idx_boundary_v)
    one_rings = one_ring_neighour(idx_boundary_v, mesh, is_dic=True, mask_set=idx_boundary_v_set)

    paths = []
    path_y_mean = []
    while len(idx_boundary_v_set):
        path, idx_boundary_v_set = connect_2_way(idx_boundary_v_set, one_rings, boundary_edges)

        paths.append(path)
        path_y_mean.append(mesh.vertices[path, 1].mean())

    if len(paths) != 2:
        raise Exception("Something wrong: len(paths) - get_connected_paths!!")

    bottom_path_i = path_y_mean.index(min(path_y_mean))
    bottom_path = paths[bottom_path_i]
    top_path_i = path_y_mean.index(max(path_y_mean))
    top_path = paths[top_path_i]

    return_paths = [top_path, bottom_path]

    if sorted([bottom_path_i, top_path_i]) != [0, 1]:
        raise Exception("Something wrong get_connected_paths!!")

    return return_paths

def remove_path(paths, path_y_mean, path_x_mean):

    paths_len = [len(p) for p in paths]
    remove_i = paths_len.index(min(paths_len))

    paths_new = []
    path_y_mean_new = []
    path_x_mean_new = []
    for i in range(len(paths)):
        if i == remove_i:
            continue

        paths_new.append(paths[i])
        path_y_mean_new.append(path_y_mean[i])
        path_x_mean_new.append(path_x_mean[i])

    return paths_new, path_y_mean_new, path_x_mean_new
        
def get_connected_paths_trousers(mesh, idx_boundary_v, boundary_edges):
    idx_boundary_v_set = set(idx_boundary_v)
    one_rings = one_ring_neighour(idx_boundary_v, mesh, is_dic=True, mask_set=idx_boundary_v_set)

    paths = []
    path_y_mean = []
    path_x_mean = []
    while len(idx_boundary_v_set):
        path, idx_boundary_v_set = connect_2_way(idx_boundary_v_set, one_rings, boundary_edges)

        paths.append(path)
        path_y_mean.append(mesh.vertices[path, 1].mean())
        path_x_mean.append(mesh.vertices[path, 0].mean())
    
    if len(paths) != 3:
        #raise Exception("Something wrong: len(paths) - get_connected_paths!!")
        paths, path_y_mean, path_x_mean = remove_path(paths, path_y_mean, path_x_mean)

    top_path_i = path_y_mean.index(max(path_y_mean))
    top_path = paths[top_path_i]

    left_right_i = set([0,1,2])
    left_right_i.remove(top_path_i)
    left_right_i = list(left_right_i)
    left_right_i = left_right_i if path_x_mean[left_right_i[0]] < path_x_mean[left_right_i[1]] else left_right_i[::-1]
    left_i, right_i = left_right_i
    left_path = paths[left_i]
    right_path = paths[right_i]

    return_paths = [top_path, left_path, right_path]

    if sorted([left_i, right_i, top_path_i]) != [0, 1, 2]:
        raise Exception("Something wrong get_connected_paths!!")

    return return_paths

def one_ring_neighour(idx_v, mesh, is_dic=False, mask_set=None):
    g = nx.from_edgelist(mesh.edges_unique)
    valid_v_i = set(np.unique(mesh.faces.flatten()).tolist())
    one_ring = []
    if mask_set is not None:
        for i in idx_v:
            if i in valid_v_i:
                one_ring.append(set(g[i].keys()).intersection(mask_set))
            else:
                one_ring.append(set([]))
    else:
        for i in idx_v:
            if i in valid_v_i:
                one_ring.append(set(g[i].keys()))
            else:
                one_ring.append(set([]))

    if is_dic:
        one_ring_dic = {}
        for i in range(len(idx_v)):
            one_ring_dic[idx_v[i]] = one_ring[i]

        one_ring = one_ring_dic
    return one_ring

def create_uv_mesh(x_res, y_res, debug=False):
    x = np.linspace(1, -1, x_res)
    y = np.linspace(1, -1, y_res)

    # exchange x,y to make everything consistent:
    # x is the first coordinate, y is the second!
    xv, yv = np.meshgrid(y, x)
    uv = np.stack((xv, yv), axis=-1)
    vertices = uv.reshape(-1, 2)
    
    tri = Delaunay(vertices)
    faces = tri.simplices
    vertices = np.concatenate((vertices, np.zeros((len(vertices), 1))), axis=-1)

    if debug:
        # x in plt is vertical
        # y in plt is horizontal
        plt.figure()
        plt.triplot(vertices[:,0], vertices[:,1], faces)
        plt.plot(vertices[:,0], vertices[:,1], 'o', markersize=2)
        plt.savefig('../tmp/tri.png')

    return vertices, faces

def select_collar_offset(mesh_gar, mesh_body, offset_y = 0.005, propagate_iter=3):
    triangles_center = mesh_gar.triangles_center.copy()
    triangles_center[:, 1] += offset_y
    idx_candidate = np.logical_and(triangles_center[:,1] > 0.21, np.abs(triangles_center[:,0]) < 0.21)
    idx_candidate = np.array([i for i in range(len(triangles_center)) if idx_candidate[i]])
    normals = mesh_gar.face_normals[idx_candidate]
    
    f_candidate = triangles_center[idx_candidate]
    closest_face_points, _, face_id = trimesh.proximity.closest_point(mesh_body, f_candidate)

    normal_b2g = f_candidate - closest_face_points

    idx_collar = (normals*normal_b2g).sum(axis=-1) < 0 
    idx_collar = idx_candidate[idx_collar]

    f_collar = mesh_gar.faces[idx_collar]
    idx_collar_v = np.unique(f_collar.flatten())

    idx_collar_v_propogate = idx_collar_v.tolist()
    while propagate_iter and len(idx_collar_v_propogate) != 0:
        idx_collar_v_propogate = one_ring_neighour(idx_collar_v_propogate, mesh_gar, is_dic=False, mask_set=None)
        idx_collar_v_propogate = set(reduce(lambda a,b: list(a)+list(b), idx_collar_v_propogate))
        idx_collar_v_propogate = list(idx_collar_v_propogate)
        propagate_iter -= 1

    idx_collar_v_propogate = np.array(idx_collar_v_propogate)

    return idx_collar_v_propogate

def remove_collar(mesh, idx_collar_v):
    idx_collar_v_set = set(idx_collar_v)

    faces = mesh.faces
    faces_new = []
    for f in faces:
        if f[0] in idx_collar_v_set and f[1] in idx_collar_v_set and f[2] in idx_collar_v_set:
            continue
        else:
            faces_new.append(f)
    
    faces_new = np.stack(faces_new, axis=0)

    mesh_new = trimesh.Trimesh(mesh.vertices, faces_new, validate=False, process=False)
    return mesh_new

def smooth_boundary(mesh, idx_collar_v=None):
    border_edges = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    if not (idx_collar_v is None):
        border_edges_new = []
        for e_i in border_edges:
            e = mesh.edges_sorted[e_i]
            if e[0] in idx_collar_v and e[1] in idx_collar_v:
                border_edges_new.append(e_i)
        border_edges = np.array(border_edges_new)

    # build a dictionnary of (u,l): l is the list of vertices that are adjacent to u
    neighbours = defaultdict(lambda: [])
    for u, v in mesh.edges_sorted[border_edges]:
        neighbours[u].append(v)
        neighbours[v].append(u)
    border_vertices = np.array(list(neighbours.keys()))

    # build a sparse matrix for computing laplacian
    pos_i, pos_j = [], []
    for k, ns in enumerate(neighbours.values()):
        for j in ns:
            pos_i.append(k)
            pos_j.append(j)

    sparse = coo_matrix(
        (np.ones(len(pos_i)), (pos_i, pos_j)),  # put ones at these locations
        shape=(len(border_vertices), len(mesh.vertices)),
    )

    # smoothing operation:
    lambda_ = 0.3
    for _ in range(20):
        border_neighbouring_averages = sparse @ mesh.vertices / sparse.sum(axis=1)
        laplacian = border_neighbouring_averages - mesh.vertices[border_vertices]
        mesh.vertices[border_vertices] = (
            mesh.vertices[border_vertices] + lambda_ * laplacian
        )

    return mesh

def select_sleeve(mesh_gar, mesh_body):
    idx_v = np.abs(mesh_gar.vertices[:,0]) > 0.28
    return idx_v

def select_sleeve_boundary(mesh, idx_sleeve):
    unique_edges = mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
    idx_boundary_v = np.unique(unique_edges.flatten())

    sleeve_edges = []
    for i in range(len(unique_edges)):
        if unique_edges[i,0] in idx_sleeve and unique_edges[i,1] in idx_sleeve:
            sleeve_edges.append(unique_edges[i])
    sleeve_edges = np.array(sleeve_edges)
    return sleeve_edges