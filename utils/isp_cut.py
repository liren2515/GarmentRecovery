import os, sys
import numpy as np 
import trimesh
import networkx as nx
from scipy.spatial import Delaunay
from functools import reduce

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPoint, Point
from shapely.prepared import prep


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
