import os,sys
import numpy as np 
import trimesh
import pymesh

def concatenate_mesh(mesh_left, mesh_right):
    verts = np.concatenate((mesh_left.vertices, mesh_right.vertices), axis=0)
    faces = np.concatenate((mesh_left.faces, len(mesh_left.vertices) + mesh_right.faces), axis=0)

    mesh_new = trimesh.Trimesh(verts, faces, validate=False, process=False)
    return mesh_new

def flip_mesh(mesh):
    verts = mesh.vertices.copy()
    verts[:, 0] *=-1
    mesh_new = trimesh.Trimesh(verts, mesh.faces[:,[0,2,1]], validate=False, process=False)
    return mesh_new

def repair_pattern(mesh_trimesh, res=128):

    mesh = pymesh.form_mesh(mesh_trimesh.vertices, mesh_trimesh.faces)
    count = 0
    target_len_long = 2/res*np.sqrt(2)*1.2
    target_len_short = 2/res*0.4
    print("before repair #v{}".format(mesh.num_vertices))
    mesh, __ = pymesh.split_long_edges(mesh, target_len_long)

    num_vertices = mesh.num_vertices
    print("#v: {}".format(num_vertices))
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len_short, preserve_feature=True)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 120.0, 100)
        if mesh.num_vertices == num_vertices:
            break

        num_vertices = mesh.num_vertices
        print("#v: {}".format(num_vertices))
        count += 1
        if count > 10: break

    mesh_trimesh_new  = trimesh.Trimesh(mesh.vertices, mesh.faces, validate=False, process=False)
    return mesh_trimesh_new

def barycentric_faces(mesh_query, mesh_base):
    v_query = mesh_query.vertices
    base = trimesh.proximity.ProximityQuery(mesh_base)
    closest_pt, _, closest_face_idx = base.on_surface(v_query)
    triangles = mesh_base.triangles[closest_face_idx]
    v_barycentric = trimesh.triangles.points_to_barycentric(triangles, closest_pt)
    return v_barycentric, closest_face_idx

def project_waist(v_waist, mesh_base, threshold=0.01):
    base = trimesh.proximity.ProximityQuery(mesh_base)
    closest_pt, _, closest_face_idx = base.on_surface(v_waist)
    face_normals = mesh_base.face_normals[closest_face_idx]
    
    closest_pt = closest_pt+face_normals*threshold
    return closest_pt