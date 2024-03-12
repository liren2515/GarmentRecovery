import numpy as np
import torch
import trimesh

def get_shape_matrix(x):
    if x.ndim == 3:
        return torch.stack([x[:, 0] - x[:, 2], x[:, 1] - x[:, 2]], dim=-1)

    elif x.ndim == 4:
        return torch.stack([x[:, :, 0] - x[:, :, 2], x[:, :, 1] - x[:, :, 2]], dim=-1)

    raise NotImplementedError

def find_nearest_neighbour(A, B, dtype=np.int32):
    nearest_neighbour = np.argmin(pairwise_distance(A, B), axis=1)
    return nearest_neighbour.astype(dtype)

def get_vertex_connectivity(faces):
    '''
    Returns a list of unique edges in the mesh. 
    Each edge contains the indices of the vertices it connects
    '''
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    edges = set()
    for f in faces:
        num_vertices = len(f)
        for i in range(num_vertices):
            j = (i + 1) % num_vertices
            edges.add(tuple(sorted([f[i], f[j]])))

    return torch.LongTensor(list(edges))


def get_face_connectivity(faces):
    '''
    Returns a list of adjacent face pairs
    '''
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    edges = get_vertex_connectivity(faces).numpy()

    G = {tuple(e): [] for e in edges}
    for i, f in enumerate(faces):
        n = len(f)
        for j in range(n):
            k = (j + 1) % n
            e = tuple(sorted([f[j], f[k]]))
            G[e] += [i]

    adjacent_faces = []
    for key in G:
        
        assert len(G[key]) < 3
        G[key] = G[key][:2]
        if len(G[key]) == 2:
            adjacent_faces += [G[key]]
   
    return torch.LongTensor(adjacent_faces)


def get_face_connectivity_edges(faces):
    '''
    Returns a list of edges that connect two faces
    (i.e., all the edges except borders)
    '''
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    edges = get_vertex_connectivity(faces).numpy()

    G = {tuple(e): [] for e in edges}
    for i, f in enumerate(faces):
        n = len(f)
        for j in range(n):
            k = (j + 1) % n
            e = tuple(sorted([f[j], f[k]]))
            G[e] += [i]

    adjacent_face_edges = []
    for key in G:
        assert len(G[key]) < 3
        G[key] = G[key][:2]
        if len(G[key]) == 2:
            adjacent_face_edges += [list(key)]

    return torch.LongTensor(adjacent_face_edges)

def get_collar_body_connectivity_indicator(adjacent_face_edges, adjacent_faces, faces, idx_collar_v):
    indicator_idx = []
    for i in range(len(adjacent_face_edges)):
        edge = adjacent_face_edges[i]
        if edge[0] in idx_collar_v and edge[1] in idx_collar_v:
            edge = set(edge.tolist())
            face0 = faces[adjacent_faces[i][0]]
            face1 = faces[adjacent_faces[i][1]]
            face0 = set(face0.tolist())
            face1 = set(face1.tolist())
            v_rest_0 = list(face0 - edge)[0]
            v_rest_1 = list(face1 - edge)[0]
            if (v_rest_0 in idx_collar_v and v_rest_1 in idx_collar_v) or (v_rest_1 in idx_collar_v and v_rest_0 in idx_collar_v):
                indicator_idx.append(i)

    indicator = torch.zeros(len(adjacent_face_edges)).bool()
    indicator[indicator_idx] = 1
    return indicator


def get_vertex_mass(vertices, faces, density):
    '''
    Computes the mass of each vertex according to triangle areas and fabric density
    '''
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    areas = get_face_areas(vertices, faces)
    triangle_masses = density * areas

    vertex_masses = np.zeros(vertices.shape[0])
    np.add.at(vertex_masses, faces[:,0], triangle_masses/3)
    np.add.at(vertex_masses, faces[:,1], triangle_masses/3)
    np.add.at(vertex_masses, faces[:,2], triangle_masses/3)

    return torch.FloatTensor(vertex_masses)


def get_face_areas(vertices, faces):
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy()

    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    v0 = vertices[faces[:,0]]
    v1 = vertices[faces[:,1]]
    v2 = vertices[faces[:,2]]

    u = v2 - v0
    v = v1 - v0

    return np.linalg.norm(np.cross(u, v), axis=-1) / 2.0


def get_edge_length(vertices, edges):
    v0_idx = edges[:, 0]
    v1_idx = edges[:, 1]
    if vertices.ndim == 3:
        v0 = vertices[:, v0_idx]
        v1 = vertices[:, v1_idx]
    elif vertices.ndim == 2:
        v0 = vertices[v0_idx]
        v1 = vertices[v1_idx]
    else:
        raise NotImplementedError
    return torch.norm(v0 - v1, p=2, dim=-1)


def load_obj(filename, tex_coords=False):
    vertices = []
    faces = []
    uvs = []
    faces_uv = []

    with open(filename, 'r') as fp:
        for line in fp:
            line_split = line.split()
            
            if not line_split:
                continue

            elif tex_coords and line_split[0] == 'vt':
                uvs.append([line_split[1], line_split[2]])

            elif line_split[0] == 'v':
                vertices.append([line_split[1], line_split[2], line_split[3]])

            elif line_split[0] == 'f':
                vertex_indices = [s.split("/")[0] for s in line_split[1:]]
                faces.append(vertex_indices)

                if tex_coords:
                    uv_indices = [s.split("/")[1] for s in line_split[1:]]
                    faces_uv.append(uv_indices)

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32) - 1

    if tex_coords:
        uvs = np.array(uvs, dtype=np.float32)
        faces_uv = np.array(faces_uv, dtype=np.int32) - 1
        return vertices, faces, uvs, faces_uv

    return vertices, faces


def rotate_triangle(triangles):
    num_tri = len(triangles)
    triangles_rotated = torch.zeros_like(triangles)
    e1 = triangles[:, 0] - triangles[:, 2]
    e2 = triangles[:, 1] - triangles[:, 2]
    n = torch.cross(e1, e2, dim=-1)

    x = e1/torch.norm(e1, p=2, dim=-1, keepdim=True)
    n = n/torch.norm(n, p=2, dim=-1, keepdim=True)
    y = torch.cross(n, x, dim=-1)
    y = y/torch.norm(y, p=2, dim=-1, keepdim=True)

    coord_old = torch.stack([x, y, n], dim=-1)
    coord_new = torch.eye(3).unsqueeze(0).repeat(num_tri, 1, 1).cuda()
    matrix_rot = torch.einsum('nij,njk->nik', coord_new, coord_old.permute(0, 2, 1))

    e1_rot = torch.einsum('nij,nj->ni', matrix_rot, e1)
    e2_rot = torch.einsum('nij,nj->ni', matrix_rot, e2)

    shape_matrix = torch.stack([e1_rot, e2_rot], dim=-1)
    return shape_matrix


##################################### Classes #####################################
#                                Modified from SNUG                               #
##################################### Classes #####################################
class Material:
    '''
    This class stores parameters for the StVK material model
    '''

    def __init__(self, density=426,       # Fabric density (kg / m2)
                       thickness=4.7e-4     # Fabric thickness (m):
                ):
                       
        self.density = 426
        self.thickness = 0.47e-3 # 0.47 mm
        self.area_density = self.density*self.thickness

        self.young_modulus = 0.7e5
        self.poisson_ratio = 0.485
        self.stretch_multiplier = 1
        self.bending_multiplier = 50
        
        # Bending and stretching coefficients (ARCSim)
        self.A = self.young_modulus / (1.0 - self.poisson_ratio**2)
        self.stretch_coeff = self.A
        self.stretch_coeff *= self.stretch_multiplier
        
        self.bending_coeff = self.A / 12.0 * (self.thickness ** 3) 
        self.bending_coeff *= self.bending_multiplier

        self.collision_coeff = 250

        # Lame coefficients
        self.lame_mu =  0.5 * self.stretch_coeff * (1.0 - self.poisson_ratio)
        self.lame_lambda = self.stretch_coeff * self.poisson_ratio


class Cloth_from_NP: 
    '''
    This class stores mesh and material information of the garment
    '''
    
    def __init__(self, v, f, material, dtype=torch.float32):
        self.dtype = dtype  
        self.material = material

        v = torch.FloatTensor(v).cuda()
        f = torch.LongTensor(f).cuda()

        # Vertex attributes
        self.v_template = v
        self.v_mass = get_vertex_mass(v, f, self.material.area_density).cuda()
        self.v_velocity = torch.zeros(1, v.shape[0], 3).cuda() # Vertex velocities in global coordinates
        self.v = torch.zeros(1, v.shape[0], 3).cuda() # Vertex position in global coordinates
        self.v_psd = torch.zeros(1, v.shape[0], 3).cuda() # Pose space deformation of each vertex
        self.v_weights = None # Vertex skinning weights
        self.num_vertices = self.v_template.shape[0]
    
        # Face attributes
        self.f = f
        self.f_connectivity = get_face_connectivity(f).cuda() # Pairs of adjacent faces
        self.f_connectivity_edges = get_face_connectivity_edges(f).cuda() # Edges that connect faces
        self.f_area = torch.FloatTensor(get_face_areas(v, f)).cuda()
        self.num_faces = self.f.shape[0]

        # Edge attributes
        self.e = get_vertex_connectivity(f).cuda() # Pairs of connected vertices
        self.e_rest = get_edge_length(v, self.e) # Rest lenght of the edges (world space)
        self.num_edges = self.e.shape[0]

        self.closest_body_vertices = None

        tri = v[f.reshape(-1)]
        tri = tri.reshape(len(f), 3, 3)
        self.tri = tri
        self.Dm = rotate_triangle(tri).detach()[:, :2, :]
        self.Dm_inv = torch.linalg.inv(self.Dm).detach()


    def compute_skinning_weights(self, smpl):
        # self.v_template: numpy.array
        # smpl.template_vertices: numpy.array
        # smpl.skinning_weights: torch.tensor
        if type(self.closest_body_vertices) == type(None):
            self.closest_body_vertices = find_nearest_neighbour(self.v_template, smpl.template_vertices)
        self.closest_body_vertices = torch.LongTensor(self.closest_body_vertices)
        self.v_weights = smpl.skinning_weights[self.closest_body_vertices]
        return self.v_weights

class Body:
    def __init__(self, faces):
        self.f = faces
        self.vb = None
        self.nb = None

    def update_body(self, verts_batch):
        self.vb = verts_batch
        self.nb = self.get_verts_normal(verts_batch)

    def get_verts_normal(self, verts_batch):
        # verts_batch: [batch_size, N, 3]
        nb = []
        for i in range(len(verts_batch)):
            mesh_body = trimesh.Trimesh(verts_batch[i].detach().cpu().numpy(), self.f, process=False)
            nb.append(mesh_body.vertex_normals)

        nb = torch.FloatTensor(nb).cuda()
        return nb


class FaceNormals:
    def __init__(self, normalize=True):
        self.normalize = normalize

    def call(self, vertices, faces):
        v = vertices
        f = faces

        num_faces = len(f)
        if v.ndim == 3:
            triangles = v[:, f.reshape(-1)]
            triangles = triangles.reshape(-1, num_faces, 3, 3)
        elif v.ndim == 2:
            triangles = v[f.reshape(-1)]
            triangles = triangles.reshape(num_faces, 3, 3)
        else:
            raise NotImplementedError

        v0, v1, v2 = torch.unbind(triangles, dim=-2)
        e1 = v0 - v1
        e2 = v2 - v1
        face_normals = torch.cross(e2, e1, dim=-1) 
        
        if self.normalize:
            face_normals = face_normals/(torch.norm(face_normals, p=2, dim=-1, keepdim=True)+1e-6)

        return face_normals

    def call_batch(self, vertices, faces):
        v = vertices
        f = faces

        triangles = self.gather_triangles_batch(v, f)

        v0, v1, v2 = torch.unbind(triangles, dim=-2)
        e1 = v0 - v1
        e2 = v2 - v1
        face_normals = torch.cross(e2, e1, dim=-1) 
        
        if self.normalize:
            face_normals = (face_normals/torch.norm(face_normals, p=2, dim=-1, keepdim=True)+1e-6)

        return face_normals

    def gather_triangles_batch(self, vertices, faces):
        batch_size = faces.shape[0]
        num_faces = faces.shape[1]
        faces = faces.reshape(batch_size, -1)
        faces = faces.unsqueeze(-1).repeat(1, 1, 3)
        triangles = torch.gather(vertices, 1, faces)
        triangles = triangles.reshape(batch_size, num_faces, 3, 3)
        return triangles
