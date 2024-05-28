import os, sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from utils.readfile import load_pkl

def separate_arms(poses, angle=20, left_arm=17, right_arm=16):
    num_joints = poses.shape[-1] //3

    poses = poses.reshape((-1, num_joints, 3))
    rot = R.from_euler('z', -angle, degrees=True)
    poses[:, left_arm] = (rot * R.from_rotvec(poses[:, left_arm])).as_rotvec()
    rot = R.from_euler('z', angle, degrees=True)
    poses[:, right_arm] = (rot * R.from_rotvec(poses[:, right_arm])).as_rotvec()

    poses[:, 23] *= 0.1
    poses[:, 22] *= 0.1

    return poses.reshape((poses.shape[0], -1))

def finite_diff(x, h, diff=1):
    if diff == 0:
        return x

    v = np.zeros(x.shape, dtype=x.dtype)
    v[1:] = (x[1:] - x[0:-1]) / h

    return finite_diff(v, h, diff-1)

def pairwise_distance(A, B):
    rA = np.sum(np.square(A), axis=1)
    rB = np.sum(np.square(B), axis=1)
    distances = - 2*np.matmul(A, np.transpose(B)) + rA[:, np.newaxis] + rB[np.newaxis, :]
    return distances

def find_nearest_neighbour(A, B, dtype=np.int32):
    nearest_neighbour = np.argmin(pairwise_distance(A, B), axis=1)
    return nearest_neighbour.astype(dtype)


def get_shape_matrix(x):
    if x.ndim == 3:
        return torch.stack([x[:, 0] - x[:, 2], x[:, 1] - x[:, 2]], dim=-1)

    elif x.ndim == 4:
        return torch.stack([x[:, :, 0] - x[:, :, 2], x[:, :, 1] - x[:, :, 2]], dim=-1)

    raise NotImplementedError
    

def gather_triangles(vertices, indices):
    # indices: [num_faces, 3]
    # vertices: [batch_size, num_points, 3]
    num_faces = len(indices)
    triangles = vertices[:, indices.reshape(-1)]
    triangles = triangles.reshape(-1, num_faces, 3, 3)
    return triangles


################################# Loss Computation #################################
#                                Modified from SNUG                                #
################################# Loss Computation #################################
from snug.snug_class import FaceNormals

def deformation_gradient(triangles, Dm_inv):
    # Dm_inv: [num_faces, 2, 2]
    Ds = get_shape_matrix(triangles)
    if Ds.ndim == 3:
        return torch.einsum('nij,njk->nik', Ds, Dm_inv)
    elif Ds.ndim == 4 and Dm_inv.ndim == 3:
        return torch.einsum('bnij,njk->bnik', Ds, Dm_inv)
    elif Ds.ndim == 4 and Dm_inv.ndim == 4:
        return torch.einsum('bnij,bnjk->bnik', Ds, Dm_inv)
    raise NotImplementedError


def green_strain_tensor(F):
    I = torch.eye(2, dtype=F.dtype, device=F.device)
    Ft = torch.permute(F, [0, 1, 3, 2])
    return (torch.einsum('bnij,bnjk->bnik', Ft, F) - I[None, None, :, :])*0.5


def stretching_energy(v, cloth, return_average=True, weight_f=None): 
    '''
    Computes strech energy of the cloth for the vertex positions v
    Material model: Saint-Venant-Kirchhoff (StVK)
    Reference: ArcSim (physics.cpp)
    '''

    batch_size = v.shape[0]
    triangles = gather_triangles(v, cloth.f)

    Dm_inv = cloth.Dm_inv.clone()

    F = deformation_gradient(triangles, Dm_inv)
    G = green_strain_tensor(F)

    # Energy
    mat = cloth.material
    I = torch.eye(2, dtype=G.dtype, device=G.device).unsqueeze(0).unsqueeze(0).repeat(batch_size, G.shape[1], 1, 1)
    trace_G = torch.einsum('bnii->bn', G)
    S = mat.lame_mu * G + 0.5 * mat.lame_lambda * trace_G[:, :, None, None] * I
    energy_density = torch.einsum('bnij,bnjk->bnik', torch.permute(S, [0, 1, 3, 2]), G)
    energy_density = torch.einsum('bnii->bn', energy_density)
    energy = cloth.f_area[None] * mat.thickness * energy_density

    if not (weight_f is None):
        energy = energy*weight_f

    if return_average:
        return energy.sum() / batch_size
    
    return energy.sum(dim=-1)

def bending_energy(v, cloth, return_average=True): 
    '''
    Computes the bending energy of the cloth for the vertex positions v
    Reference: ArcSim (physics.cpp)
    '''

    batch_size = v.shape[0]

    # Compute face normals
    fn = FaceNormals().call(v, cloth.f)
    
    face0_idx = cloth.f_connectivity[:, 0]
    face1_idx = cloth.f_connectivity[:, 1]
    n0 = fn[:, face0_idx]
    n1 = fn[:, face1_idx]

    # Compute edge length
    v0_idx = cloth.f_connectivity_edges[:, 0]
    v1_idx = cloth.f_connectivity_edges[:, 1]
    e = v[:, v1_idx] - v[:, v0_idx]
    l = torch.norm(e, p=2, dim=-1, keepdim=True)
    e_norm = e/l

    # Compute area
    f_area = cloth.f_area.unsqueeze(0).repeat(batch_size, 1)
    a0 = f_area[:, face0_idx]
    a1 = f_area[:, face1_idx]
    a = a0 + a1

    # Compute dihedral angle between faces
    cos = (n0 * n1).sum(dim=-1)
    sin = (e_norm * torch.cross(n0, n1, dim=-1)).sum(dim=-1)
    theta = torch.atan2(sin, cos)
    
    # Compute bending coefficient according to material parameters,
    # triangle areas (a) and edge length (l)
    mat = cloth.material
    scale = l[:, :, 0]**2 / (4*a)
    valid = ~torch.isnan(theta)

    # Bending energy
    energy = mat.bending_coeff * scale * (theta ** 2) / 2
    energy = energy[valid]

    if return_average:
        return energy.sum() / batch_size

    return energy.sum(dim=-1)

def dihedral_angle(v, cloth):
    # Compute face normals
    fn = FaceNormals().call(v, cloth.f)

    face0_idx = cloth.f_connectivity[:, 0]
    face1_idx = cloth.f_connectivity[:, 1]
    n0 = fn[:, face0_idx]
    n1 = fn[:, face1_idx]
    
    v0_idx = cloth.f_connectivity_edges[:, 0]
    v1_idx = cloth.f_connectivity_edges[:, 1]
    e = v[:, v1_idx] - v[:, v0_idx]
    l = torch.norm(e, p=2, dim=-1, keepdim=True)
    e_norm = e/l

    cos = (n0 * n1).sum(dim=-1)
    sin = (e_norm * torch.cross(n0, n1, dim=-1)).sum(dim=-1)
    theta = torch.atan2(sin, cos)

    return theta, l


def gravitational_energy(x, mass, g=9.81, return_average=True, shift_ground=False, offset=0, z=False):
    batch_size = x.shape[0]
    if shift_ground:
        x[:, :, 1] += offset
    U = g * mass[None, None, :] * x[:, :, 1]

    if z:
        U = g * mass[None, None, :] * x[:, :, -1]

    if return_average:
        return U.sum() / batch_size

    return U.sum(dim=-1)


def collision_penalty(va, vb, nb, eps=2e-3, kcollision=2500):#250): # eps=2e-3 ????
    batch_size = va.shape[0]
    vec = va[:, :, None] - vb[:, None]
    dist = torch.sum(vec**2, dim=-1)
    closest_vertices = torch.argmin(dist, dim=-1)
    
    closest_vertices = closest_vertices.unsqueeze(-1).repeat(1,1,3)
    vb = torch.gather(vb, 1, closest_vertices)
    nb = torch.gather(nb, 1, closest_vertices)

    distance = (nb*(va - vb)).sum(dim=-1) 
    interpenetration = torch.nn.functional.relu(eps - distance)

    return (interpenetration**3).sum() / batch_size * kcollision

def shrink_penalty(va, vb):

    vec = va[:, None] - vb[None]
    dist = torch.sum(vec**2, dim=-1)
    closest_vertices = torch.argmin(dist, dim=-1)
    vb = vb[closest_vertices]
    distance = (va-vb).norm(p=2, dim=-1)

    return distance.mean()

partsIdx = load_pkl('../extra-data/parts14_vertex_idx.pkl')
idx_leg = partsIdx[3]+partsIdx[4]+partsIdx[5]+partsIdx[6]+partsIdx[7]+partsIdx[8]
idx_arm = partsIdx[1]+partsIdx[2]+partsIdx[9]+partsIdx[10]+partsIdx[11]+partsIdx[12]
idx_leg = torch.LongTensor(idx_leg).cuda()
idx_arm = torch.LongTensor(idx_arm).cuda()
indicator_leg = torch.zeros(6890).cuda().bool()
indicator_arm = torch.zeros(6890).cuda().bool()
indicator_leg[idx_leg] = 1
indicator_arm[idx_arm] = 1
indicator_rest = torch.ones(6890).cuda().bool()
indicator_rest[idx_leg] = 0
indicator_rest[idx_arm] = 0
def collision_penalty_skirt(va, na, vb, nb, eps=2e-3, kcollision=2500):

    vec = va[:, None] - vb[None]
    dist = torch.sum(vec**2, dim=-1)
    closest_vertices = torch.argmin(dist, dim=-1)
    indicator_leg_closest = indicator_leg[closest_vertices]
    
    vb = vb[closest_vertices]
    nb = nb[closest_vertices]
    
    distance_not_leg = (nb[~indicator_leg_closest]*(va[~indicator_leg_closest] - vb[~indicator_leg_closest])).sum(dim=-1)
    interpenetration_not_leg = torch.nn.functional.relu(eps - distance_not_leg)
    distance_leg = (nb[indicator_leg_closest]*(va[indicator_leg_closest] - vb[indicator_leg_closest])).sum(dim=-1)
    interpenetration_leg = torch.nn.functional.relu(eps - distance_leg)


    indicator_flip = torch.logical_and((na[indicator_leg_closest]*nb[indicator_leg_closest]).sum(dim=-1) < 0, distance_leg < 0)
    sign = torch.ones_like(interpenetration_leg)
    sign[indicator_flip] = -1
    interpenetration_leg = interpenetration_leg*sign

    interpenetration = (interpenetration_not_leg**3).sum() + (interpenetration_leg**3).sum()

    return interpenetration * kcollision