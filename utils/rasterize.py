import os,sys
import numpy as np 
import torch

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    RasterizationSettings, 
    MeshRasterizer,  
    TexturesVertex,
)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def get_raster(render_res, scale, faces_per_pixel=2, sigma=1e-7):
    dis = 100.0
    mesh_y_center = 0.0
    cam_pos = torch.tensor([
                    (0, mesh_y_center, dis),
                    (0, mesh_y_center, -dis),
                ])
    R, T = look_at_view_transform(
        eye=cam_pos[[0]],
        at=((0, mesh_y_center, 0), ),
        up=((0, 1, 0), ),
    )

    cameras = FoVOrthographicCameras(
        device=device,
        R=R,
        T=T,
        znear=100.0,
        zfar=-100.0,
        max_y=100.0,
        min_y=-100.0,
        max_x=100.0,
        min_x=-100.0,
        scale_xyz=(scale * np.ones(3), ) * len(R),
    )

    raster_settings_hard = RasterizationSettings(
        image_size=render_res, 
        blur_radius=np.log(1.0 / 1e-4)*sigma,#1e-5, 
        faces_per_pixel=faces_per_pixel,#1, 
        max_faces_per_bin=500000,
        perspective_correct=False,
    )

    meshRas_hard = MeshRasterizer(cameras=cameras, raster_settings=raster_settings_hard)

    return meshRas_hard

def get_pix_to_face(verts_gar, faces_gar, verts_body, faces_body, raster):
    len_faces_gar = len(faces_gar)
    verts = torch.cat((verts_gar, verts_body), dim=0)
    faces = torch.cat((faces_gar, faces_body+len(verts_gar)), dim=0)
    mesh_py3d = Meshes(
            verts=[verts],   
            faces=[faces],
            textures=TexturesVertex(verts_features=torch.ones_like(verts[None]))
        )

    Fragments = raster(mesh_py3d)
    pix_to_face = Fragments.pix_to_face.squeeze()
    valid_pixel = torch.logical_and(pix_to_face > -1, pix_to_face < len_faces_gar)

    valid_faces = pix_to_face[valid_pixel]

    valid_faces = torch.unique(valid_faces.flatten())
    valid_vertices = torch.unique(faces_gar[valid_faces].flatten())
    return valid_faces, valid_vertices
