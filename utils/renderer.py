import os,sys
import numpy as np 
import cv2
import torch

from utils.readfile import load_pkl

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    OrthographicCameras,
    FoVOrthographicCameras,
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
    blending,
)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

from typing import NamedTuple, Sequence
class BlendParams_blackBG(NamedTuple):
    sigma: float = 1e-4
    gamma: float = 1e-4
    background_color: Sequence = (0.0, 0.0, 0.0)

class cleanShader(torch.nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs):

        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = blending.softmax_rgb_blend(texels, fragments, blend_params, znear=-256, zfar=256)

        return images

def get_transform(scale=80.0):
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
    transform = cameras.get_full_projection_transform()
    return transform

def get_render():
    render_res = 256
    dis = 100.0
    scale = 80.0
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
        blur_radius=0, 
        faces_per_pixel=1, 
        max_faces_per_bin=500000,
        perspective_correct=False,
    )

    meshRas_hard = MeshRasterizer(cameras=cameras, raster_settings=raster_settings_hard)

    renderer_textured_hard = MeshRenderer(
        rasterizer=meshRas_hard,
        shader=cleanShader(blend_params=BlendParams_blackBG())
    )

    return renderer_textured_hard

def verts_color(partsIdx):
    # 0: torsor 1
    # 1: R hand 2
    # 2: L hand 3
    # 3: L foot 4
    # 4: R foot 5
    # 5: R upper leg 6
    # 6: L upper leg 7
    # 7: R lower leg 5
    # 8: L lower leg 4
    # 9: L upper arm 8
    # 10: R upper arm 9
    # 11: L lower arm 3
    # 12: R lower arm 2
    # 13: head 10
    color_map = [1, 2, 3, 4, 5, 6, 7, 5, 4, 8, 9, 3, 2, 10]

    color = np.zeros((6890, 3))
    for i in range(14):
        color[partsIdx[i]] = color_map[i]
    return color

def render_body_seg(verts, faces_body, renderer_textured):
    partsIdx = load_pkl('../extra-data/parts14_vertex_idx.pkl')
    color = verts_color(partsIdx)
    color = torch.FloatTensor(color).cuda()/10
    textures = TexturesVertex(verts_features=color[None])
            
    with torch.no_grad():
        mesh_py3d = Meshes(
            verts=[verts],   
            faces=[faces_body],
            textures=textures
        )

        body_seg = renderer_textured(mesh_py3d)[0, :, :, :3].detach()
        body_seg = torch.round(torch.clamp(body_seg, 0, 1)*10)*25
        body_seg = body_seg.cpu().numpy().astype(np.uint8)

    return  body_seg


def vis_xyz(xyz):
    xyz_image = (xyz*255).astype(np.uint8)
    return xyz_image

def render_body_xyz(verts, faces_body, renderer_textured, vis_back=False):
    
    color = (verts.clone() + 1)/2
    if vis_back:
        verts[:, 0] *= -1
        verts[:, -1] *= -1
        faces_body = faces_body[:,[0,2,1]]
    textures = TexturesVertex(verts_features=color[None])
            
    with torch.no_grad():
        mesh_py3d = Meshes(
            verts=[verts],   
            faces=[faces_body],
            textures=textures
        )

        body_xyz = renderer_textured(mesh_py3d)[0, :, :, :3].detach()
        body_xyz = body_xyz.squeeze().cpu().numpy()
        body_xyz_image = vis_xyz(body_xyz.copy())
        
    if vis_back:
        body_xyz_image = np.fliplr(body_xyz_image)
        body_xyz = np.fliplr(body_xyz)

    body_xyz = body_xyz * 2 - 1

    return  body_xyz, body_xyz_image
            



        