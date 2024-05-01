
import trimesh
import torch
import numpy as np

from networks import SDF, convnext, diffusion
from utils.isp_cut import create_uv_mesh
from smpl_pytorch.body_models import SMPL
from utils.smpl_utils import infer_smpl

def init_uv_mesh(x_res=128, y_res=128, garment='Skirt'):
    uv_vertices, uv_faces = create_uv_mesh(x_res, y_res, debug=False)
    mesh_uv = trimesh.Trimesh(uv_vertices, uv_faces, validate=False, process=False)
    uv_vertices = torch.FloatTensor(uv_vertices).cuda()
    edges = torch.LongTensor(mesh_uv.edges).cuda()

    pattern_mean = np.load('../extra-data/pattern-mean-%s.npz'%garment)
    verts_uv_cano_f_mean = torch.FloatTensor(pattern_mean['verts_uv_cano_f_mean']).cuda()
    verts_uv_cano_b_mean = torch.FloatTensor(pattern_mean['verts_uv_cano_b_mean']).cuda()
    '''
    if garment == 'Skirt':
        verts_uv_cano_f_mean[:,1] += 0.1
        verts_uv_cano_b_mean[:,1] += 0.1
    '''
    
    verts_uv_cano_mean = [verts_uv_cano_f_mean, verts_uv_cano_b_mean]

    diffusion_pattern_mean = np.load('../extra-data/diffusion-pattern-mean-%s.npz'%garment)
    weight_f = torch.FloatTensor(diffusion_pattern_mean['weight_f']).cuda()
    weight_b = torch.FloatTensor(diffusion_pattern_mean['weight_b']).cuda()
    return mesh_uv, uv_vertices, uv_faces, edges, verts_uv_cano_mean, weight_f, weight_b


def init_smpl_sever(gender='f', model_path='../smpl_pytorch'):
    smpl_server = SMPL(model_path=model_path,
                            gender=gender,
                            use_hands=False,
                            use_feet_keypoints=False,
                            dtype=torch.float32).cuda()

    pose = torch.zeros(1, 72).cuda()
    beta = torch.zeros(1, 10).cuda()
    pose = pose.reshape(24,3)
    pose[1, 2] = .15
    pose[2, 2] = -.15
    pose = pose.reshape(-1).unsqueeze(0)
    w, tfs, verts_zero, pose_offsets, shape_offsets = infer_smpl(pose, beta, smpl_server)
    Rot_rest = torch.einsum('nk,kij->nij', w.squeeze(), tfs.squeeze()) 
    pose_offsets_rest = pose_offsets.squeeze()
    return smpl_server, Rot_rest, pose_offsets_rest

def load_model(numG=400, garment='Skirt'):
    rep_size = 32
    if garment == 'Skirt':
        num_labels = 3
    elif garment == 'Shirt' or garment == 'Jacket':
        num_labels = 4
    elif garment == 'Trousers':
        num_labels = 4
    else:
        raise Exception('%s is not included!!!!!!'%garment)
    model_sdf_f = SDF.SDF2branch_deepSDF(d_in=2+rep_size, d_out=1+num_labels, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
    model_sdf_b = SDF.SDF2branch_deepSDF(d_in=2+rep_size, d_out=1+num_labels, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
    model_rep = SDF.learnt_representations(rep_size=rep_size, samples=numG).cuda()
    model_atlas_f = SDF.SDF(d_in=2+rep_size, d_out=3, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
    model_atlas_b = SDF.SDF(d_in=2+rep_size, d_out=3, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()

    model_sdf_f.load_state_dict(torch.load('../checkpoints/sdf_f_%s.pth'%garment))
    model_sdf_b.load_state_dict(torch.load('../checkpoints/sdf_b_%s.pth'%garment))
    model_rep.load_state_dict(torch.load('../checkpoints/rep_%s.pth'%garment))
    model_atlas_f.load_state_dict(torch.load('../checkpoints/atlas_f_%s.pth'%garment))
    model_atlas_b.load_state_dict(torch.load('../checkpoints/atlas_b_%s.pth'%garment))
    latent_codes = model_rep.weights.detach()
    
    extractor = convnext.ConvNeXtExtractor(n_stages=4).cuda()
    extractorBody = convnext.ConvNeXtExtractorCustom(in_channel=4, n_stages=4).cuda()
    featuror = convnext.FeatureNetwork_xyz(context_dims=(96, 192, 384, 768), ave=False, cat_xyz=True).cuda()
    field = convnext.MLP(d_in=featuror.feature_dim*4, d_out=4, width=400, depth=9, gaussian=True, skip_layer=[5]).cuda()

    extractor.load_state_dict(torch.load('../checkpoints/extractor_%s.pth'%garment))
    extractorBody.load_state_dict(torch.load('../checkpoints/extractorBody_%s.pth'%garment))
    featuror.load_state_dict(torch.load('../checkpoints/featuror_%s.pth'%garment))
    field.load_state_dict(torch.load('../checkpoints/field_%s.pth'%garment))

    extractor.eval()
    extractorBody.eval()
    featuror.eval()
    field.eval()

    diffusion_skin = diffusion.skip_connection(d_in=3, width=512, depth=8, d_out=6890, skip_layer=[]).cuda()
    diffusion_skin.load_state_dict(torch.load('../checkpoints/diffusion_skin.pth'))
    diffusion_skin.eval()
    
    model_cnn_regressor = [extractor, extractorBody, featuror, field, diffusion_skin]
    model_isp = [model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b]
    return model_isp, latent_codes, model_cnn_regressor