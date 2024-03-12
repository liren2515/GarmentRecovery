import cv2
import numpy as np
from utils import renderer

def process_image(image, size=256):

    image = cv2.resize(image, (size, size), interpolation = cv2.INTER_LINEAR)
    
    IMG_NORM_MEAN = [0.485, 0.456, 0.406]
    IMG_NORM_STD = [0.229, 0.224, 0.225]

    image = image.astype(np.float32)/255
    image[:,:, 0] = (image[:,:, 0] - IMG_NORM_MEAN[0])/IMG_NORM_STD[0]
    image[:,:, 1] = (image[:,:, 1] - IMG_NORM_MEAN[1])/IMG_NORM_STD[1]
    image[:,:, 2] = (image[:,:, 2] - IMG_NORM_MEAN[2])/IMG_NORM_STD[2]

    return image

def process_seg(image):

    image = image.astype(np.float32)/255
    image= (image - 0.5)/0.5

    return image

def prepare_input(normal_econ_512, mask_sam, body_smpl, body_renderer, result_path, save_img=True):
    
    normal_econ_512 = vid['normal_F'][0].permute(1,2,0).cpu().numpy()
    normal_econ_512 = ((normal_econ_512*0.5+0.5)*255).astype(np.uint8)

    mask_normal = normal_econ_512*mask_sam[:,:,None]
    cv2.imwrite(os.path.join(result_path, 'mask_normal.png'), mask_normal[:,:,[2,1,0]])

    body_seg_f = renderer.render_body_seg(torch.FloatTensor(body_smpl.vertices).cuda(), torch.LongTensor(body_smpl.faces).cuda(), body_renderer)[:,:,[0]]

    verts = body_smpl.vertices.copy()
    verts[:, 0] *= -1
    verts[:, -1] *= -1
    body_seg_b = renderer.render_body_seg(torch.FloatTensor(verts).cuda(), torch.LongTensor(body_smpl.faces).cuda(), body_renderer)[:,:,[0]]
    body_seg_b = np.fliplr(body_seg_b)
    cv2.imwrite(os.path.join(result_path, 'body_seg_f.png'), body_seg_f)
    cv2.imwrite(os.path.join(result_path, 'body_seg_b.png'), body_seg_b)

    body_xyz_f, body_xyz_f_image = renderer.render_body_xyz(torch.FloatTensor(body_smpl.vertices).cuda(), torch.LongTensor(body_smpl.faces).cuda(), body_renderer, vis_back=False)
    body_xyz_b, body_xyz_b_image = renderer.render_body_xyz(torch.FloatTensor(body_smpl.vertices).cuda(), torch.LongTensor(body_smpl.faces).cuda(), body_renderer, vis_back=True)
    cv2.imwrite(os.path.join(result_path, 'body_xyz_f.png'), body_xyz_f_image)
    cv2.imwrite(os.path.join(result_path, 'body_xyz_b.png'), body_xyz_b_image)

    normal_img = process_image(normal_econ)
    body_seg_f = process_seg(body_seg_f)
    body_seg_b = process_seg(body_seg_b)
    images = np.transpose(normal_img, (2,0,1))
    images_body = np.concatenate((body_seg_f, body_xyz_f, body_seg_b, body_xyz_b), axis=-1)
    images_body = np.transpose(images_body, (2,0,1))
    images = torch.FloatTensor(images).unsqueeze(0).cuda()
    images_body = torch.FloatTensor(images_body).unsqueeze(0).cuda()

    return images, images_body