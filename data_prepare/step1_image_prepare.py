##########################
# step 1
##########################
# build and run ECON

'''
###build_normal.sh
build a100 with 'conda install -c conda-forge libstdcxx-ng=12'
directly run on colab...
python -m apps.infer -cfg ./configs/econ.yaml -in_dir /cvlabdata2/home/ren/cloth-from-image/fitting-data/open-shirt/images/ -out_dir /cvlabdata2/home/ren/cloth-from-image/fitting-data/open-shirt/processed/
'''
'''
import os, sys
import cv2
# extract cropped images
input_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/open-shirt/images/'
econ_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/open-shirt/processed/econ/png/'
output_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/open-shirt/processed/crop'
input_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/skirt/images/'
econ_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/skirt/processed/econ/png/'
output_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/skirt/processed/crop'
input_dir = '/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video/images/'
econ_dir = '/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video/processed/econ/png/'
output_dir = '/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video/processed/crop'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
images = os.listdir(input_dir)
for i in range(len(images)):
    image = os.path.join(econ_dir, '%s_crop.png'%images[i].split('.')[0])
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    image = image[:, :512]
    cv2.imwrite(os.path.join(output_dir, '%s_crop.png'%images[i].split('.')[0]), image)
sys.exit()
'''
##########################
# step 2
##########################
# get segmentaion from Self-Correction-Human-Parsing
'''
switch to conda pytorch3d 
python simple_extractor.py --dataset lip --model-restore checkpoints/lip.pth --input-dir /cvlabdata2/home/ren/cloth-from-image/fitting-data/open-shirt/processed/crop --output-dir /cvlabdata2/home/ren/cloth-from-image/fitting-data/open-shirt/processed/segmentation
'''

##########################
# step 3
##########################
# insatll and get segmentaion from SAM
'''
pip install git+https://github.com/facebookresearch/segment-anything.git
'''

'''
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np 
import torch
import matplotlib.pyplot as plt
import sys, os

input_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/open-shirt/images'
crop_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/open-shirt/processed/crop'
econ_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/open-shirt/processed/econ/png/'
output_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/open-shirt/processed/segmentation'
input_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/skirt/images'
crop_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/skirt/processed/crop'
econ_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/skirt/processed/econ/png/'
output_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/skirt/processed/segmentation'

input_dir = '/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video/images'
crop_dir = '/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video/processed/crop'
econ_dir = '/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video/processed/econ/png/'
output_dir = '/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video/processed/segmentation'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
images = os.listdir(input_dir)

def show_anns(anns):
    np.random.seed(0)
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    #ax = plt.gca()
    #ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.85]])
        img[m] = color_mask
        print(ann['area'], ann['predicted_iou'], ann['stability_score'])
    #ax.imshow(img)

    return img


def parse_segmentation(segmentaion):
    seg = segmentaion.copy()[:,:,0]*0
    silh = seg.copy()
    idx_tee = (segmentaion[:,:,0] == 128) * (segmentaion[:,:,1] == 0) * (segmentaion[:,:,2] == 128)
    idx_jacket = (segmentaion[:,:,0] == 128) * (segmentaion[:,:,1] == 128) * (segmentaion[:,:,2] == 128)
    idx_pants = (segmentaion[:,:,0] == 0) * (segmentaion[:,:,1] == 0) * (segmentaion[:,:,2] == 192)
    idx_dress = (segmentaion[:,:,0] == 128) * (segmentaion[:,:,1] == 0) * (segmentaion[:,:,2] == 64)
    idx_silh = (segmentaion[:,:,0] > 0) + (segmentaion[:,:,1] > 0) + (segmentaion[:,:,2] > 0)
    seg[idx_tee] = 1
    seg[idx_jacket] = 2
    seg[idx_pants] = 3
    seg[idx_dress] = 4
    silh[idx_silh] = 1

    
    idx_dress_long = (segmentaion[:,:,0] == 128) * (segmentaion[:,:,1] == 128) * (segmentaion[:,:,2] == 0)
    seg[idx_dress_long] = 4
    return seg, silh

def parse_sam(sam_result):
    vec = sam_result.reshape(-1,3).tolist()
    vec = list(set([tuple(v) for v in vec]))
    num_labels = len(vec)
    sam_new = np.zeros_like(sam_result)[:,:,0].astype(int)
    
    for i in range(len(vec)):
        label = vec[i]
    
        idx = (sam_result[:,:,0] == label[0]) * (sam_result[:,:,1] == label[1]) * (sam_result[:,:,2] == label[2])
        sam_new[idx] = i

    return sam_new, num_labels

def overlap_ratios(sam_result, SCHP_result, sam_label_i, thresh_valid=0.5):

    num_SCHP_labels = SCHP_result.max()
    ratios = np.zeros((num_SCHP_labels+1,))

    for i in range(num_SCHP_labels+1):
        mask_SCHP = SCHP_result == i
        mask_sam = sam_result == sam_label_i

        mask_sam_intersection = mask_sam*mask_SCHP
        ratios[i] = mask_sam_intersection.sum()/mask_sam.sum()

    ratios[ratios<thresh_valid] = 0

    label = np.argmax(ratios)
    ratio_max = np.max(ratios)

    if ratio_max == 0:
        return None
    else:
        return label

def assign_label_to_sam(sam_result, SCHP_result):

    sam_new, num_labels = parse_sam(sam_result)

    sam_garment_label = np.zeros_like(sam_result)

    for i in range(num_labels):
        label = overlap_ratios(sam_new, SCHP_result, i, thresh_valid=0.7)
        if label is not None:
            sam_garment_label[sam_new==i] = label

    return sam_garment_label


sam_checkpoint = "/scratch/cvlab/home/ren/code/Others/segment-anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(model=sam, min_mask_region_area=1000)

for i in range(len(images)):
    #i = 27-1
    image = os.path.join(crop_dir, '%s_crop.png'%images[i].split('.')[0])
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    #print(image.shape)
    #cv2.imwrite('../tmp/image.png',image)
    #sys.exit()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)

    sam_result = show_anns(masks)
    sam_result = (sam_result*255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, images[i].split('.')[0]+'-sam.png'), sam_result)
    #print(sam_result.shape)
    #sys.exit()

    SCHP_result = cv2.imread(os.path.join(output_dir, '%s_crop.png'%images[i].split('.')[0]))
    #print(os.path.join(output_dir, '%s_crop.png'%images[i].split('.')[0]))
    #print(SCHP_result.shape)
    SCHP_result, silh = parse_segmentation(SCHP_result)
    sam_garment_label = assign_label_to_sam(sam_result[:,:,:3], SCHP_result)
    cv2.imwrite(os.path.join(output_dir, images[i].split('.')[0]+'-sam-labeled.png'), (sam_garment_label*60).astype(np.uint8))

'''

###### point guided SAM

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import cv2
import numpy as np 
import torch
import matplotlib.pyplot as plt
import sys, os

input_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/open-shirt/images'
crop_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/open-shirt/processed/crop'
econ_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/open-shirt/processed/econ/png/'
output_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/open-shirt/processed/segmentation'
input_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/skirt/images'
crop_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/skirt/processed/crop'
econ_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/skirt/processed/econ/png/'
output_dir = '/cvlabdata2/home/ren/cloth-from-image/fitting-data/skirt/processed/segmentation'

input_dir = '/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video/images'
crop_dir = '/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video/processed/crop'
econ_dir = '/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video/processed/econ/png/'
output_dir = '/scratch/cvlab/home/ren/code/cloth-from-image/fitting-data/video/processed/segmentation'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
images = os.listdir(input_dir)

def show_anns(anns):
    np.random.seed(0)
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    #ax = plt.gca()
    #ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.85]])
        img[m] = color_mask
        print(ann['area'], ann['predicted_iou'], ann['stability_score'])
    #ax.imshow(img)

    return img

def parse_sam(sam_result):
    vec = sam_result.reshape(-1,3).tolist()
    vec = list(set([tuple(v) for v in vec]))
    num_labels = len(vec)
    sam_new = np.zeros_like(sam_result)[:,:,0].astype(int)
    
    for i in range(len(vec)):
        label = vec[i]
    
        idx = (sam_result[:,:,0] == label[0]) * (sam_result[:,:,1] == label[1]) * (sam_result[:,:,2] == label[2])
        sam_new[idx] = i

    return sam_new, num_labels

def match_sam(sam_result, bbox_result):

    mask_bbox = bbox_result == 255

    sam_new, num_labels = parse_sam(sam_result)

    IoU_max = -1
    i_max = -1
    for i in range(num_labels):

        mask_sam = sam_new == i

        mask_sam_intersection = mask_sam*mask_bbox
        IoU = mask_sam_intersection.sum()/((mask_sam+mask_bbox)>0).sum()

        if IoU > IoU_max:
            IoU_max = IoU
            i_max = i
            #print(i_max)

    return sam_new == i_max


sam_checkpoint = "/scratch/cvlab/home/ren/code/Others/segment-anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(model=sam, min_mask_region_area=1000)

for i in range(len(images)):

    image = os.path.join(crop_dir, '%s_crop.png'%images[i].split('.')[0])
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    #print(image.shape)
    #cv2.imwrite('../tmp/image.png',image)
    #sys.exit()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    #input_point = np.array([[256, 300],[240, 300],[270, 300]])
    #input_label = np.array([1,1,1])
    input_box = np.array([125, 200, 375, 500])
    input_point = np.array([[256, 300]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=False,
    )
    #print(masks.shape, type(masks))
    #sys.exit()

    masks = masks[0]
    sam_result_bbox = (masks*255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, images[i].split('.')[0]+'-sam-bbox.png'), sam_result_bbox)

    masks = mask_generator.generate(image)
    sam_result = show_anns(masks)
    sam_result = (sam_result*255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, images[i].split('.')[0]+'-sam.png'), sam_result)
    #print(sam_result.shape)
    #sys.exit()

    sam_garment_label = match_sam(sam_result[:,:,:3], sam_result_bbox)
    cv2.imwrite(os.path.join(output_dir, images[i].split('.')[0]+'-sam-labeled.png'), (sam_garment_label*255).astype(np.uint8))

    #sys.exit()