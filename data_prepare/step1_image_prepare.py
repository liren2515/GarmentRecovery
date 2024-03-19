##########################
# step 1.1
##########################
# get segmentaion from Self-Correction-Human-Parsing
'''
python simple_extractor.py --dataset lip --model-restore checkpoints/lip.pth --input-dir $ROOT_PATH/fitting-data/garment/processed/crop --output-dir $ROOT_PATH/fitting-data/garment/processed/segmentation
'''

##########################
# step 1.2
##########################
# insatll and get segmentaion from SAM
'''
pip install git+https://github.com/facebookresearch/segment-anything.git
'''


from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np 
import torch
import matplotlib.pyplot as plt
import sys, os

input_dir = './fitting-data/garment/images'
crop_dir = './fitting-data/garment/processed/crop'
econ_dir = './fitting-data/garment/processed/econ/png/'
output_dir = './fitting-data/garment/processed/segmentation'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
images = os.listdir(input_dir)

def show_anns(anns):
    np.random.seed(0)
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.85]])
        img[m] = color_mask
        print(ann['area'], ann['predicted_iou'], ann['stability_score'])
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


sam_checkpoint = "./segment-anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(model=sam, min_mask_region_area=1000)

for i in range(len(images)):
    image = os.path.join(crop_dir, '%s_crop.png'%images[i].split('.')[0])
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)

    sam_result = show_anns(masks)
    sam_result = (sam_result*255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, images[i].split('.')[0]+'-sam.png'), sam_result)

    SCHP_result = cv2.imread(os.path.join(output_dir, '%s_crop.png'%images[i].split('.')[0]))
    SCHP_result, silh = parse_segmentation(SCHP_result)
    sam_garment_label = assign_label_to_sam(sam_result[:,:,:3], SCHP_result)
    cv2.imwrite(os.path.join(output_dir, images[i].split('.')[0]+'-sam-labeled.png'), (sam_garment_label*60).astype(np.uint8))