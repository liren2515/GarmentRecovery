##########################
# step 0
##########################
# build and run ECON

'''
python -m apps.infer -cfg ./configs/econ.yaml -in_dir $ROOT_PATH/fitting-data/garment/images/ -out_dir $ROOT_PATH/fitting-data/garment/processed/
'''


import os, sys
import cv2

# change ROOT_PATH to where you put your images
ROOT_PATH = '.'

# extract cropped images
input_dir = './fitting-data/garment/images/'
econ_dir = './fitting-data/garment/processed/econ/png/'
output_dir = './fitting-data/garment/processed/crop'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
images = os.listdir(input_dir)
for i in range(len(images)):
    image = os.path.join(econ_dir, '%s_crop.png'%images[i].split('.')[0])
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    image = image[:, :512]
    cv2.imwrite(os.path.join(output_dir, '%s_crop.png'%images[i].split('.')[0]), image)
