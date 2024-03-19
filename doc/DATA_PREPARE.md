# Prepare your own data for fitting
We provide the scripts in `./data_prepare` to show how to prepare the date required by our fitting method.

## Step 0 - ECON
Suppose you have images put at `./fitting-data/garment/images`. You first need to run [ECON](https://github.com/YuliangXiu/ECON) to get the normal and SMPL parameter estimations for the images. Detailed instructions can be found at ... 

Once you have the results of ECON with the folder name of `econ`, run
```
mkdir ./fitting-data/garment/processed
```
Then copy `econ` to `./fitting-data/garment/processed`. The folder hierarchy should be
```
./fitting-data/garment
└── images
└── econ
│   └── BNI
│   └── obj
|   └── png
│   └── vid
```
Then run
```
python ./data_prepare/step0_run_ECON.py
```

## Step 1 - Segmentation
We use [SAM](https://github.com/facebookresearch/segment-anything) to get the segmentaion masks and [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) to assign sematic labels for the segmentation, respectively. 

First, you can follow the instruction of [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) to install it and download the LIP checkpoint. Then you can use the following cmd to get the sematic segmentations.
```
python simple_extractor.py --dataset lip --model-restore checkpoints/lip.pth --input-dir $ROOT_PATH/fitting-data/garment/processed/crop --output-dir $ROOT_PATH/fitting-data/garment/processed/segmentation
```

Second, install [SAM](https://github.com/facebookresearch/segment-anything) and download the checkpoint of `vit_h` model.
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
Then, change the value of `sam_checkpoint` to the path where you store the `vit_h` checkpoint in `./data_prepare/step0_run_ECON.py`, and run 
```
python ./data_prepare/step0_run_ECON.py
```
Note that sometimes the segmentaion results can be bad... You need to manually check them unless you have a better (reliable) model for garment segmentation.

