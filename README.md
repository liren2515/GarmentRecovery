# Garment Recovery with Shape and Deformation Priors
<p align="center"><img src="figs/overview.png"></p>

This is the repo for [**Garment Recovery with Shape and Deformation Priors**](https://liren2515.github.io/page/prior/prior.html).

## Setup & Install
See [INSTALL.md](doc/INSTALL.md)

## Inference
For garment generation:
```
python infer_isp.py --which tee/pants/skirt --save_path tmp --save_name skirt --res 256 --idx_G 0
```

For layering inference:
```
python infer_layering.py
```

## Fitting
For fitting ISP to 3D garment mesh in rest pose:
```
python fitting_3D_mesh.py --which tee/pants/skirt --save_path tmp --save_name skirt-fit --res 256
```

For fitting ISP to images:
```
python fitting_image.py
```
The example files are under `./extra-data/fitting-sample/`, including the segmentation mask `mask.png` and the SMPL parameters `mocap.pkl`. We use [Self-Correction-Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) to produce garment masks, and [frankmocap](https://github.com/facebookresearch/frankmocap) to estimate SMPL parameters.

## Citation
If you find our work useful, please cite it as:
```
@inproceedings{li2024garment,
  author = {Li, Ren and Dumery, Corentin and Guillard, Benoit and Fua, Pascal},
  title = {{Garment Recovery with Shape and Deformation Priors}},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2024}
}
```
