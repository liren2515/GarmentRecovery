# Garment Recovery with Shape and Deformation Priors
<p align="center"><img src="figs/overview.png"></p>

This is the repo for [**Garment Recovery with Shape and Deformation Priors**](https://liren2515.github.io/page/prior/prior.html).

## Setup & Install
See [INSTALL.md](doc/INSTALL.md)

## Fitting
You can use the scripts under `./fit` to recover garment from the prepared images in `./fitting-data`:
```
cd fit
python fit_xxx.py # xxx is the type of the garment.
```

## Prepare your own data
If you want to prepare your own data for fitting, please check [DATA_PREPARE.md](doc/DATA_PREPARE.md)

## Citation
If you find our work useful, please cite it as:
```
@inproceedings{Li2023isp,
  author = {Li, Ren and Guillard, Benoit and Fua, Pascal},
  title = {{ISP: Multi-Layered Garment Draping with Implicit Sewing Patterns}},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2023}
}

@inproceedings{li2024garment,
  author = {Li, Ren and Dumery, Corentin and Guillard, Benoit and Fua, Pascal},
  title = {{Garment Recovery with Shape and Deformation Priors}},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2024}
}
```
