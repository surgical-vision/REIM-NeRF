# REIM-NeRF

Realistic endoscopic illumination modelling for NeRF-based data generation

## Overview

You can download a copy of the corresponding MICCAI-23 paper from [here](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_51)

**Important**
Model weights for every model in the paper have been uploaded.

pre-processed trajectory files to aid with rendering without needed to download an processed c3vd will be uploaded soon.
The repository contains code migrated from internal projects. Should you have any issues running the scripts or recreating results, please open an issue.

### Paper's Abstract

Expanding training and evaluation data is a major step towards building and deploying reliable localization and 3D reconstruction
techniques during colonoscopy screenings. However, training and evaluating pose and depth models in colonoscopy is hard as available datasets
are limited in size. This paper proposes a method for generating new pose
and depth datasets by fitting NeRFs in already available colonoscopy
datasets. Given a set of images, their associated depth maps and pose
information, we train a novel light source location-conditioned NeRF to
encapsulate the 3D and color information of a colon sequence. Then, we
leverage the trained networks to render images from previously unobserved camera poses and simulate different camera systems, effectively
expanding the source dataset. Our experiments show that our model is
able to generate RGB images and depth maps of a colonoscopy sequence
from previously unobserved poses with high accuracy.

## Recreating the paper's results

### Download the pre-trained models

#### Using the provided script

The following script to downloads and extracts all pretrained models used in the paper under 
`./ckpts/c3vd/main_results_iter15000_w270_h216/`. From the root directory of the repository, run:

```bash
python -m scripts/download_models.py
```

#### Download manually

We host additional files supporting the paper in UCL's research repository.
You can download a copy of all the pre-trained models from [here](https://rdr.ucl.ac.uk/articles/model/REIM-NeRF_pretrained_models_on_C3VD/24418297)

If you want to render C3VD sequences without training models, continue reading from [Render endoscopic sequences](#endering-c3vd-from-trained-models).

### Data pre-processing workflow

To train and test our models you need to have a copy of the registered videos of [C3VD](https://durrlab.github.io/C3VD/)
and based on it, generate the following:

- json files describing each dataset, including calibration parameters, file-paths and etc.
- distance maps ( like depth-maps but encoding distance in 3D from the camera center instead of just z distance).
- pixel masks, masking out edge pixels where calibration parameters are expected to be less accurate and therefore, depth and rgb information is best to be ignored during training.

1. Download a copy of all registered videos .zip files from C3VD and extract them under a single directory.
After downloading and extracting all datasets, your local copy directory tree should look like this:

    ```tree
    c2vd_registered_videos_raw
    ├── trans_t1_a_under_review
    │   └── ...
    ├── trans_t2_b_under_review
    │   ├── 0000_color.png
    │   ├── 0000_depth.tiff
    │   ├── 0000_normals.tiff
    │   ├── 0000_occlusion.png
    │   └── ...
    └── ...
    ```

2. Run the provided preprocessing script which will generate the training, evaluation and test datasets

    ```bash
    python -m scripts.pre-process_c3vd\
            {path_to_c3vd_registered_videos_raw dir}\
            {path_to_a_directory_to_store_the_data}
    ```

    The rest of the parameters should remain default.

After both steps are complete, the process dataset should have the following structure

```tree
processed
   ├── trans_t1_a_under_review
   │   ├── images
   │   │   ├── 0000_color.png
   │   │   ├── 0001_color.png
   │   │   └── ....
   │   ├── distmaps
   │   │   ├── 0000_color.png
   │   │   ├── 0001_color.png
   │   │   └── ....
   │   ├── rgb_mask.png
   │   ├── transforms_test.json
   │   ├── transforms_train.json
   │   ├── transforms_true_test.json
   │   └── transforms_val.json
   ├── trans_t1_a_under_review
   └── ....
```

- **images/**: contain a copy of the source rgb images.
- **distmaps/**:  contain the distance maps named after the rgb image they correspond to. Note, those files are different from the detphmaps provided with the dataset and encode ray distance expressed in (0,pi) scale instead of z distance expressed in (0,100mm) scale.
- **rgb_mask.png**: is used during both training and validation to mask out pixels in the periphery of the frames where calibration parameters are expected to be less accurate.
- **transforms_test.json**: information for every frame, we use this to re-render video sequences
- **transforms_train.json**: information for training frame
- **transforms_eval.json**: information for evaluation frames
- **transforms_true_test.json**: information for frames used to generate the paper's results. This json does not contain training frames.

### Training

We provide scripts to train all variants of models presented in the paper, across all C3VD sequences. This allows readers to recreate the ablation study presented in the paper.

1. Go through all the steps of the [Data pre-processing workflow](#data pre-processing workflow) sections
2. Modify `train_nerf.sh`, `train_nerf_depth.sh`, `train_nerf_plus_light-source.sh`, `train_reim-nerf.sh` under `REIM-NeRF/scripts/bash/c3vd/training`, by replacing the placeholder value of variable `dataset_root_dir` with the path of the root directory of the pre-processed C3VD dataset (generated in step 1).
3. Modify the above scripts to match your system GPU resources.
4. Run the appropriate script:
   - `train_all.sh`: trains all models. It runs the following 4 scripts in sequence
   - `train_nerf.sh`: trains only the vanilla nerf
   - `train_nerf_depf.sh`: trains only the vanilla nerf with sparse depth supervision
   - `train_nerf_plus_light-source.sh`: trains the light-source location conditioned model.
   - `train_reim-nerf.sh`: trains the light-source conditioned model with sparse depth supervision. This is our full model

### Rendering C3VD from trained models

The provided scripts assume will work with models and trajectories downloaded using the provided scripts. Be sure to modify them to point to either your version of pre-processed C3VD or your trained models.

1. Go through all the steps of the [Data pre-processing workflow](#data pre-processing workflow) sections. This is required because the rendering process relies on camera poses.
2. Modify `inference_nerf.sh`, `inference_nerf_depth.sh`, `inference_nerf_plus_light-source.sh`, `inference_reim-nerf.sh` under `REIM-NeRF/scripts/bash/c3vd/inference`, by replacing the placeholder value of variable `sequences_root_dir` with the path of the root directory of the pre-processed C3VD dataset (generated in step 1). Furthermore, replace the `checkpoints_root_dir` with the root directory of saved models for C3VD for each of the models.
3. Run the appropriate script:
   - `inference_all.sh`: run inference on C3VD with all models. This script runs the following 4 scripts in sequence
   - `inference_nerf.sh`: Inference with the original nerf model
   - `inference_nerf_depf.sh`: Inference with the original nerf model, trained with sparse depth supervision
   - `inference_nerf_plus_light-source.sh`: Inference with the light-source location conditioned model.
   - `inference_reim-nerf.sh`: Inference with the light-source location conditioned model, trained with sparse depth supervision. This is our full model

### Evaluate models trained on C3VD

1. Go through all the steps of the [Data pre-processing workflow](#data pre-processing workflow) sections.
2. Either train or download the pre-trained models(download script and links will be added shortly)
3. Modify `scripts/bash/c3vd/evaluate/evaluate_template.sh`, by changing the placeholders variables paths. `dataset_root_dir` should point to the pre-processed version of C3VD and `predictions_root_dir` should point to the root directory containing checkpoints for all c3vd dataset of a specific model variant.

## Acknowledgements

This research was funded, in whole, by the Wellcome/EPSRC
Centre for Interventional and Surgical Sciences (WEISS) [203145/Z/16/Z]; the
Engineering and Physical Sciences Research Council (EPSRC) [EP/P027938/1,
EP/R004080/1, EP/P012841/1]; the Royal Academy of Engineering Chair in Emerging Technologies Scheme, and Horizon 2020 FET (863146). For the purpose of open
access, the author has applied a CC BY public copyright licence to any author accepted
manuscript version arising from this submission.

We also thank [kwea123](https://github.com/kwea123), who open-sourced a [multi-GPU implementation of NeRF](https://github.com/kwea123/nerf_pl) upon which we build our approach.

### Citation

If you found this work usefull in your research, consider citing our paper.

``` bibtex
@inproceedings{psychogyios2023realistic,
  title={Realistic Endoscopic Illumination Modeling for NeRF-Based Data Generation},
  author={Psychogyios, Dimitrios and Vasconcelos, Francisco and Stoyanov, Danail},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={535--544},
  year={2023},
  organization={Springer}
}
```
