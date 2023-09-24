# REIM-NeRF

Realistic endoscopic illumination modelling for NeRF-based data generation

## Overview

### Paper's Abstruct

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

### Data preparation

To train and test our models you need to have a copy of the registered videos of [C3VD](https://durrlab.github.io/C3VD/)
and process it to generate:

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
    python -m scripts.generate_reim_c3vd\
            {path to c3vd_registered_videos_raw dir}\
            {path to a directory to store the data}
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

### Evaluating Models

Will be added soon

### Training Models

Will be added soon
