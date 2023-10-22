#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

python -m scripts.evaluate_c3vd \
    --dataset_root_dir /path_to_c3vd_dataset/ \
    --predictions_root_dir /path_to_directory_containing_the_output_of_inference/ \