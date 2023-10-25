#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH 

echo "Rendering C3VD using vanilla nerf trained with sparce depth supervision"

width=270
height=216
model_type=nerf
depth_ratio=0.03

# get the absolute path of the pretrained models
root_dir="$(readlink -m "${0}")"
for _ in {0..4}; do
    root_dir="$(dirname "$root_dir")"
done

checkpoints_root_dir=$root_dir/ckpts/c3vd/main_results_iter15000_w270_h216/nerf_${model_type}_${depth_ratio}

sequences_root_dir=$data/c3vd_processed

for dataset_path in ${sequences_root_dir}/*
do
    dataset_name=$(basename "$dataset_path")
    checkpoint_directory=${checkpoints_root_dir}/${dataset_name}

    latest_checkpoint_name=$(ls -Art "$checkpoint_directory"|tail -n 1)
    checkpoint_path=${checkpoint_directory}/${latest_checkpoint_name}
    echo $checkpoint_path
    echo $dataset_path

    python -m scripts.inference \
    --root_dir ${dataset_path} \
    --dataset_name snerf_json \
    --img_wh ${width} ${height} --N_importance 64 \
    --variant ${model_type} \
    --save_depth \
    --split test \
    --ckpt_path ${checkpoint_path} \
    --fps 30 \
    --scene_name c3vd/main_results_iter${total_samples}_w${width}_h${height}/${model_type}_depth_${depth_ratio}/${dataset_name}
    # scene_name is {dataset_name}/{experiment_name}/{model_type}/{sub_dataset_name}

done
