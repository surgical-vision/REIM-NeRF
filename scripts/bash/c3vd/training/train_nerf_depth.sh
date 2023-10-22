#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH 


echo "Training NeRF with sparce depth supervision"


dataset_root_dir=/path_to_c3vd_dataset/
width=270
height=216
total_samples=15000
rgb_loss=L2
depth_loss=L1
init=glorot
dataset_type=reim_json

model_type=nerf
depth_ratio=0.03


for dataset_path in ${dataset_root_dir}/*
do 

    dataset_name=$(basename "$dataset_path")
    image_count=$(ls -l ${dataset_path}/images | wc -l)
    epochs=$(($total_samples/image_count))
    step1=$((epochs/2))
    step2=$((((epochs-step1)/2)+step1))

    python -m scripts.train \
    --dataset_name ${dataset_type} \
    --root_dir ${dataset_path} \
    --N_importance 64 --img_wh ${width} ${height} \
    --num_epochs ${epochs} --batch_size 1024 \
    --optimizer adam --lr 5e-4 \
    --num_gpus 4 \
    --lr_scheduler steplr --decay_step ${step1} ${step2} --decay_gamma 0.5 \
    --init_type ${init} \
    --variant ${model_type} \
    --rgb_loss ${rgb_loss} \
    --supervise_depth \
    --depth_loss ${depth_loss} \
    --depth_loss_levels all \
    --depth_ratio ${depth_ratio} \
    --exp_name c3vd/main_results_iter${total_samples}_w${width}_h${height}/${model_type}_depth_${depth_ratio}/${dataset_name}
# exp_name is {dataset_name}/{experiment_name}/{model_type}/{sub_dataset_name}
done