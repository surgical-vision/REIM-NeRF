#!/bin/bash

parent_dir="$(dirname "${0}")"
sh $parent_dir/inference_nerf.sh
sh $parent_dir/inference_nerf_depth.sh
sh $parent_dir/inference_nerf_plus_light-source.sh # our extension to nerf
sh $parent_dir/inference_reim-nerf.sh # our full model

