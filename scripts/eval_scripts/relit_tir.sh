#!/bin/bash

light_dir="data/tensoir_synthetic/env/high_res_envmaps_2k/"
output_dir=$1
scene_list="lego armadillo"

light_list="bridge city fireplace forest night"

for i in $scene_list; do
    for j in $light_list; do
        python relight.py \
        -m $output_dir/${i} \
        --render_mode defer+split_sum \
        --skip_train \
        --transform tir \
        --envmap data/tensoir_synthetic/env/high_res_envmaps_2k/${j}.hdr
    done
done
