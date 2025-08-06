#!/bin/zsh

light_dir="data/tensoir_synthetic/env/high_res_envmaps_1k"
output_dir=$1
scene_list="helmet ball car coffee teapot toaster"

light_abbr="bridge"


for i in $scene_list; do
    for j in $light_abbr; do
        python relight.py \
        -m ${output_dir}/${i} \
        -s data/shiny_blender/${i} \
        --render_mode defer+split_sum \
        --skip_train \
        --transform tir \
        --envmap ${light_dir}/${j}.hdr
    done
done