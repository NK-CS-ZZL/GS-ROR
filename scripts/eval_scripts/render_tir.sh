#!/bin/bash

output_dir=$1
scene_list="armadillo lego ficus hotdog"

for i in $scene_list; do
        python render.py \
        -m ${output_dir}/${i} \
        --render_mode defer \
        --skip_train 
done
