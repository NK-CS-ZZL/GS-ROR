#!/bin/bash

output_dir=$1
scene_list="angel bell cat horse luyu potion tbell teapot"
scene_list="angel"

for i in $scene_list; do
        python render.py --eval \
        -m ${output_dir}/${i} \
        --render_mode defer \
        --skip_train \
        --env_mode envmap
done
