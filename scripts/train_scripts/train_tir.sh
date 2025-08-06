#!/bin/bash

root_dir="data/tensoir_synthetic/"
list="armadillo lego ficus hotdog"

for i in $list; do
python train.py --eval \
-w \
--render_mode defer+split_sum \
-s ${root_dir}${i} \
-m outputs/relit/tir/${i}/ \
-w --sh_degree -1 \
--lambda_predicted_normal 0.2 \
--lambda_zero_one 0.2 \
--envmap_res 512 \
--env_mode envmap \
--enable_vis_from_iteration 30001 \
--port 12993 \
--lambda_brdf_smoothness 0.01 \
--pbr_from_iter 0 \
--lambda_light_reg 0.0 \
--zero_one_use_gt \
--sdf_from_iter 1100 \
--batchify \
--batch_size 1024 \
--sdf_mode TensoSDF \
--sdf_init_iters 5000 \
--lambda_normal_sdf2g 0.05 \
--lambda_depth_sdf2g 0.05 \
--sdf_remove_from_iter 15000 \
--iterations 24000 \
--metallic 0 

done