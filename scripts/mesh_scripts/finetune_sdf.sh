root_dir=$1
list="angel bell cat horse luyu potion tbell teapot"

for i in $list; do
python finetune_sdf.py \
-m ${root_dir}/${i}  --render_mode split_sum+defer \
--mesh_dir finetuned \
--sdf_mode TensoSDF --zero_one_use_gt \
--lambda_normal_g2sdf 10. --lambda_normal_sdf2g 0. \
--lambda_gaussian 0.001 --lambda_tv_sdf 0.1 \
--lambda_mask 5  --lr_decay_iters 6000 \
--gridSize 400 --load_iteration 24000
done