
output_dir=$1
scene_list="angel bell cat horse luyu potion tbell teapot"
for i in $scene_list; do
        python eval/metrics.py \
        --model_paths ${output_dir}/${i}
done
