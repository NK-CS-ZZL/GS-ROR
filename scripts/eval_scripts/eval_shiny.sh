
output_dir=$1
scene_list="ball car coffee helmet teapot toaster"
for i in $scene_list; do
        python eval/metrics.py \
        --model_paths ${output_dir}/${i}
done
