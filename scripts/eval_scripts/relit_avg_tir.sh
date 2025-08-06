
output_dir=$1
scene_list="armadillo lego ficus hotdog"
for i in $scene_list; do
        python eval/cal_relit_avg.py \
        --root ${output_dir}/${i}/relight 
done
