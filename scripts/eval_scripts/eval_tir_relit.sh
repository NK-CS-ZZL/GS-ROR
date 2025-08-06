
output_dir=$1
scene_list="hotdog armadillo lego ficus"
for i in $scene_list; do
        python eval/metrics_relit_tir.py \
        --img_paths ${output_dir}/${i}/relight \
        --gt_paths data/tensoir_synthetic/${i} 
done
