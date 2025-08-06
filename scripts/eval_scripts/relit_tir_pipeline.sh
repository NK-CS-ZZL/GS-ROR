output_dir=$1

sh scripts/eval_scripts/relit_tir.sh $output_dir
sh scripts/eval_scripts/eval_tir_relit.sh $output_dir
sh scripts/eval_scripts/relit_avg_tir.sh $output_dir