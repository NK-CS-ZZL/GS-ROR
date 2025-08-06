output_dir=$1

sh scripts/eval_scripts/relit_glossy.sh $output_dir
sh scripts/eval_scripts/eval_glossy_relit.sh $output_dir
sh scripts/eval_scripts/relit_avg_glossy.sh $output_dir