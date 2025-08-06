output_dir=$1

sh scripts/eval_scripts/render_glossy.sh $output_dir
sh scripts/eval_scripts/eval_glossy.sh $output_dir