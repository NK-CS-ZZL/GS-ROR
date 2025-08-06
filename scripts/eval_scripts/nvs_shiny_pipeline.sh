output_dir=$1

sh scripts/eval_scripts/render_shiny.sh $output_dir
sh scripts/eval_scripts/eval_shiny.sh $output_dir