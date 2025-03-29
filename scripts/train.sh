#!/bin/bash


scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools

dropouts=(0.0 0.2 0.4 0.6 0.8)  # Five different dropout values

log_dir="$base/logs"  # Directory to store logs
chmod +w $log_dir

mkdir -p $log_dir

num_threads=4
device="cuda"

SECONDS=0

for dropout in "${dropouts[@]}"; do
    model_name="model_dropout_${dropout}"
    log_file="$log_dir/${model_name}.log"
    
    echo "Training $model_name with dropout=$dropout..."
    echo "Log file path: $log_file"
    (cd $tools/pytorch-examples/word_language_model && python main.py \
        --data $data/bible \
        --emsize 256 \
        --nhid 256 \
        --dropout $dropout \
        --cuda \
        --epochs 40 \
        --log_perplexity $log_file \
        --save $models/$model_name.pt
    )
done

echo "time taken:"
echo "$SECONDS seconds"