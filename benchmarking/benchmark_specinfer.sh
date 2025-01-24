#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"

# export BUILD_TYPE=Debug
# ../config/config.linux
make -j
source ./set_python_envs.sh
# reset

model_name=meta-llama/Llama-3.1-70B-Instruct
NGPUS=8
NCPUS=16
FSIZE=36000
ZSIZE=200000
CSIZE=100000

# comment these lines in for debugging
# model_name=meta-llama/Llama-3.1-8B-Instruct
# NGPUS=8
# FSIZE=36000
# ZSIZE=30000
# CSIZE=100000
######################################

small_model_names=(
    Zhuominc/Llama-3-330M
    meta-llama/Llama-3.2-1B-Instruct
    meta-llama/Llama-3.2-3B-Instruct
    meta-llama/Llama-3.1-8B-Instruct
)

MAX_SEQ_LEN=7000
tokens_per_batch=1024
max_tree_depth=8
expansion_degree=3

batch_sizes=(
    8
    4
)

request_per_second_values=(
    -1
    1
    2
    4
    8
)

dataset_name="sharegpt"
dataset_fp="../benchmarking/${dataset_name}.json"
partition_name="all"

export LEGION_BACKTRACE=1

# python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='meta-llama/Llama-3.1-70B-Instruct', allow_patterns='*.safetensors', max_workers=30)"
python ../inference/utils/download_hf_model.py --half-precision-only $model_name
for small_model_name in "${small_model_names[@]}"; do
    python ../inference/utils/download_hf_model.py --half-precision-only $small_model_name
done

for k in "${!request_per_second_values[@]}"; do
for j in "${!batch_sizes[@]}"; do
for i in "${!small_model_names[@]}"; do
    small_model_name=${small_model_names[$i]}
    batch_size=${batch_sizes[$j]}
    request_per_second=${request_per_second_values[$k]}
    
    echo "Running dataset ${dataset_fp} with model ${model_name}, draft model ${small_model_name}, batch size ${batch_size}, tokens per batch ${tokens_per_batch}, and request per second ${request_per_second}"
    # create model name version where "/" is replaced with "-"
    model_name_=$(echo $model_name | tr / -)
    small_model_name_=$(echo $small_model_name | tr / -)
    if [ $request_per_second -gt 0 ]; then
        rate=$request_per_second
    else
        rate="offline"
    fi
    log_fp="/usr/FlexFlow/inference/output/specinfer_llm_${model_name_}_ssm_${small_model_name_}_bz_${batch_size}_rate_${rate}_dataset_${dataset_name}.log"
    output_fp="/usr/FlexFlow/inference/output/specinfer_llm_${model_name_}_ssm_${small_model_name_}_bz_${batch_size}_rate_${rate}_dataset_${dataset_name}.json"
    metrics_fp="/usr/FlexFlow/inference/output/specinfer_llm_${model_name_}_ssm_${small_model_name_}_bz_${batch_size}_rate_${rate}_dataset_${dataset_name}.csv"
    rm $metrics_fp $output_fp $log_fp || true

    time ./inference/suffix_decoding/specinfer \
        -ll:gpu $NGPUS -ll:cpu $NCPUS -ll:util $NCPUS \
        -tensor-parallelism-degree $NGPUS \
        -ssm-tp-degree $NGPUS \
        -ll:fsize $FSIZE -ll:zsize $ZSIZE -ll:csize $CSIZE \
        --fusion \
        --max-sequence-length $MAX_SEQ_LEN \
        --max-requests-per-batch $batch_size \
        --max-tokens-per-batch $tokens_per_batch \
        --max-output-length 1024 \
        --max-tree-depth ${max_tree_depth} \
        --expansion-degree ${expansion_degree} \
        --request-per-second ${request_per_second} \
        -llm-model $model_name \
        -ssm-model $small_model_name \
        -trace ${dataset_fp} \
        -trace-output-path ${output_fp} \
        -csv-output-path $metrics_fp \
        -target-partition ${partition_name} \
        2>&1 | tee ${log_fp}
done
done
done