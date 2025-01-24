#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"


##################### General parameters #####################

OUTPUT_FOLDER="${PSCRATCH}/suffix_decoding_artifact/rebuttal/results-backup-v3/suffix_decoding"
SUFFIX_DECODING_TRACES_FOLDER="${PWD}/../../suffix-tree-decoding/trace"
# model_name=meta-llama/Meta-Llama-3-70B-Instruct
model_name=meta-llama/Llama-3.1-70B-Instruct
NGPUS=8
NCPUS=16
FSIZE=76000
ZSIZE=200000
CSIZE=200000
MAX_SEQ_LEN=8000
max_spec_factor=4.0
tokens_per_batch=1024
batch_size=8
max_tree_depth=64
# comment these lines in for debugging
# model_name=meta-llama/Meta-Llama-3-8B-Instruct
# FSIZE=70000
# ZSIZE=30000
# CSIZE=60000
## or these
# model_name=meta-llama/Llama-3.2-1B-Instruct
# FSIZE=70000
# ZSIZE=22000
# CSIZE=100000
matching_strategies=(
    linear_token_path
    dynamic_token_tree
)
online_tree_update=(
    # true
    false
)

##################### Dataset parameters #####################
traces=(
    cortex
    spider
    wildchat
    magicoder
)
trace_files=(
    ${SUFFIX_DECODING_TRACES_FOLDER}/llama70b/cortex-llama3.1-70b.json
    ${SUFFIX_DECODING_TRACES_FOLDER}/llama70b/spider-llama3.1-70b.json
    ${SUFFIX_DECODING_TRACES_FOLDER}/llama70b/wildchat-llama3.1-70b.json
    ${SUFFIX_DECODING_TRACES_FOLDER}/llama70b/magicoder-llama3.1-70b.json
)

##################### Environment setup #####################
mkdir -p $OUTPUT_FOLDER
make -j
source set_python_envs.sh
# download all models and small models
python ../inference/utils/download_hf_model.py --half-precision-only $model_name ${small_model_names[@]}
export LEGION_BACKTRACE=1

##################### Main loop #####################
for i in "${!traces[@]}"; do
    trace=${traces[$i]}
    trace_file=${trace_files[$i]}
    if [ ! -f "$trace_file" ]; then
        echo "Trace file $trace_file does not exist. Skipping trace ${trace}."
        exit 1
    fi
    if [ "$trace" == "cortex" ]; then
        partitions=(
            QUESTION_SUGGESTION
            CATEGORIZATION
            FEATURE_EXTRACTION
            SQL_FANOUT1
            SQL_FANOUT2
            SQL_FANOUT3
            SQL_COMBINE
        )
    else
        partitions=(all)
    fi
    
    for partition_name in "${partitions[@]}"; do
    for k in "${!matching_strategies[@]}"; do
    for l in "${!online_tree_update[@]}"; do
        partition_name=${partitions[$i]}
        matching_strategy=${matching_strategies[$k]}
        otu=${online_tree_update[$l]}
        echo "Running trace '${trace}' partition '${partition_name}' with model '${model_name}', batch size ${batch_size}, and tokens per batch ${tokens_per_batch} with matching strategy ${matching_strategy}, online tree update ${otu}, and max tree depth ${max_tree_depth}"
        # create model name version where "/" is replaced with "-"
        model_name_=$(echo $model_name | tr / -)
        output_log_file="${OUTPUT_FOLDER}/${trace}_specinfer_${partition_name}_${model_name_}_${batch_size}_${matching_strategy}_otu-${otu}_max_tree_depth-${max_tree_depth}.out"
        output_csv_file="${OUTPUT_FOLDER}/${trace}_specinfer_${partition_name}_${model_name_}_${batch_size}_${matching_strategy}_otu-${otu}_max_tree_depth-${max_tree_depth}.csv"
        trace_output_file="${OUTPUT_FOLDER}/${trace}_ff_${partition_name}_${model_name_}.json"

        rm $output_log_file || true
        rm $output_csv_file || true

        otu_arg=""
        if [ "$otu" = true ]; then
            otu_arg="--disable-online-tree-update"
        fi

        time ./inference/suffix_decoding/suffix_decoding \
            -ll:gpu $NGPUS -ll:cpu $NCPUS -ll:util $NCPUS \
            -tensor-parallelism-degree $NGPUS \
            -ll:fsize $FSIZE -ll:zsize $ZSIZE -ll:csize $CSIZE \
            --fusion \
            --max-sequence-length $MAX_SEQ_LEN \
            --max-requests-per-batch $batch_size \
            --max-tokens-per-batch $tokens_per_batch \
            --max-output-length 900 \
            --matching-strategy $matching_strategy ${otu_arg} \
            --max-tree-depth $max_tree_depth \
            --max-spec-factor $max_spec_factor \
            -llm-model $model_name \
            -trace $trace_file -target-partition ${partition_name} \
            -trace-output-path ${trace_output_file} -output-file $output_log_file -csv-output-path $output_csv_file
    done
    done
    done
done