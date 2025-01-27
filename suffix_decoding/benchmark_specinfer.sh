#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"


##################### General parameters #####################
OUTPUT_FOLDER="${PWD}/../inference/output/specinfer"
SUFFIX_DECODING_TRACES_FOLDER="${PWD}/../../suffix-tree-decoding/trace/llama70b" # assume suffix-tree-decoding is cloned in the same directory as flexflow-serve
# model_name=meta-llama/Meta-Llama-3-70B-Instruct
model_name=meta-llama/Llama-3.1-70B-Instruct
small_model_names=(
    meta-llama/Llama-3.2-1B-Instruct
    # meta-llama/Llama-3.1-8B-Instruct
)
NGPUS=8
NCPUS=16
FSIZE=76000
ZSIZE=200000
CSIZE=200000
MAX_SEQ_LEN=8200
max_output_length=3500
tokens_per_batch=1024
batch_size=8
max_tree_depth=8
expansion_degree=3
ssm_tp_degrees=(
    # 8
    1
)
# # comment these lines in for debugging
# model_name=meta-llama/Meta-Llama-3-8B-Instruct
# FSIZE=30000
# ZSIZE=90000
# CSIZE=120000
# max_trace_requests=10


##################### Dataset parameters #####################
traces=(
    cortex
    spider
    magicoder
    wildchat
)
trace_files=(
    ${SUFFIX_DECODING_TRACES_FOLDER}/cortex-llama3.1-70b.json
    ${SUFFIX_DECODING_TRACES_FOLDER}/spider-llama3.1-70b.json
    ${SUFFIX_DECODING_TRACES_FOLDER}/magicoder25k-llama3.1-70b.json
    ${SUFFIX_DECODING_TRACES_FOLDER}/wildchat25k-llama3.1-70b.json
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
    for small_model_name in "${small_model_names[@]}"; do
    for ssm_tp_degree in "${ssm_tp_degrees[@]}"; do
        echo "Running trace '${trace}' partition '${partition_name}' with model '${model_name}', ssm '${small_model_name}' (ssm tp degree=${ssm_tp_degree}), batch size ${batch_size}, and tokens per batch ${tokens_per_batch} with max tree depth ${max_tree_depth} and expansion degree ${expansion_degree}"
        # create model name version where "/" is replaced with "-"
        model_name_=$(echo $model_name | tr / -)
        small_model_name_=$(echo $small_model_name | tr / -)
        output_log_file="${OUTPUT_FOLDER}/${trace}_${partition_name}_${model_name_}_${small_model_name_}_ssm-tp-degree-${ssm_tp_degree}_${batch_size}_${max_tree_depth}_${expansion_degree}.out"
        output_csv_file="${OUTPUT_FOLDER}/${trace}_${partition_name}_${model_name_}_${small_model_name_}_ssm-tp-degree-${ssm_tp_degree}_${batch_size}_${max_tree_depth}_${expansion_degree}.csv"
        trace_output_file="${OUTPUT_FOLDER}/${trace}_${partition_name}_${model_name_}_${small_model_name_}_ssm-tp-degree-${ssm_tp_degree}_${batch_size}_${max_tree_depth}_${expansion_degree}.json"
        
        rm $output_log_file || true
        rm $output_csv_file || true
        
        time ./inference/suffix_decoding/specinfer \
            -ll:gpu $NGPUS -ll:cpu $NCPUS -ll:util $NCPUS \
            -tensor-parallelism-degree $NGPUS -ssm-tp-degree $ssm_tp_degree \
            -ll:fsize $FSIZE -ll:zsize $ZSIZE -ll:csize $CSIZE \
            --fusion \
            --max-sequence-length $MAX_SEQ_LEN \
            --max-requests-per-batch $batch_size \
            --max-tokens-per-batch $tokens_per_batch \
            --max-output-length $max_output_length \
            --max-tree-depth ${max_tree_depth} \
            --expansion-degree ${expansion_degree} \
            -llm-model $model_name -ssm-model $small_model_name \
            -trace $trace_file -target-partition ${partition_name} \
            -trace-output-path ${trace_output_file} -output-file $output_log_file -csv-output-path $output_csv_file \
            ${max_trace_requests:+--max-trace-requests ${max_trace_requests}}

    done
    done
    done
done

