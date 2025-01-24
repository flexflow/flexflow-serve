#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"


##################### General parameters #####################
OUTPUT_FOLDER="${PSCRATCH}/suffix_decoding_artifact/rebuttal/results-backup-v3/specinfer"
SUFFIX_DECODING_TRACES_FOLDER="${PWD}/../../suffix-tree-decoding/trace"
# model_name=meta-llama/Meta-Llama-3-70B-Instruct
model_name=meta-llama/Llama-3.1-70B-Instruct
small_model_names=(
    meta-llama/Llama-3.2-1B-Instruct
    meta-llama/Llama-3.1-8B-Instruct
)
NGPUS=8
NCPUS=16
FSIZE=76000
ZSIZE=200000
CSIZE=200000
MAX_SEQ_LEN=8000
tokens_per_batch=1024
batch_size=8
max_tree_depth=8
expansion_degree=3
# # comment these lines in for debugging
# model_name=meta-llama/Meta-Llama-3-8B-Instruct
# FSIZE=30000
# ZSIZE=90000
# CSIZE=120000


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
    for small_model_name in "${small_model_names[@]}"; do
        echo "Running trace '${trace}' partition '${partition_name}' with model '${model_name}', ssm '${small_model_name}', batch size ${batch_size}, and tokens per batch ${tokens_per_batch} with max tree depth ${max_tree_depth} and expansion degree ${expansion_degree}"
        # create model name version where "/" is replaced with "-"
        model_name_=$(echo $model_name | tr / -)
        small_model_name_=$(echo $small_model_name | tr / -)
        output_log_file="${OUTPUT_FOLDER}/${trace}_specinfer_${partition_name}_${model_name_}_${small_model_name_}_${batch_size}_${max_tree_depth}_${expansion_degree}.out"
        output_csv_file="${OUTPUT_FOLDER}/${trace}_specinfer_${partition_name}_${model_name_}_${small_model_name_}_${batch_size}_${max_tree_depth}_${expansion_degree}.csv"
        trace_output_file="${OUTPUT_FOLDER}/${trace}_ff_${partition_name}_${model_name_}_${small_model_name_}.json"
        
        rm $output_log_file || true
        rm $output_csv_file || true
        
        time ./inference/suffix_decoding/specinfer \
            -ll:gpu $NGPUS -ll:cpu $NCPUS -ll:util $NCPUS \
            -tensor-parallelism-degree $NGPUS \
            -ll:fsize $FSIZE -ll:zsize $ZSIZE -ll:csize $CSIZE \
            --fusion \
            --max-sequence-length $MAX_SEQ_LEN \
            --max-requests-per-batch $batch_size \
            --max-tokens-per-batch $tokens_per_batch \
            --max-output-length 900 \
            --max-tree-depth ${max_tree_depth} \
            --expansion-degree ${expansion_degree} \
            -llm-model $model_name -ssm-model $small_model_name \
            -trace $trace_file -target-partition ${partition_name} \
            -trace-output-path ${trace_output_file} -output-file $output_log_file -csv-output-path $output_csv_file
    done
    done
done

