#! /usr/bin/env bash
set -x
set -e

cleanup() {
    rm -rf ~/.cache/flexflow/debug
}

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/.."

MODEL_NAME=${MODEL_NAME:-"goliaro/llama-3.2-1b-lora"}
BASE_MODEL_NAME=${BASE_MODEL_NAME:-"unsloth/Llama-3.2-1B-Instruct"}
MEMORY_PER_GPU=${MEMORY_PER_GPU:-14000}
ZCOPY_MEMORY=${ZCOPY_MEMORY:-40000}
TP_DEGREE=${TP_DEGREE:-4}
PP_DEGREE=${PP_DEGREE:-1}
FF_CACHE_PATH=${FF_CACHE_PATH:-"~/.cache/flexflow"}
FULL_PRECISION=${FULL_PRECISION:-false}
BFLOAT16_PRECISION=${BFLOAT16_PRECISION:-true}
FUSION=${FUSION:-false} # false because we save the debugging tensors in lora_linear.cc
LEARNING_RATE=${LEARNING_RATE:-0.001}
NUM_GPUS=$((TP_DEGREE * PP_DEGREE))

# Token to access private huggingface models (e.g. LLAMA-2)
HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-none}
if [[ "$HUGGINGFACE_TOKEN" != "none" ]]; then
    huggingface-cli login --token "$HUGGINGFACE_TOKEN"
fi

# Clean up before test (just in case)
cleanup

# Create test prompt file
mkdir -p ./inference/prompt
echo '["Two things are infinite: "]' > ./inference/prompt/peft.json
echo '["“Two things are infinite: the universe and human stupidity; and I'\''m not sure about the universe.”"]' > ./inference/prompt/peft_dataset.json


# Create output folder
mkdir -p ./inference/output

# Enable backtrace in case we run into a segfault or assertion failure
export LEGION_BACKTRACE=1

# Download test model
python ./inference/utils/download_peft_model.py "${MODEL_NAME}"

if [ "$FULL_PRECISION" = "true" ]; then full_precision_flag="--use-full-precision"; else full_precision_flag=""; fi
if [ "$BFLOAT16_PRECISION" = "true" ]; then bfloat16_precision_flag="--use-bfloat16-precision"; else bfloat16_precision_flag=""; fi
if [ "$FUSION" = "true" ]; then fusion_flag="--fusion"; else fusion_flag=""; fi

# Run PEFT in Huggingface to get ground truth tensors
eval python ./tests/peft/hf_finetune.py --peft-model-id "${MODEL_NAME}" --save-peft-tensors "${full_precision_flag}" "${bfloat16_precision_flag}" -lr "${LEARNING_RATE}"

# Python test
echo "Python test"
json_config=$(cat <<-END
    {
        "num_gpus": ${NUM_GPUS},
        "memory_per_gpu": ${MEMORY_PER_GPU},
        "zero_copy_memory_per_node": ${ZCOPY_MEMORY},
        "num_cpus": 4,
        "legion_utility_processors": 4,
        "data_parallelism_degree": 1,
        "tensor_parallelism_degree": ${TP_DEGREE},
        "pipeline_parallelism_degree": ${PP_DEGREE},
        "enable_peft": true,
        "inference_debugging": true,
        "fusion": ${FUSION},
        "refresh_cache": false,
        "base_model": "${BASE_MODEL_NAME}",
        "inference_peft_model_id": "${MODEL_NAME}",
        "finetuning_peft_model_id": "${MODEL_NAME}",
        "cache_path": "${FF_CACHE_PATH:-}",
        "full_precision": ${FULL_PRECISION},
        "prompt": "",
        "finetuning_dataset": "./inference/prompt/peft_dataset.json",
        "output_file": "",
        "max_requests_per_batch": 1,
        "max_seq_length": 128,
        "max_tokens_per_batch": 128,
        "max_concurrent_adapters": 1
    }
END
)
echo "$json_config" > /tmp/peft_config.json
python ./inference/python/ff_peft.py -config-file /tmp/peft_config.json
# Check alignment
python ./tests/peft/peft_alignment_test.py -m "${MODEL_NAME}" -tp "${TP_DEGREE}" -lr "${LEARNING_RATE}"

# C++ test
echo "C++ test"

if [ "$BFLOAT16_PRECISION" = "true" ]; then bfloat16_precision_flag="--use-bf16-precision"; else bfloat16_precision_flag=""; fi
./build/inference/peft/peft \
    -ll:gpu ${NUM_GPUS} -ll:cpu 4 -ll:util 4 \
    -tensor-parallelism-degree "${TP_DEGREE}" \
    -ll:fsize "${MEMORY_PER_GPU}" -ll:zsize "${ZCOPY_MEMORY}" \
    --max-requests-per-batch 1 \
    --max-sequence-length 128 \
    --max-tokens-per-batch 128 \
    -llm-model "${BASE_MODEL_NAME}" \
    -finetuning-dataset ./inference/prompt/peft_dataset.json \
    -peft-model "$MODEL_NAME" \
    -enable-peft \
    ${full_precision_flag} \
    ${fusion_flag} \
    ${bfloat16_precision_flag} \
    --inference-debugging
     


# Print succeess message
echo ""
echo "PEFT tests passed!"
echo ""

# Cleanup after the test
cleanup
