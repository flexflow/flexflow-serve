#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"

MODEL_NAME="JackFram/llama-160m"
PEFT_MODEL_NAME="goliaro/llama-160m-lora"
NGPUS=1
NCPUS=16
FSIZE=30000
ZSIZE=20000

MAX_SEQ_LEN=2900
MAX_TOKENS_PER_BATCH=512
BATCH_SIZE=8
MAX_TRAINING_STEPS=1000

OUTPUT_FOLDER="/usr/flexflow-serve/inference/output"
LOG_FILE="/usr/flexflow-serve/inference/output/test.log"

reset
make -j install

# python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download(repo_id=\"${MODEL_NAME}\", allow_patterns=\"*.safetensors\", max_workers=30)"
# python ../inference/utils/download_hf_model.py $MODEL_NAME
# python ../inference/utils/download_peft_model.py $PEFT_MODEL_NAME

export LEGION_BACKTRACE=1
rm $LOG_FILE $OUTPUT_FILE || true

# export FF_USE_FLASHINFER=0

# gdb -ex run --args 
./inference/incr_decoding/incr_decoding \
    -ll:cpu $NCPUS -ll:gpu $NGPUS -ll:util $NCPUS \
    -ll:fsize $FSIZE -ll:zsize $ZSIZE \
    -llm-model $MODEL_NAME  \
    -tensor-parallelism-degree $NGPUS \
    -output-folder $OUTPUT_FOLDER \
    --max-requests-per-batch $BATCH_SIZE \
    --max-tokens-per-batch $MAX_TOKENS_PER_BATCH \
    --max-sequence-length $MAX_SEQ_LEN \
    --max-training-steps $MAX_TRAINING_STEPS \
    2>&1 | tee $LOG_FILE