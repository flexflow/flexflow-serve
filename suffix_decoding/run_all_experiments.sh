#! /usr/bin/env bash
set -x
set -e

# Usage (from main repo dir): 
#   nohup ./suffix_decoding/run_all_experiments.sh &

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Check existence of traces folder
SUFFIX_DECODING_TRACES_FOLDER="${PWD}/../../suffix-tree-decoding/trace/llama70b" # assume suffix-tree-decoding is cloned in the same directory as flexflow-serve
# Check that the SUFFIX_DECODING_TRACES_FOLDER exists, and throw an error if it doesn't
if [ ! -d "$SUFFIX_DECODING_TRACES_FOLDER" ]; then
    echo "The SUFFIX_DECODING_TRACES_FOLDER does not exist. Please clone the suffix-tree-decoding repository in the same directory as flexflow-serve."
    exit 1
fi

# Inflate/convert the 25k traces
bash ${SUFFIX_DECODING_TRACES_FOLDER}/get25k_traces.sh

# Run all experiments
./benchmark_suffix_tree.sh
./benchmark_specinfer.sh
