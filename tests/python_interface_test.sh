#! /usr/bin/env bash
set -x
set -e


FF_HOME="$(realpath "${BASH_SOURCE[0]%/*}/..")"
export FF_HOME
# Edit the folder below if you did not build FlexFlow in $FF_HOME/build
BUILD_FOLDER="${FF_HOME}/build"
export BUILD_FOLDER

# Token to access private huggingface models (e.g. LLAMA-2)
HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-none}
if [[ "$HUGGINGFACE_TOKEN" != "none" ]]; then
    huggingface-cli login --token "$HUGGINGFACE_TOKEN"
fi

installation_status=${1:-"before-installation"}
echo "Running Python interface tests (installation status: ${installation_status})"
if [[ "$installation_status" == "before-installation" ]]; then
	# Check availability of flexflow modules in Python
	export PYTHONPATH="${FF_HOME}/python:${BUILD_FOLDER}/deps/legion/bindings/python:${PYTHONPATH}"
	export LD_LIBRARY_PATH="${BUILD_FOLDER}:${LD_LIBRARY_PATH}"
	python -c "import flexflow.core; import flexflow.serve as ff; exit()"
	unset PYTHONPATH
	unset LD_LIBRARY_PATH
	# Run simple python inference test
	export LD_LIBRARY_PATH="${BUILD_FOLDER}:${BUILD_FOLDER}/deps/legion/lib:${LD_LIBRARY_PATH}"
	export PYTHONPATH="${FF_HOME}/python:${BUILD_FOLDER}/deps/legion/bindings/python:${PYTHONPATH}"
	python "$FF_HOME"/inference/python/incr_decoding.py
	unset PYTHONPATH
	unset LD_LIBRARY_PATH
elif [[ "$installation_status" == "after-installation" ]]; then
	# Check availability of flexflow modules in Python
	python -c "import flexflow.core; import flexflow.serve as ff; exit()"
	# Run simple python inference test
	python "$FF_HOME"/inference/python/incr_decoding.py
else
	echo "Invalid installation status!"
	echo "Usage: $0 {before-installation, after-installation}"
	exit 1
fi
