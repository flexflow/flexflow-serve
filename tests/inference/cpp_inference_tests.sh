#! /usr/bin/env bash
set -x
set -e

# Cd into root directory of repo
cd "${BASH_SOURCE[0]%/*}/../.."

# Function to launch specinfer with flags from a JSON config file.
run_cpp_inference() {
  local config_file="$1"

  # Check that a config file was provided and exists
  if [[ -z "$config_file" ]]; then
    echo "Usage: launch_specinfer <config_file.json>"
    return 1
  fi
  if [[ ! -f "$config_file" ]]; then
    echo "Config file not found: $config_file"
    return 1
  fi

  # Check for mandatory keys in the config file (including model_name)
  for req in num_gpus memory_per_gpu zero_copy_memory_per_node llm_model; do
    if ! jq -e --arg key "$req" 'has($key)' "$config_file" >/dev/null; then
      echo "Error: Missing required parameter: $req"
      return 1
    fi
  done

  # Download the model using the model_name key
  llm_model=$(jq -r '.llm_model' "$config_file")
  echo "Downloading model: $llm_model"
  if ! python3 ./inference/utils/download_hf_model.py "$llm_model" --half-precision-only; then
    echo "Error: Failed to download model $llm_model"
    return 1
  fi

  # Declare an associative array mapping config keys to system flags
  declare -A ff_arg_to_sysarg=(
    ["num_gpus"]="-ll:gpu"
    ["memory_per_gpu"]="-ll:fsize"
    ["zero_copy_memory_per_node"]="-ll:zsize"
    ["cpu_memory_per_node"]="-ll:csize"
    ["num_cpus"]="-ll:cpu"
    ["legion_utility_processors"]="-ll:util"
    ["data_parallelism_degree"]="-data-parallelism-degree"
    ["tensor_parallelism_degree"]="-tensor-parallelism-degree"
    ["pipeline_parallelism_degree"]="-pipeline-parallelism-degree"
    ["offload"]="-offload"
    ["offload_reserve_space_size"]="-offload-reserve-space-size"
    ["use_4bit_quantization"]="--4bit-quantization"
    ["use_8bit_quantization"]="--8bit-quantization"
    ["peft_support_mode"]="--peft_support_mode"
    ["profiling"]="--profiling"
    ["benchmarking"]="--benchmarking"
    ["inference_debugging"]="--inference-debugging"
    ["fusion"]="--fusion"
    ["llm_model"]="-llm-model"
    # ["cache_path"]="-cache-folder"
    ["full_precision"]="--use-full-precision"
    ["prompt"]="-prompt"
    ["output_file"]="-output-file"
    ["max_seq_length"]="--max-sequence-length"
    ["max_requests_per_batch"]="--max-requests-per-batch"
    ["max_length"]="--max-length"
    ["max_tokens_per_batch"]="--max-tokens-per-batch"
    ["log_instance_creation"]="--log-instance-creation"
    ["disable_control_replication"]="--disable-control-replication"
    ["dataset"]="--dataset"
    ["enable_inplace_optimizations"]="--enable-inplace-optimization"
  )

  # Build the command line arguments array
  args=()

  # Process keys in the order they appear in the JSON file.
  # Use jq to output tab-separated key-value pairs.
  while IFS=$'\t' read -r key value; do
    # Skip "model_name" (already used for downloading).
    if [[ "$key" == "model_name" ]]; then
      continue
    fi

    # Process the "ssms" block specially.
    if [[ "$key" == "ssms" ]]; then
      # For each element in the "ssms" array, download and add the -ssm-model flag.
      ssm_models=$(jq -r '.ssms[] | .ssm_model' "$config_file")
      for ssm_model in $ssm_models; do
        echo "Downloading ssm_model: $ssm_model"
        if ! python3 ./inference/utils/download_hf_model.py "$ssm_model" --half-precision-only; then
          echo "Error: Failed to download ssm_model $ssm_model"
          return 1
        fi
        args+=("-ssm-model" "$ssm_model")
      done
      continue
    fi

    # If the key is recognized in the mapping, add the corresponding flag.
    if [[ -n "${ff_arg_to_sysarg[$key]}" ]]; then
      flag="${ff_arg_to_sysarg[$key]}"
      if [[ "$value" == "true" ]]; then
        args+=("$flag")
      elif [[ "$value" == "false" ]]; then
        continue
      else
        args+=("$flag" "$value")
      fi
    fi
  done < <(jq -r 'to_entries[] | "\(.key)\t\(.value|tostring)"' "$config_file")

  # Determine which executable to run based on file contents:
  # Use ./incr_dec if the file contains "incr_dec", else if "spec_infer" is found use ./specinfer.
  if [[ $config_file == *"incr_dec"* ]]; then
    executable="./build/inference/incr_decoding/incr_decoding"
  elif [[ $config_file == *"spec_infer"* ]]; then  
    executable="./build/inference/spec_infer/spec_infer"
  else
    echo "Error: Config file does not specify a valid mode (incr_dec or spec_infer)"
    return 1
  fi

  # Launch the chosen program with the constructed arguments
  $executable "${args[@]}"
}


############## Create prompt ################################
# Clean up before test
rm -rf inference/prompt inference/output inference/inf_test_configs || true
# Create test prompt file
mkdir -p ./inference/prompt
echo '["Three tips for staying healthy are: "]' > ./inference/prompt/test.json
# Create output folder
mkdir -p ./inference/output

############## Run inference in flexflow-serve ##############

echo "Running inference in flexflow-serve (C++)..."

# Generate test configs
python ./tests/inference/generate_inf_test_configs.py

# Loop through .json files in the ./inference/inf_test_configs dir 
for file in ./inference/inf_test_configs/*.json; do
    # Run script
    run_cpp_inference "$file"
done

##############  Run inference in HuggingFace ##############

echo "Running inference in huggingface..."

model_names=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "facebook/opt-6.7b"
    "facebook/opt-125m"
)
for model_name in "${model_names[@]}"; do
    # set model_name_ to the content of model_name after the first "/", transformed into lowercase
    model_name_=$(echo "$model_name" | cut -d'/' -f2 | tr '[:upper:]' '[:lower:]')
    python ./tests/inference/huggingface_inference.py \
        --model-name "$model_name" \
        --max-length 255 \
        --prompt-file "${PWD}/inference/prompt/test.json" \
        --output-file "${PWD}/inference/output/huggingface_$model_name_.json"
done

##############  Check alignment between results ##############
echo "Checking alignment of results..."
pytest -v ./tests/inference/test_inference_output.py
