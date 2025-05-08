# Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Running Instructions:
- To run this FastAPI application, make sure you have FastAPI and Uvicorn installed.
- Save this script as 'fastapi_incr.py'.
- Run the application using the command: `uvicorn fastapi_incr:app --reload --port PORT_NUMBER`
- The server will start on `http://localhost:PORT_NUMBER`. Use this base URL to make API requests.
- Go to `http://localhost:PORT_NUMBER/docs` for API documentation.
"""


from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import flexflow.serve as ff
from flexflow.core import *
from flexflow.type import (
    OptimizerType
)
import uvicorn
import json, os, argparse
from types import SimpleNamespace
from typing import Optional, List, Dict
import time
from huggingface_hub import hf_hub_download, HfFolder
from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
import time

# Initialize FastAPI application
app = FastAPI()

# Define the request model
class PromptRequest(BaseModel):
    prompt: str

# data models
class Message(BaseModel):
    role: str
    content: str

# For inference request
class ChatCompletionRequest(BaseModel):
    max_new_tokens: Optional[int] = 1024
    messages: List[Message]
    peft_model_id: Optional[str] = None  # Add optional PEFT model ID for adapter

# For finetuning request
class FinetuneRequest(BaseModel):
    token: str
    peft_model_id: str
    dataset_option: str
    dataset: Optional[List[str]] = None
    dataset_name: Optional[str] = None
    config_name: Optional[str] = None
    selected_split: Optional[str] = None
    selected_column: Optional[str] = None
    lora_rank: int = 16
    lora_alpha: int = 16
    target_modules: Optional[List[str]] = ["down_proj"]
    learning_rate: float = 1e-5
    optimizer_type: str
    momentum: float
    weight_decay: float
    nesterov: bool = False
    max_training_epochs: int = 2

# For uploading model request
class UploadModelRequest(BaseModel):
    token: str
    peft_model_id: str
    upload_peft_model_id: str
    private: bool = False

# Global variable to store the LLM model
llm = None

OPTIMIZER_TYPE_MAP = {
    "SGD": OptimizerType.OPTIMIZER_TYPE_SGD,
    "Adam": OptimizerType.OPTIMIZER_TYPE_ADAM,
}

# Registry to store CFFI objects mapped to string IDs
adapter_registry = {}

def get_configs():

    # Fetch configuration file path from environment variable
    config_file = os.getenv("CONFIG_FILE", "")

    # Load configs from JSON file (if specified)
    if config_file:
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found.")
        try:
            with open(config_file) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print("JSON format error:")
            print(e)
    else:
        # Define sample configs
        ff_init_configs = {
            # required parameters
            "num_gpus": 4,
	        "memory_per_gpu": 34000,
            "zero_copy_memory_per_node": 40000,
            "log_instance_creation": True,
            # optional parameters
            "num_cpus": 4,
            "legion_utility_processors": 8,
            "data_parallelism_degree": 1,
            "tensor_parallelism_degree": 1,
            "pipeline_parallelism_degree": 4,
            "offload": False,
            "offload_reserve_space_size": 8 * 1024, # 8GB
            "use_4bit_quantization": False,
            "use_8bit_quantization": False,
            "peft_support_mode": ff.PeftSupportMode.COSERVING,
            "profiling": False,
            "benchmarking": False,
            "inference_debugging": False,
            "fusion": True,
        }
        llm_configs = {
            # required parameters
            "llm_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            # optional parameters
            "cache_path": os.environ.get("FF_CACHE_PATH", ""),
            "refresh_cache": False,
            "full_precision": False,
            "prompt": "",
            "output_file": "",
            "max_requests_per_batch": 128,
            "max_seq_length": 3000,
            "max_tokens_per_batch": 128,
            "max_concurrent_adapters": 4,
            "num_kv_cache_slots": 100000,
        }
        # Merge dictionaries
        ff_init_configs.update(llm_configs)
        return ff_init_configs
    

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    global llm

    # Initialize your LLM model configuration here
    configs_dict = get_configs()
    configs = SimpleNamespace(**configs_dict)
    ff.init(configs_dict)

    ff_data_type = (
        ff.DataType.DT_FLOAT if configs.full_precision else ff.DataType.DT_HALF
    )
    llm = ff.LLM(
        configs.llm_model,
        data_type=ff_data_type,
        cache_path=configs.cache_path,
        refresh_cache=configs.refresh_cache,
        output_file=configs.output_file,
    )

    generation_config = ff.GenerationConfig(
        do_sample=False, temperature=0.9, topp=0.8, topk=1
    )
    llm.compile(
        generation_config,
        max_requests_per_batch=configs_dict.get("max_requests_per_batch", 1)
        + 1,  # +1 for the finetuning request
        max_seq_length=configs_dict.get("max_seq_length", 256),
        max_tokens_per_batch=configs_dict.get("max_tokens_per_batch", 128),
        num_kv_cache_slots=configs_dict.get("num_kv_cache_slots", -1),
        max_concurrent_adapters=configs_dict.get("max_concurrent_adapters", 1)
        + 1,  # +1 for the finetuning request
        peft_support_mode=ff.PeftSupportMode.COSERVING,
    )
    llm.start_server()


# API endpoint to register the lora adapter
@app.post("/register_adapter/")
async def register_adapter(peft_model_name: str = Query(..., description="PEFT model name")):
    """
    Attempt to register a LoRA adapter for inference.
    Download weights and validate the base model if not already done.
    """
    try:
        if llm is None:
            raise HTTPException(status_code=503, detail="LLM model is not initialized.")

        if not peft_model_name:
            raise ValueError("PEFT model name is required.")

        # Validate and download the LoRA adapter weights if needed
        llm.download_peft_adapter_if_needed(peft_model_name)

        lora_inference_config = ff.LoraLinearConfig(
            llm.cache_path,
            peft_model_name.lower(), # convert the peft_model_name to lowercase 
            base_model_name_or_path=llm.model_name,
        )

        # Try to get the peft model id
        try:
            # If the adapter is already registered, retrieve its ID
            peft_model_id = llm.get_ff_peft_id(lora_inference_config)
            print(f"Adapter {peft_model_name} already registered with ID {peft_model_id}.")

        except ValueError as e:
            # If the adapter is not registered, register it and then retrieve its ID
            print(f"Adapter {peft_model_name} not registered. Registering now...")
            llm.register_peft_adapter(lora_inference_config)
            peft_model_id = llm.get_ff_peft_id(lora_inference_config)

        # Save the mapping of PEFT model name to CFFI object
        adapter_registry[peft_model_name] = peft_model_id

        return {"peft_model_id": str(peft_model_id), "status": "success"}

    except Exception as e:
        error_message = f"Error during adapter registration: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)


# API endpoint to register the lora adapter
@app.post("/chat/completions/")
async def chat_completions(request: ChatCompletionRequest):
    """
    Generate a response, optionally using a registered LoRA adapter.
    """
    try:
        if llm is None:
            raise HTTPException(status_code=503, detail="LLM model is not initialized.")

        print("received request:", request)

        # Use the PEFT adapter if specified
        if request.peft_model_id:
            # Retrieve the CFFI object using the PEFT model name
            peft_model_cffi = adapter_registry.get(request.peft_model_id)
            prompt = llm._LLM__chat2prompt([message.dict() for message in request.messages])

            request = Request(
                ff.RequestType.REQ_INFERENCE,
                prompt=prompt,
                max_new_tokens=request.max_new_tokens,
                peft_model_id=peft_model_cffi,
            )
            result = llm.generate(request)[0].output_text.decode("utf-8")
        else:
            result = llm.generate(
                [message.dict() for message in request.messages],
                max_new_tokens=request.max_new_tokens,
            )[0].output_text.decode("utf-8")

        print("Returning response:", result)
        return {"response": result, "status": "success"}

    except Exception as e:
        error_message = f"Failed to generate response: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)


# API endpoint for getting dataset config names
@app.get("/get_dataset_configs/")
async def get_dataset_configs(dataset_name: str):
    """
    Given a dataset name, return the available config names.
    """
    try:
        config_names = get_dataset_config_names(dataset_name)
        if config_names == ['default']: # No configs in dataset
            config_names = []
        return {"dataset_name": dataset_name, "config_names": config_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching dataset config names: {str(e)}")

# API endpoint for getting dataset splits
@app.get("/get_dataset_splits/")
async def get_dataset_splits(dataset_name: str, config_name: Optional[str] = None):
    """
    Given a dataset name, return the available splits (e.g., train, validation, test).
    """
    try:
        splits = get_dataset_split_names(dataset_name, config_name=config_name)
        return {"dataset_name": dataset_name, "splits": splits}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching dataset splits: {str(e)}")

# API endpoint for getting dataset available columns
@app.get("/get_dataset_columns/")
async def get_dataset_columns(dataset_name: str, split: str, config_name: Optional[str] = None):
    """
    Given a dataset name and split, return available columns.
    """
    try:
        dataset = load_dataset(dataset_name, data_dir=config_name, split=split)
        return {"dataset_name": dataset_name, "columns": dataset.column_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching dataset columns: {str(e)}")

# API endpoint for finetuning request
@app.post("/finetuning/")
async def finetune(request: FinetuneRequest):
    """
    Endpoint to start LoRA finetuning based on the provided parameters.
    """
    try:
        if llm is None:
            raise HTTPException(status_code=503, detail="LLM model is not initialized.")

        print("received request:", request)

        llm.download_peft_adapter_if_needed(request.peft_model_id)

        if request.optimizer_type not in OPTIMIZER_TYPE_MAP:
            raise ValueError(f"Unsupported optimizer type: {request.optimizer_type}")

        optimizer_type = OPTIMIZER_TYPE_MAP[request.optimizer_type]

        # Prepare LoRA configuration for finetuning
        lora_finetuning_config = ff.LoraLinearConfig(
            llm.cache_path,
            request.peft_model_id.lower(),
            trainable=True,
            init_lora_weights=True,
            base_model_name_or_path=llm.model_name,
            optimizer_type=optimizer_type,
            target_modules=request.target_modules,
            optimizer_kwargs={
                "learning_rate": request.learning_rate,
                "momentum": request.momentum,
                "weight_decay": request.weight_decay,
                "nesterov": request.nesterov,
            },
        )

        llm.register_peft_adapter(lora_finetuning_config)

        cache_folder = os.path.expanduser(llm.cache_path)
        # Load the dataset
        file_path = None
        total_entries = None
        remaining_entries = None
        if request.dataset_option == "Upload JSON":
            dataset_dir = os.path.join(cache_folder, "datasets", "uploaded")
            os.makedirs(dataset_dir, exist_ok=True)

            file_path = os.path.join(dataset_dir, "dataset.json")
            with open(file_path, "w") as f:
                json.dump(request.dataset, f)
        elif request.dataset_option == "Hugging Face Dataset":
            dataset_dir = os.path.join(cache_folder, "datasets", "huggingface")
            os.makedirs(dataset_dir, exist_ok=True)

            json_subdir = os.path.join(dataset_dir, request.dataset_name)
            os.makedirs(json_subdir, exist_ok=True)
            
            from transformers import AutoTokenizer
            # Load dataset from Hugging Face
            dataset_info = f"{request.dataset_name}/{request.config_name}/{request.selected_split}" \
                if request.config_name else f"{request.dataset_name}/{request.selected_split}"
            json_filename = f"{dataset_info.replace('/', '_')}.json"

            print(f"Loading dataset: {dataset_info}")
            dataset = load_dataset(request.dataset_name, data_dir=request.config_name, split=request.selected_split)

            total_entries = len(dataset)
            print(f"Found {total_entries} entries in the dataset.")

            max_length = 10000 # Change if needed

            # Load a pre-trained tokenizer.
            tokenizer = AutoTokenizer.from_pretrained(request.peft_model_id)

            # Function to tokenize text and add a token count.
            def tokenize_count(example):
                # Tokenize the selected field
                tokens = tokenizer.tokenize(example[request.selected_column])
                # Save the number of tokens to a new field
                example["token_count"] = len(tokens)
                return example

            # Apply the function to each example in the dataset.
            tokenized_dataset = dataset.map(tokenize_count)
            # Filter entries with token_count less than max_length.
            filtered_dataset = tokenized_dataset.filter(lambda example: example["token_count"] < min(max_length, llm.max_seq_length))
            # Extract the original selected field from the filtered examples.
            text_list = filtered_dataset[request.selected_column]

            remaining_entries = len(filtered_dataset)
            print(f"Filtering out entries longer than {llm.max_seq_length} tokens...")
            print(f"{remaining_entries} entries remaining after filtering.")

            # Save the text list to a JSON file.
            file_path = os.path.join(json_subdir, json_filename)
            with open(file_path, "w") as f:
                json.dump(text_list, f, indent=2)
            
        print(f"Dataset saved to {file_path}")

        # Create finetuning request
        finetuning_request = ff.Request(
            ff.RequestType.REQ_FINETUNING,
            peft_model_id=llm.get_ff_peft_id(lora_finetuning_config),
            dataset_filepath=file_path,
            max_training_epochs=request.max_training_epochs,
        )

        results = llm.generate(finetuning_request)
        print(f"Finish fine-tuning")

        return {"results": results, "status": "success", "total_entries": total_entries, "remaining_entries": remaining_entries}

    except Exception as e:
        error_message = f"Error during finetuning: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)


# API endpoint for uploading model request
@app.post("/upload_peft_model/")
async def upload_peft_model(request: UploadModelRequest):
    """
    Endpoint to upload the fine-tuned PEFT model to Hugging Face Hub.
    """
    try:
        if llm is None:
            raise HTTPException(status_code=503, detail="LLM model is not initialized.")

        from transformers import AutoModelForCausalLM
        from peft import get_peft_model
        import torch
        import numpy as np

        cache_folder = os.path.expanduser(llm.cache_path)
        lora_config_filepath = os.path.join(
            cache_folder, 
            "finetuned_models", 
            request.peft_model_id.lower(), 
            "config", 
            "ff_config.json"
        )
        
        TIMEOUT_SECONDS = 30
        start_time = time.time()
        while not os.path.exists(lora_config_filepath):
            if time.time() - start_time > TIMEOUT_SECONDS:
                raise TimeoutError(f"Timeout: {lora_config_filepath} not found after {TIMEOUT_SECONDS} seconds.")
            time.sleep(0.5)  # Check every 0.5 seconds

        peft_config = ff.LoraLinearConfig.from_jsonfile(lora_config_filepath)
        hf_peft_config = peft_config.to_hf_config()

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.float32 if peft_config.precision == "fp32" else torch.float16,
            device_map=None  # Prevent meta tensor issues
        )
        model = get_peft_model(model, hf_peft_config, autocast_adapter_dtype=False)
        
        in_dim = model.config.intermediate_size
        out_dim = model.config.hidden_size

        weight_folder = os.path.join(
            cache_folder, "finetuned_models", request.peft_model_id.lower(), "weights", "shard_0"
        )
        num_shards = 1
        while os.path.exists(weight_folder.replace("shard_0", f"shard_{num_shards}")):
            num_shards += 1
        if not in_dim % num_shards == 0:
            raise ValueError(
                f"Number of shards ({num_shards}) must divide the input dimension ({in_dim})"
            )

        lora_weight_files = os.listdir(weight_folder)
        for lora_file in sorted(lora_weight_files):
            lora_filename = ".weight".join(lora_file.split(".weight")[:-1])
            hf_parameter_name = f"base_model.model.model.{lora_filename}.default.weight"
            if hf_parameter_name not in model.state_dict().keys():
                raise KeyError(f"Parameter {lora_file} not found in HF model.")

            ff_dtype = np.float32 if peft_config.precision == "fp32" else np.float16
            weight_path = os.path.join(weight_folder, lora_file)
            # LoRA_A: [in_dim, rank]
            # LoRA_B: [rank, out_dim]
            if "lora_A" in lora_file:
                weight_data = []
                for shard_id in range(num_shards):
                    weight_path_shard = weight_path.replace("shard_0", f"shard_{shard_id}")
                    weight_data_shard = np.fromfile(weight_path_shard, dtype=ff_dtype)
                    weight_data_shard = weight_data_shard.reshape(
                        (in_dim // num_shards, peft_config.rank), order="F"
                    )
                    weight_data.append(weight_data_shard)
                weight_data = np.concatenate(weight_data, axis=0).T
            elif "lora_B" in lora_file:
                weight_data = np.fromfile(weight_path, dtype=ff_dtype)
                weight_data = weight_data.reshape((peft_config.rank, out_dim), order="F").T
            weight_tensor = torch.from_numpy(weight_data)

            param = model.state_dict()[hf_parameter_name]

            actual_numel = weight_tensor.numel()
            expected_numel = param.numel()
            if actual_numel != expected_numel:
                raise ValueError(
                    f"Parameter {lora_file} has unexpected parameter count: {actual_numel} (actual) != {expected_numel} (expected)"
                )

            if weight_tensor.shape != param.shape:
                raise ValueError(
                    f"Parameter {lora_file} has unexpected shape: {weight_tensor.shape} (actual) != {param.shape} (expected)"
                )

            if weight_tensor.dtype != param.dtype:
                raise ValueError(
                    f"Parameter {lora_file} has unexpected dtype: {weight_tensor.dtype} (actual) != {param.dtype} (expected)"
                )

            with torch.no_grad():
                param.copy_(weight_tensor)

        # Ensure all parameters are properly initialized
        for name, param in model.named_parameters():
            if param.device.type == "meta":
                print(f"Parameter {name} is still on 'meta' device. Moving to CPU.")
                param.data = torch.zeros_like(param, device="cpu")  # Allocate real memory

        model = model.to("cpu")

        # Upload model to Hugging Face Hub
        model.push_to_hub(request.upload_peft_model_id, token=request.token, private=request.private)
        print(f"Upload process for {request.upload_peft_model_id} completed.")
        
        return {"status": "success"}

    except Exception as e:
        error_message = f"Error during model upload: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)


# Shutdown event to stop the model server
@app.on_event("shutdown")
async def shutdown_event():
    global llm
    if llm is not None:
        llm.stop_server()

# Main function to run Uvicorn server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Running within the entrypoint folder:
# uvicorn fastapi_incr:app --reload --port

# Running within the python folder:
# uvicorn entrypoint.fastapi_incr:app --reload --port 3000