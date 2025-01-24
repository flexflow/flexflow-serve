from dataclasses import asdict, dataclass, field
import json
import os
import random
import requests
from tqdm.asyncio import tqdm
from typing import List, Optional
from collections import OrderedDict
from transformers import AutoTokenizer

SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

@dataclass
class TraceEntry:
    prompt: str
    response: str
    prompt_length: int
    response_length: int

@dataclass
class TracePartition:
    partition_name: str
    model_name: str
    num_warmup_requests: int
    training_entries: List[TraceEntry]
    eval_entries: List[TraceEntry]

@dataclass
class TraceMetadata:
    avg_entries_per_partition: float
    max_prompt_length: int
    min_prompt_length: int
    avg_prompt_length: float
    max_response_length: int
    min_response_length: int
    avg_response_length: float
    max_total_length: int

@dataclass
class Trace:
    partitions: List[TracePartition]
    metadata: TraceMetadata = field(default_factory=lambda: TraceMetadata(0, 0, 0, 0, 0, 0, 0,0))

def download_and_cache_file(url: str, filename: Optional[str] = None):
    """Read and cache a file from a url."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # Stream the response to show the progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for request errors

    # Total size of the file in bytes
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024  # Download in chunks of 1KB

    # Use tqdm to display the progress bar
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename

def get_warmup_entries(model_name: str, num_warmup_requests: int) -> List[TraceEntry]:
    """
    Get a list of warmup entries for a model.
    
    Args:
    model_name (str): The name of the model.
    num_warmup_requests (int): The number of warmup requests to generate.
    
    Returns:
    List[TraceEntry]: A list of warmup entries.
    """
    warmup_entries = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for i in range(num_warmup_requests):
        prompt = "Hello, how are you?"
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        response = "I'm doing well, thank you for asking."
        prompt_length = len(tokenizer(prompt)["input_ids"])
        response_length = len(tokenizer(response)["input_ids"])
        warmup_entries.append(TraceEntry(prompt, response, prompt_length, response_length))
    return warmup_entries

def build_trace(model_name: str, num_entries: int, num_warmup_requests: int, seed: int):
    # Download sharegpt if necessary
    dataset_path = download_and_cache_file(SHAREGPT_URL)

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f, object_pairs_hook=OrderedDict)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
        if data["conversations"][0]["from"] == "human" and data["conversations"][1]["from"] == "gpt"
    ]

    # Shuffle the dataset.
    random.seed(seed)
    random.shuffle(dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    trace = Trace(partitions=[])
    partition = TracePartition(
        partition_name="all",
        model_name=model_name,
        num_warmup_requests=num_warmup_requests,
        training_entries=[],
        eval_entries=[],
    )
    trace_metadata = TraceMetadata(
        avg_entries_per_partition=0,
        max_prompt_length=0,
        min_prompt_length=float("inf"),
        avg_prompt_length=0,
        max_response_length=0,
        min_response_length=float("inf"),
        avg_response_length=0,
        max_total_length=0,
    )

    partition.eval_entries += get_warmup_entries(model_name, num_warmup_requests)
    
    for i in tqdm(range(len(dataset))):
        if len(partition.eval_entries) == num_entries:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        response = dataset[i][1]
        prompt_length = len(tokenizer(prompt)["input_ids"])
        response_length = len(tokenizer(response)["input_ids"])
        new_entry = TraceEntry(prompt, response, prompt_length, response_length)
        partition.eval_entries.append(new_entry)
        trace_metadata.max_prompt_length = max(trace_metadata.max_prompt_length, prompt_length)
        trace_metadata.min_prompt_length = min(trace_metadata.min_prompt_length, prompt_length)
        trace_metadata.avg_prompt_length += prompt_length
        trace_metadata.max_response_length = max(trace_metadata.max_response_length, response_length)
        trace_metadata.min_response_length = min(trace_metadata.min_response_length, response_length)
        trace_metadata.avg_response_length += response_length
        trace_metadata.max_total_length = max(trace_metadata.max_total_length, prompt_length + response_length)
    trace_metadata.avg_prompt_length /= len(partition.eval_entries)
    trace_metadata.avg_response_length /= len(partition.eval_entries)
    trace_metadata.avg_entries_per_partition = len(partition.eval_entries)

    trace.partitions.append(partition)
    trace.metadata = trace_metadata

    return trace

def save_trace(trace: Trace, output_path: str):
    """
    Save a Trace instance to a JSON file.
    
    Args:
    trace (Trace): The trace to save.
    output_path (str): The path where the JSON file will be saved.
    """
    # Convert the Trace instance to a dictionary
    trace_dict = asdict(trace)
    
    # Save the dictionary as a JSON file
    with open(output_path, 'w') as f:
        json.dump(trace_dict, f, indent=2)
    
    print(f"Trace saved to {output_path}")

if __name__ == "__main__":
    # Change directory to that holding this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    num_entries=125
    num_warmup_requests=8
    seed=42

    trace = build_trace("meta-llama/Llama-3.1-70B-Instruct", num_entries, num_warmup_requests, seed)
    print(trace.metadata)
    # Save prompts list to a json file
    save_trace(trace, "sharegpt.json")