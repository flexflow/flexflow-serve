import datasets
from transformers import AutoTokenizer
from tqdm import tqdm
import json, os

def build_trace(dataset: datasets.Dataset, model_name: str, num_entries: int, seed: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    dataset = dataset["train"].filter(
        lambda x: x["model"] == "gpt-4" and x["turn"] == 1 and x["language"] == "English"
    ).shuffle(seed=seed).select(range(num_entries))
    pairs = []
    for row in dataset:
        assert len(row["conversation"]) == 2
        assert row["conversation"][0]["role"] == "user"
        assert row["conversation"][1]["role"] == "assistant"
        pairs.append((
            row["conversation"][0]["content"],
            row["conversation"][1]["content"],
        ))

    prompts = []
    avg_prompt_length = 0
    min_prompt_length = float("inf")
    max_prompt_length = 0
    avg_response_length = 0
    min_response_length = float("inf")
    max_response_length = 0
    max_total_length = 0
    for prompt, response in tqdm(pairs, desc="Processing HF trace"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_length = len(tokenizer(prompt)["input_ids"])
        response_length = len(tokenizer(response)["input_ids"])
        prompts.append(prompt)
        avg_prompt_length += prompt_length
        avg_response_length += response_length
        min_prompt_length = min(min_prompt_length, prompt_length)
        min_response_length = min(min_response_length, response_length)
        max_prompt_length = max(max_prompt_length, prompt_length)
        max_response_length = max(max_response_length, response_length)
        max_total_length = max(max_total_length, prompt_length + response_length)
    avg_prompt_length /= len(prompts)
    avg_response_length /= len(prompts)

    return prompts, max_prompt_length, max_response_length, avg_prompt_length, avg_response_length, min_prompt_length, min_response_length, max_total_length

if __name__ == "__main__":
    # Change directory to that holding this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    dataset = datasets.load_dataset("allenai/WildChat")
    prompts, max_prompt_length, max_response_length, avg_prompt_length, avg_response_length, min_prompt_length, min_response_length, max_total_length = build_trace(dataset, "meta-llama/Llama-3.1-70B-Instruct", 250, 42)
    print(f"Number of prompts: {len(prompts)}")
    print(f"Prompt lengths: [{min_prompt_length} -> {max_prompt_length}] (avg: {avg_prompt_length})")
    print(f"Response lengths: [{min_response_length} -> {max_response_length}] (avg: {avg_response_length})")
    print(f"Max total length: {max_total_length}")
    # Save prompts list to a json file

    with open("wildchat.json", "w") as f:
        json.dump(prompts, f, indent=2)