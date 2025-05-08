#!/usr/bin/env python
import os, json
from collections import namedtuple

# Base configs dictionaries
ff_init_configs = {
    # required parameters
    "num_gpus": 4,
    "memory_per_gpu": 14000,
    "zero_copy_memory_per_node": 40000,
    # optional parameters
    "num_cpus": 8,
    "cpu_memory_per_node": 2048,
    "legion_utility_processors": 8,
    "data_parallelism_degree": 1,
    "tensor_parallelism_degree": 1,
    "pipeline_parallelism_degree": 4,
    "offload": False,
    "offload_reserve_space_size": 8 * 1024, # 8 GB
    "use_4bit_quantization": False,
    "use_8bit_quantization": False,
    "peft_support_mode": "PEFT_DISABLED",
    "profiling": False,
    "benchmarking": False,
    "inference_debugging": False,
    "fusion": True,
}
llm_configs = {
    # required parameters
    "llm_model": "meta-llama/Llama-3.1-8B-Instruct",
    # optional parameters
    "cache_path": os.environ.get("FF_CACHE_PATH", ""),
    "refresh_cache": False,
    "full_precision": True,
    "prompt": "",
    "output_file": "",
    "max_length": 255,
}
ssm_configs = {
    "ssms": [
        {
            # required ssm parameter
            "ssm_model": "meta-llama/Llama-3.2-1B-Instruct",
            # optional ssm parameters
            "cache_path": "",
            "refresh_cache": False,
            "full_precision": False,
        },
    ]
}
# Merge dictionaries
ff_init_configs.update(llm_configs)

Parallelism = namedtuple('Parallelism', ['tp', 'pp'])
SpecModelPair = namedtuple('SpecModelPair', ['big_model', 'small_model'])

def gen_incr_dec_configs(prompt_file, output_folder, incr_dec_models, parallelism_settings, full_precision_settings, config_output_folder):
    # Generate incremental decoding configs
    for model_name in incr_dec_models:
        for full_precision in full_precision_settings:
            for parallelism_degrees in parallelism_settings:
                tp, pp = parallelism_degrees
                
                _, after_slash = model_name.rsplit("/", maxsplit=1)
                filename = (
                    "incr_dec-"
                    + "python-"
                    + after_slash.lower()
                    + ("-full_prec-" if full_precision else "-half_prec-")
                    + f"{tp}_tp_{pp}_pp"
                )
                test_configs_file = os.path.join(config_output_folder, f"{filename}.json")
                output_file = os.path.join(output_folder, filename + ".json")

                ff_init_configs["tensor_parallelism_degree"] = tp
                ff_init_configs["pipeline_parallelism_degree"] = pp
                ff_init_configs["llm_model"] = model_name
                ff_init_configs["full_precision"] = full_precision
                ff_init_configs["output_file"] = output_file
                ff_init_configs["prompt"] = prompt_file

                with open(test_configs_file, "w+") as outfile:
                    json.dump(ff_init_configs, outfile, indent=4)

# Generate speculative inference configs
def gen_spec_configs(prompt_file, output_folder, specinfer_model_pairs, parallelism_settings, full_precision_settings, config_output_folder):
    for model_pair in specinfer_model_pairs:
        for full_precision in full_precision_settings:
            for parallelism_degrees in parallelism_settings:
                big_model, small_model = model_pair
                tp, pp = parallelism_degrees

                _, after_slash = big_model.rsplit("/", maxsplit=1)
                filename = (
                    "spec_infer-"
                    + "python-"
                    + after_slash.lower()
                    + ("-full_prec-" if full_precision else "-half_prec-")
                    + f"{tp}_tp_{pp}_pp"
                )
                test_configs_file = os.path.join(config_output_folder, f"{filename}.json")
                output_file = os.path.join(output_folder, filename + ".json")

                ff_init_configs["tensor_parallelism_degree"] = tp
                ff_init_configs["pipeline_parallelism_degree"] = pp
                ff_init_configs["llm_model"] = big_model
                ff_init_configs["full_precision"] = full_precision
                ff_init_configs["output_file"] = output_file
                ff_init_configs["prompt"] = prompt_file

                ssm_configs["ssms"][0]["ssm_model"] = small_model
                ssm_configs["ssms"][0]["full_precision"] = full_precision
                ff_init_configs.update(ssm_configs)

                with open(test_configs_file, "w+") as outfile:
                    json.dump(ff_init_configs, outfile, indent=4)

if __name__ == "__main__":
    # Change working dir to root of repo
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    # root_dir = great-grandparent dir
    root_dir = os.path.dirname(os.path.dirname(dname))
    os.chdir(root_dir)
    # print current working dir
    print("CWD: ", os.getcwd())

    prompt_file = os.path.abspath("./inference/prompt/test.json")
    output_folder = os.path.abspath("./inference/output")
    config_output_folder = os.path.abspath("./inference/inf_test_configs")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(config_output_folder, exist_ok=True)

    # Models
    llama_models = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", ]
    opt_models = ["facebook/opt-6.7b", "facebook/opt-125m"]

    # Incr decoding configs
    # large models, only tp=4, pp=1
    gen_incr_dec_configs(prompt_file, 
                         output_folder, 
                         incr_dec_models=[llama_models[0], opt_models[0]],
                         parallelism_settings=[Parallelism(4, 1)], 
                         full_precision_settings=[False,], 
                         config_output_folder=config_output_folder
    )
    # small models tp=2, pp=2
    gen_incr_dec_configs(prompt_file, 
                         output_folder, 
                         incr_dec_models=[llama_models[1], opt_models[1]],
                         parallelism_settings=[Parallelism(2, 2)], 
                         full_precision_settings=[False,], 
                         config_output_folder=config_output_folder
    )
    # Spec decoding configs
    # llama, tp=4, tp=2, tp=1
    gen_spec_configs(prompt_file, 
                     output_folder, 
                     specinfer_model_pairs=[SpecModelPair(llama_models[0], llama_models[1]),], 
                     parallelism_settings=[Parallelism(4, 1), Parallelism(2, 2), Parallelism(1, 4)], 
                     full_precision_settings=[False,], 
                     config_output_folder=config_output_folder
    )
    # opt, tp=4 only
    gen_spec_configs(prompt_file, 
                     output_folder, 
                     specinfer_model_pairs=[SpecModelPair(opt_models[0], opt_models[1]),], 
                     parallelism_settings=[Parallelism(4, 1)], 
                     full_precision_settings=[False,], 
                     config_output_folder=config_output_folder
    )
