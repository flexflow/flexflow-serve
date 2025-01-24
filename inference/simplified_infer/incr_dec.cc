/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "flexflow/inference.h"
#include "flexflow/request_manager.h"
#include "models/falcon.h"
#include "models/llama.h"
#include "models/mpt.h"
#include "models/opt.h"
#include <cassert>
#include <wordexp.h>

using namespace FlexFlow;
using namespace Legion;
using json = nlohmann::json;

Legion::Logger log_app("llama");

struct FilePaths {
  std::string cache_folder_path;
  std::string trace_file_path;
  std::string trace_output_path;
  std::string log_file_path;
  std::string csv_file_path;
};

void parse_input_args(char **argv,
                      int argc,
                      FilePaths &paths,
                      std::string &llm_model_name,
                      bool &use_full_precision,
                      bool &verbose,
                      int &max_requests_per_batch,
                      int &max_tokens_per_batch,
                      int &max_sequence_length,
                      int &max_output_length,
                      bool &do_sample,
                      int &request_per_second,
                      bool &add_special_tokens,
                      std::string &target_partition) {
  for (int i = 1; i < argc; i++) {
    // llm model type
    if (!strcmp(argv[i], "-llm-model")) {
      llm_model_name = std::string(argv[++i]);
      for (char &c : llm_model_name) {
        c = std::tolower(c);
      }
      continue;
    }
    // cache folder
    if (!strcmp(argv[i], "-cache-folder")) {
      paths.cache_folder_path = std::string(argv[++i]);
      continue;
    }
    // traces
    if (!strcmp(argv[i], "-trace")) {
      paths.trace_file_path = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-trace-output-path")) {
      paths.trace_output_path = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-target-partition")) {
      target_partition = std::string(argv[++i]);
      continue;
    }
    // output file
    if (!strcmp(argv[i], "-log-output-path")) {
      paths.log_file_path = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-csv-output-path")) {
      paths.csv_file_path = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--use-full-precision")) {
      use_full_precision = true;
      continue;
    }
    // verbose logging to stdout
    if (!strcmp(argv[i], "--verbose")) {
      verbose = true;
      continue;
    }
    if (!strcmp(argv[i], "--do-sample")) {
      do_sample = true;
      continue;
    }
    if (!strcmp(argv[i], "--max-requests-per-batch")) {
      max_requests_per_batch = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--max-tokens-per-batch")) {
      max_tokens_per_batch = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--max-sequence-length")) {
      max_sequence_length = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--max-output-length")) {
      max_output_length = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--request-per-second")) {
      request_per_second = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--add-special-tokens")) {
      add_special_tokens = true;
      continue;
    }
  }
  if (paths.cache_folder_path.empty()) {
    char const *ff_cache_path = std::getenv("FF_CACHE_PATH");
    paths.cache_folder_path = ff_cache_path ? std::string(ff_cache_path)
                                            : std::string("~/.cache/flexflow");
  }
  // Expand ~ to the home directory if needed
  wordexp_t p;
  wordexp(paths.cache_folder_path.c_str(), &p, 0);
  paths.cache_folder_path = p.we_wordv[0];
  wordfree(&p);
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffconfig;
  if (ffconfig.cpu_offload == false && ffconfig.quantization_type != DT_NONE) {
    assert(false && "Doesn't support quantization in non-offload mode");
  }
  FilePaths file_paths;
  std::string llm_model_name;
  bool use_full_precision = false;
  bool verbose = false;
  bool do_sample = false;
  int max_requests_per_batch = 8;
  int max_tokens_per_batch = 128;
  int max_sequence_length = 512;
  int max_output_length = 512;
  int num_warmup_requests = 0;
  double warmup_delay = 15.0;
  RequestManager::DecodingMode decoding_mode =
      RequestManager::INCREMENTAL_DECODING;
  int sampling_seed = 0;
  int request_per_second = -1;
  bool add_special_tokens = false;
  std::string target_partition = "FEATURE_EXTRACTION";

  InputArgs const &command_args = HighLevelRuntime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  parse_input_args(argv,
                   argc,
                   file_paths,
                   llm_model_name,
                   use_full_precision,
                   verbose,
                   max_requests_per_batch,
                   max_tokens_per_batch,
                   max_sequence_length,
                   max_output_length,
                   do_sample,
                   request_per_second,
                   add_special_tokens,
                   target_partition);

  assert(ffconfig.data_parallelism_degree * ffconfig.tensor_parallelism_degree *
             ffconfig.pipeline_parallelism_degree ==
         ffconfig.numNodes * ffconfig.workersPerNode);

  // Get dataset
  std::ifstream input_file(file_paths.trace_file_path);
  assert(input_file.good() && "Prompt file does not exist.");
  nlohmann::ordered_json j = nlohmann::ordered_json::parse(input_file);
  input_file.close();

  // Find the partition with name "FEATURE_EXTRACTION"
  auto &partitions = j["partitions"];
  auto it =
      std::find_if(partitions.begin(),
                   partitions.end(),
                   [target_partition](nlohmann::ordered_json const &partition) {
                     return partition["partition_name"] == target_partition;
                   });
  nlohmann::ordered_json &partition = *it;
  if (it == partitions.end()) {
    std::cerr << "Partition " << target_partition
              << " not found in the trace file." << std::endl;
    assert(false);
  }
  // check that the max prompt + response length sum in the eval_entries in the
  // partition does not exceed the max_sequence_length
  int max_prompt_response_length = 0;
  for (auto &eval_entry : partition["eval_entries"]) {
    int prompt_length = eval_entry["prompt_length"];
    int response_length = eval_entry["response_length"];
    if (response_length >= max_output_length) {
      std::cerr << "Error: A response length from the targt partition in the "
                   "dataset (="
                << response_length
                << ") exceeds the max_output_length(=" << max_output_length
                << ")." << std::endl;
      assert(false);
    }
    max_prompt_response_length =
        std::max(max_prompt_response_length, prompt_length + response_length);
  }
  if (max_prompt_response_length >= max_sequence_length) {
    std::cerr << "Error: max prompt + response length sum (="
              << max_prompt_response_length
              << ") in the eval_entries in the partition exceeds the "
                 "max_sequence_length(="
              << max_sequence_length << ")." << std::endl;
    assert(false);
  }

  // Get model configs
  std::string config_filepath = join_path(
      {file_paths.cache_folder_path, "configs", llm_model_name, "config.json"});
  std::string tokenizer_filepath =
      join_path({file_paths.cache_folder_path, "tokenizers", llm_model_name});
  std::string weights_filepath =
      join_path({file_paths.cache_folder_path,
                 "weights",
                 llm_model_name,
                 use_full_precision ? "full-precision" : "half-precision"});
  std::ifstream config_file_handle(config_filepath);
  if (!config_file_handle.good()) {
    std::cout << "Model config file " << config_filepath << " not found."
              << std::endl;
    assert(false);
  }
  json model_config = json::parse(config_file_handle,
                                  /*parser_callback_t */ nullptr,
                                  /*allow_exceptions */ true,
                                  /*ignore_comments */ true);
  ModelType model_type = ModelType::UNKNOWN;
  auto architectures = model_config["architectures"];
  for (auto const &str : architectures) {
    if (str == "LlamaForCausalLM" || str == "LLaMAForCausalLM" ||
        str == "MistralForCausalLM") {
      model_type = ModelType::LLAMA;
      break;
    } else if (str == "OPTForCausalLM") {
      model_type = ModelType::OPT;
      break;
    } else if (str == "RWForCausalLM" || str == "FalconForCausalLM") {
      model_type = ModelType::FALCON;
      break;
    } else if (str == "MPTForCausalLM") {
      model_type = ModelType::MPT;
      break;
    }
  }
  int bos_token_id = model_config.find("bos_token_id") == model_config.end()
                         ? -1
                         : (int)model_config.at("bos_token_id");
  std::vector<int> eos_token_ids;
  if (model_config.find("eos_token_id") != model_config.end()) {
    if (model_config["eos_token_id"].is_array()) {
      for (auto &eos_token_id : model_config["eos_token_id"]) {
        eos_token_ids.push_back(eos_token_id);
      }
    } else {
      eos_token_ids.push_back(model_config["eos_token_id"]);
    }
  } else {
    eos_token_ids.push_back(-1);
  }

  assert(model_type != ModelType::UNKNOWN &&
         "Invalid LLM model type passed (or no type was passed).");

  // set request manager properties
  srand(sampling_seed);
  GenerationConfig generationConfig(do_sample, 0.8, 0.6, false, 16);
  RequestManager *rm = RequestManager::get_request_manager();
  rm->set_max_requests_per_batch(max_requests_per_batch);
  rm->set_max_tokens_per_batch(max_tokens_per_batch);
  rm->set_max_tokens_per_ssm_batch(max_tokens_per_batch);
  rm->set_max_tokens_per_prefilling_batch(max_tokens_per_batch);
  rm->set_max_sequence_length(max_sequence_length);
  rm->set_max_output_length(max_output_length);
  rm->set_decoding_mode(decoding_mode);
  rm->set_slo_violation_early_termination(false);
  rm->set_baseline_latency(50);
  rm->set_ssm_spec_latency(20);
  rm->set_llm_verify_latency(50);
  rm->set_spec_infer_old_version(true);
  rm->set_greedy_schedule(false);
  rm->set_equal_schedule(false);
  rm->set_max_tree_depth(8);
  rm->set_max_tree_width(16);
  rm->set_verbose(verbose);
  rm->set_streaming_cache(false);
  rm->register_tokenizer(
      model_type, bos_token_id, eos_token_ids, tokenizer_filepath);
  rm->register_output_filepath(file_paths.log_file_path);

  FFModel model(ffconfig, ffconfig.cpu_offload);
  if (model_type == ModelType::LLAMA) {
    LLAMA::create_llama_model(model,
                              config_filepath,
                              weights_filepath,
                              INC_DECODING_MODE,
                              generationConfig,
                              false,
                              use_full_precision);
  } else if (model_type == ModelType::OPT) {
    OPT::create_opt_model(model,
                          config_filepath,
                          weights_filepath,
                          INC_DECODING_MODE,
                          use_full_precision);
  } else if (model_type == ModelType::FALCON) {
    FALCON::create_falcon_model(model,
                                config_filepath,
                                weights_filepath,
                                INC_DECODING_MODE,
                                use_full_precision);
  } else if (model_type == ModelType::MPT) {
    MPT::create_mpt_model(model,
                          config_filepath,
                          weights_filepath,
                          INC_DECODING_MODE,
                          generationConfig,
                          use_full_precision);
  } else {
    assert(false && "unknow model type");
  }

  rm->start_background_server(&model);

  int total_num_requests = 0;
  {
    // Iterate through eval_entries
    std::vector<GenerationRequest> requests;
    std::vector<double> timestamps, ratios;
    if (partition.contains("num_warmup_requests")) {
      num_warmup_requests = partition["num_warmup_requests"];
    }
    for (auto &entry : partition["eval_entries"]) {
      std::string text = entry["prompt"];
      int max_new_tokens_ = entry["response_length"];

      bool is_warmup_request = total_num_requests < num_warmup_requests;
      double request_delay =
          1000.0 *
          (request_per_second > 0 ? (1.0 / (double)request_per_second) : 0);
      double emission_time_ms =
          is_warmup_request
              ? 0.0
              : (warmup_delay +
                 request_delay * (total_num_requests - num_warmup_requests));

      GenerationRequest inference_req(text,             // prompt
                                      -1.0,             // slo_ratio
                                      emission_time_ms, // emission_time_ms
                                      add_special_tokens);

      requests.push_back(inference_req);
      timestamps.push_back(emission_time_ms);
      ratios.push_back(1.0);
      total_num_requests++;

      if (verbose) {
        break;
      }
    }
    TraceEmissionMachine emission_machine(timestamps, ratios);
    std::vector<GenerationResult> result =
        model.generate(requests, emission_machine);
    assert(result.size() == requests.size());
    assert(result.size() == total_num_requests);
    assert(result.size() == partition["eval_entries"].size());
    int i = 0;
    for (auto &entry : partition["eval_entries"]) {
      entry["original_response"] = entry["response"];
      entry["original_response_length"] = entry["response_length"];
      std::string ff_out = result[i].output_text;
      int tot_length = result[i].output_text.length();
      entry["response"] = ff_out;
      entry["response_length"] = result[i].output_tokens.size();
      entry["specinfer_decoding_steps"] = result[i].decoding_steps;
      i++;
    }

    // Write the modified JSON to a file
    std::ofstream output_file(file_paths.trace_output_path);
    if (output_file.is_open()) {
      output_file << j.dump(2);
      output_file.close();
      std::cout << "Modified JSON has been saved to "
                << file_paths.trace_output_path << std::endl;
    } else {
      std::cerr << "Unable to open file for writing." << std::endl;
    }
  }

  // terminate the request manager by stopping the background thread
  rm->terminate_background_server();

  std::string header =
      "llm,partition,max_requests_per_batch,max_tokens_per_"
      "batch,request_per_second,is_warmup_request,request_guid,"
      "request_step_idx,timestamp,num_generated_tokens";
  // csv filepath
  // create csv filepath and add header if it doesn't exist

  bool csv_file_exists = std::filesystem::exists(file_paths.csv_file_path);
  if (!csv_file_exists) {
    // Create new file and write header
    std::ofstream file(file_paths.csv_file_path);
    if (!file.is_open()) {
      std::cerr << "Failed to open file: " << file_paths.csv_file_path
                << std::endl;
      assert(false);
    }
    file << header << "\n";
    file.close();
  }

  // Append the new row
  std::ofstream file(file_paths.csv_file_path, std::ios::app);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << file_paths.csv_file_path
              << std::endl;
  }

  std::vector<NewProfileInfo> new_profiling_info = rm->get_new_profiling_info();
  for (auto const &info : new_profiling_info) {
    file << llm_model_name + ",";
    file << target_partition + ",";
    file << std::to_string(max_requests_per_batch) + ",";
    file << std::to_string(max_tokens_per_batch) + ",";
    file << std::to_string(request_per_second) + ",";
    bool is_warmup_request =
        (info.request_guid - 1000000) < num_warmup_requests;
    file << std::to_string(is_warmup_request) + ",";
    file << info.request_guid << "," << info.request_step_idx << ","
         << info.timestamp << "," << info.num_generated_tokens << "\n";
  }
  file.close();

  // Execution fence
  {
    Future future = runtime->issue_execution_fence(ctx);
    future.get_void_result();
  }

  // float* data
  std::cout << "----------inference finished--------------" << std::endl;

  // free tokenizer space in memory
}

void FlexFlow::register_custom_tasks() {}
