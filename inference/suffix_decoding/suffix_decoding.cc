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

#include "suffix_tree.h"
#include "flexflow/inference.h"
#include "flexflow/request_manager.h"
#include "models/falcon.h"
#include "models/llama.h"
#include "models/mpt.h"
#include "models/opt.h"
#include <cassert>
#include <filesystem>
#include <string>
#include <wordexp.h>

using namespace FlexFlow;
using namespace Legion;
using RequestGuid = BatchConfig::RequestGuid;

Legion::Logger log_app("llama");

struct FilePaths {
  std::string cache_folder_path;
  std::string trace_file_path;
  std::string trace_output_path;
  std::string log_file_path;
  std::string csv_file_path;
};

struct ModelMeta {
  std::string llm_model_name;
  ModelType llm_model_type;
  std::string llm_tokenizer_path;
  std::string llm_weights_path;
  std::string llm_model_config_path;

  int bos_token_id, eos_token_id;
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
                      double &request_per_second,
                      bool &add_special_tokens,
                      std::string &target_partition,
                      std::string &matching_strategy,
                      int &max_tree_depth,
                      float &max_spec_factor,
                      bool &online_tree_update) {
  for (int i = 1; i < argc; i++) {
    // llm model name
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
    // trace
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
    if (!strcmp(argv[i], "--do-sample")) {
      do_sample = true;
      continue;
    }
    if (!strcmp(argv[i], "--request-per-second")) {
      request_per_second = std::stod(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--add-special-tokens")) {
      add_special_tokens = true;
      continue;
    }
    // suffix tree
    if (!strcmp(argv[i], "--matching-strategy")) {
      matching_strategy = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--max-tree-depth")) {
      max_tree_depth = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--max-spec-factor")) {
      max_spec_factor = std::stof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--disable-online-tree-update")) {
      online_tree_update = false;
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

void get_model_meta(FilePaths &file_paths,
                    ModelMeta &model_metadata,
                    bool use_full_precision) {
  if (model_metadata.llm_model_name.empty()) {
    assert(false && "LLM model name is not set");
  }
  model_metadata.llm_model_config_path =
      join_path({file_paths.cache_folder_path,
                 "configs",
                 model_metadata.llm_model_name,
                 "config.json"});
  model_metadata.llm_tokenizer_path =
      join_path({file_paths.cache_folder_path,
                 "tokenizers",
                 model_metadata.llm_model_name});
  model_metadata.llm_weights_path =
      join_path({file_paths.cache_folder_path,
                 "weights",
                 model_metadata.llm_model_name,
                 use_full_precision ? "full-precision" : "half-precision"});

  std::ifstream llm_config_file_handle(model_metadata.llm_model_config_path);
  if (!llm_config_file_handle.good()) {
    std::cout << "LLM Model config file "
              << model_metadata.llm_model_config_path << " not found."
              << std::endl;
    assert(false);
  }
  nlohmann::ordered_json llm_model_config =
      nlohmann::ordered_json::parse(llm_config_file_handle,
                                    /*parser_callback_t */ nullptr,
                                    /*allow_exceptions */ true,
                                    /*ignore_comments */ true);

  model_metadata.llm_model_type = ModelType::UNKNOWN;
  auto architectures = llm_model_config["architectures"];
  for (auto const &str : architectures) {
    if (str == "LlamaForCausalLM" || str == "LLaMAForCausalLM" ||
        str == "MistralForCausalLM") {
      model_metadata.llm_model_type = ModelType::LLAMA;
      break;
    } else if (str == "OPTForCausalLM") {
      model_metadata.llm_model_type = ModelType::OPT;
      break;
    } else if (str == "RWForCausalLM" || str == "FalconForCausalLM") {
      model_metadata.llm_model_type = ModelType::FALCON;
      break;
    } else if (str == "MPTForCausalLM") {
      model_metadata.llm_model_type = ModelType::MPT;
      break;
    }
  }
  model_metadata.bos_token_id =
      llm_model_config.find("bos_token_id") == llm_model_config.end()
          ? -1
          : (int)llm_model_config.at("bos_token_id");
  model_metadata.eos_token_id =
      llm_model_config.find("eos_token_id") == llm_model_config.end()
          ? -1
          : (int)llm_model_config.at("eos_token_id");

  assert(model_metadata.llm_model_type != ModelType::UNKNOWN &&
         "Invalid LLM model type passed (or no type was passed).");
}

std::string vectorToString(const std::vector<double>& vec, int precision = 4) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << "\"[";
  
  for (size_t i = 0; i < vec.size(); ++i) {
    oss << vec[i];
    if (i < vec.size() - 1) {
      oss << ",";
    }
  }
  
  oss << "]\"";
  return oss.str();
}

std::string vectorToStringInt(const std::vector<int>& vec) {
  std::ostringstream oss;
  oss << "\"[";
  
  for (size_t i = 0; i < vec.size(); ++i) {
    oss << vec[i];
    if (i < vec.size() - 1) {
      oss << ",";
    }
  }
  
  oss << "]\"";
  return oss.str();
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffconfig;
  FilePaths file_paths;
  ModelMeta model_metadata;
  bool use_full_precision = false;
  bool verbose = false;
  int max_requests_per_batch = 8;
  int max_tokens_per_batch = 128;
  int max_sequence_length = 512;
  int max_output_length = 512;

  std::string matching_strategy = "linear_token_path";
  int max_tree_depth = 16;
  float max_spec_factor = 1.0;
  bool online_tree_update = true;
  RequestManager::DecodingMode decoding_mode = RequestManager::SUFFIX_DECODING;

  bool do_sample = false;
  int sampling_seed = 0;
  double request_per_second = 1.0;
  bool add_special_tokens = false;
  std::string target_partition = "FEATURE_EXTRACTION";

  InputArgs const &command_args = HighLevelRuntime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  parse_input_args(argv,
                   argc,
                   file_paths,
                   model_metadata.llm_model_name,
                   use_full_precision,
                   verbose,
                   max_requests_per_batch,
                   max_tokens_per_batch,
                   max_sequence_length,
                   max_output_length,
                   do_sample,
                   request_per_second,
                   add_special_tokens,
                   target_partition,
                   matching_strategy,
                   max_tree_depth,
                   max_spec_factor,
                   online_tree_update);

  get_model_meta(file_paths, model_metadata, use_full_precision);

  assert(ffconfig.data_parallelism_degree * ffconfig.tensor_parallelism_degree *
             ffconfig.pipeline_parallelism_degree ==
         ffconfig.numNodes * ffconfig.workersPerNode);

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

  // Sanity check for SpecInfer old version
  // Total verified tokens
  assert(max_tokens_per_batch >= max_requests_per_batch * 21);

  // Create SentencePiece tokenizer or OPT tokenizer
  srand(sampling_seed);
  GenerationConfig generationConfig(do_sample, 0.8, 0.6, false, 16);
  InferenceManager *im = InferenceManager::get_inference_manager();
  RequestManager *rm = RequestManager::get_request_manager();
  rm->set_max_requests_per_batch(max_requests_per_batch);
  rm->set_max_tokens_per_batch(max_tokens_per_batch);
  rm->set_max_tokens_per_ssm_batch(max_tokens_per_batch);
  rm->set_max_tokens_per_prefilling_batch(max_tokens_per_batch);
  rm->set_max_sequence_length(max_sequence_length);
  rm->set_max_output_length(max_output_length);
  rm->set_verbose(verbose);
  rm->set_streaming_cache(false);
  rm->register_tokenizer(model_metadata.llm_model_type,
                         model_metadata.bos_token_id,
                         model_metadata.eos_token_id,
                         model_metadata.llm_tokenizer_path);
  rm->set_decoding_mode(decoding_mode);
  rm->set_slo_violation_early_termination(false);
  rm->set_baseline_latency(50);
  rm->set_ssm_spec_latency(20);
  rm->set_llm_verify_latency(50);
  rm->set_max_tree_depth(8);
  rm->set_max_tree_width(16);
  rm->set_spec_infer_old_version(true);
  rm->set_greedy_schedule(false);
  rm->set_equal_schedule(false);
  rm->register_output_filepath(file_paths.log_file_path);
  // SuffixTree
  assert(matching_strategy == "linear_token_path" ||
         matching_strategy == "dynamic_token_tree");
  rm->set_suffix_tree_matching_strategy(
      matching_strategy == "linear_token_path"
          ? MatchingStrategy::LINEAR_TOKEN_PATH
          : MatchingStrategy::DYNAMIC_TOKEN_TREE);
  rm->set_suffix_tree_max_depth(max_tree_depth);
  rm->set_suffix_tree_max_spec_factor(max_spec_factor);
  rm->set_suffix_tree_online_tree_update(online_tree_update);
  rm->init_suffix_tree(file_paths.trace_file_path, target_partition);

  // Create LLM model
  FFModel tree_model(ffconfig, ffconfig.cpu_offload);
  if (model_metadata.llm_model_type == ModelType::LLAMA) {
    LLAMA::create_llama_model(tree_model,
                              model_metadata.llm_model_config_path,
                              model_metadata.llm_weights_path,
                              TREE_VERIFY_MODE,
                              generationConfig,
                              false,
                              use_full_precision);
  } else if (model_metadata.llm_model_type == ModelType::OPT) {
    OPT::create_opt_model(tree_model,
                          model_metadata.llm_model_config_path,
                          model_metadata.llm_weights_path,
                          TREE_VERIFY_MODE,
                          use_full_precision);
  } else if (model_metadata.llm_model_type == ModelType::FALCON) {
    FALCON::create_falcon_model(tree_model,
                                model_metadata.llm_model_config_path,
                                model_metadata.llm_weights_path,
                                TREE_VERIFY_MODE,
                                use_full_precision);
  } else if (model_metadata.llm_model_type == ModelType::MPT) {
    MPT::create_mpt_model(tree_model,
                          model_metadata.llm_model_config_path,
                          model_metadata.llm_weights_path,
                          TREE_VERIFY_MODE,
                          generationConfig,
                          use_full_precision);
  } else {
    assert(false && "Invalid LLM model type passed (or no type was passed).");
  }

  rm->start_background_server(&tree_model);

  int total_num_requests = 0;
  {
    // Iterate through eval_entries
    std::vector<GenerationRequest> requests;
    std::vector<double> timestamps, ratios;
    for (auto &entry : partition["eval_entries"]) {
      std::string text = entry["prompt"];
      int max_new_tokens_ = entry["response_length"];
      // printf("Prompt[%d]: %s\n", total_num_requests, text.c_str());
      GenerationRequest inference_req(text, -1.0, 0, add_special_tokens);
      // inference_req.prompt = text;
      // inference_req.slo_ratio = -1.0;
      // inference_req.emission_time_ms = 0;
      // // inference_req.max_new_tokens = max_new_tokens_;
      // inference_req.add_special_tokens = false;
      requests.push_back(inference_req);
      timestamps.push_back(0);
      ratios.push_back(1.0);
      total_num_requests++;

      if (verbose) {
        break;
      }
    }
    TraceEmissionMachine emission_machine(timestamps, ratios);
    std::vector<GenerationResult> result =
        tree_model.generate(requests, emission_machine);
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

  /*
  {
    // get profliling results
    std::unordered_map<RequestGuid, RequestProfileInfo> profiling_results =
        rm->get_requests_profiling();
    std::unordered_map<RequestGuid, GenerationResult>
    request_generation_results =
        rm->get_request_generation_results();
    // save profiling results to csv file
    std::string header =
        "llm,partition,max_tree_depth,online_tree_update,matching_strategy,batch_size,tokens_per_batch,mean_speculated_tokens,mean_accepted_candidate_length,mean_acceptance_rate,mean_prefix_length,mean_total_time_ms,throughput_tokens_per_sec,mean_generated_tokens_per_step,mean_decoding_steps,mean_output_length,mean_e2e_latency,mean_llm_ttft,mean_llm_tpot,mean_tree_operation_time_per_step,mean_speculated_tokens_per_req,mean_accepted_candidate_length_per_req,mean_acceptance_rate_per_req,mean_prefix_length_per_req,generated_tokens_per_step,num_active_requests_per_step";
    std::string row = "";

    double mean_decoding_steps = 0;
    double mean_output_length = 0;
    double mean_e2e_latency = 0;
    double mean_llm_ttft = 0;
    double mean_llm_tpot = 0;
    
    std::vector<double> mean_speculated_tokens_per_req;
    std::vector<double> mean_accepted_candidate_length_per_req;
    std::vector<double> mean_acceptance_rate_per_req;
    std::vector<double> mean_prefix_length_per_req;
    std::vector<double> mean_tree_operation_time_per_req;
    double mean_speculated_tokens = 0;
    double mean_accepted_candidate_length = 0;
    double mean_acceptance_rate = 0;
    double mean_prefix_length = 0;

    std::ofstream file_debug("/home/yak/goliaro/FlexFlow/inference/output/accepted_tokens.csv");
    if (!file_debug.is_open()) {
      std::cerr << "Failed to open file: " << "/home/yak/goliaro/FlexFlow/inference/output/accepted_tokens.csv"
                << std::endl;
      assert(false);
    }
    file_debug<< "accepted tokens per step (one line per request)" << std::endl;

    std::ofstream file_debug2("/home/yak/goliaro/FlexFlow/inference/output/generated_tokens.csv");
    if (!file_debug2.is_open()) {
      std::cerr << "Failed to open file: " << "/home/yak/goliaro/FlexFlow/inference/output/generated_tokens.csv"
                << std::endl;
      assert(false);
    }
    file_debug2<< "generated tokens per step (one line per request)" << std::endl;

    for (auto &profiling_result : profiling_results) {
      RequestGuid guid = profiling_result.first;
      RequestProfileInfo &req_profile_info = profiling_result.second;
      GenerationResult &result = request_generation_results[guid];
      mean_decoding_steps += req_profile_info.llm_decoding_steps;
      mean_output_length += result.output_tokens.size();
      mean_e2e_latency += (double)(req_profile_info.finish_time - req_profile_info.start_time)/1000.0;
      // LLM ttft
      double prefilling_time_ms = 0.0;
      if (req_profile_info.start_decoding_time != 0) {
        prefilling_time_ms =
            (req_profile_info.start_decoding_time - req_profile_info.start_time) /
            1000.0;
      } else {
        prefilling_time_ms =
            (req_profile_info.finish_time - req_profile_info.start_time) / 1000.0;
      }
      mean_llm_ttft += prefilling_time_ms;
      // LLM tpot
      double per_token_time_ms = 0;
      if (req_profile_info.start_decoding_time != 0) {
        per_token_time_ms =
            (req_profile_info.finish_time - req_profile_info.start_decoding_time) /
            1000.0 / result.output_tokens.size();
      }
      mean_llm_tpot += per_token_time_ms;

      // Suffix decoding stuff
      double mean_spec_size_req = (double)std::accumulate(req_profile_info.speculated_size_per_step.begin(),
                                                          req_profile_info.speculated_size_per_step.end(),
                                                        0);
      double mean_accepted_candidate_len_req = (double)std::accumulate(req_profile_info.accepted_tokens_per_step.begin(),
                                                          req_profile_info.accepted_tokens_per_step.end(),
                                                        0);
      mean_prefix_length_per_req.push_back((double)std::accumulate(req_profile_info.prefix_length_per_step.begin(),
                                                          req_profile_info.prefix_length_per_step.end(),
                                                        0) / req_profile_info.prefix_length_per_step.size());
      double mean_acceptance_rate_req = mean_accepted_candidate_len_req/mean_spec_size_req;
      
      mean_spec_size_req /= req_profile_info.speculated_size_per_step.size();
      mean_accepted_candidate_len_req /= req_profile_info.accepted_tokens_per_step.size();
    
      mean_speculated_tokens_per_req.push_back( mean_spec_size_req );
      mean_accepted_candidate_length_per_req.push_back( mean_accepted_candidate_len_req );
      mean_acceptance_rate_per_req.push_back(mean_acceptance_rate_req);
      mean_speculated_tokens += mean_spec_size_req;
      mean_accepted_candidate_length += mean_accepted_candidate_len_req;
      mean_acceptance_rate += mean_acceptance_rate_req;
      mean_prefix_length += mean_prefix_length_per_req.back();

      file_debug << vectorToStringInt(req_profile_info.accepted_tokens_per_step) << std::endl;
      file_debug2 << vectorToStringInt(req_profile_info.generated_tokens_per_step__) << std::endl;

    }

    file_debug.close();
    
    mean_decoding_steps /= profiling_results.size();
    mean_output_length /= profiling_results.size();
    mean_e2e_latency /= profiling_results.size();
    mean_llm_ttft /= profiling_results.size();
    mean_llm_tpot /= profiling_results.size();
    mean_speculated_tokens /= profiling_results.size();
    mean_accepted_candidate_length /= profiling_results.size();
    mean_acceptance_rate /= profiling_results.size();
    mean_prefix_length /= profiling_results.size();

    ProfileInfo profile_info = rm->get_profiling_info();
    // total time
    long long total_time =
        profile_info.server_end_time - profile_info.server_start_time;
    // throughput tokens per sec
    int total_tokens = 0;
    for (int num_tokens : profile_info.generated_tokens_per_step) {
      total_tokens += num_tokens;
    }
    double throughput_tokens_per_sec = (double)total_tokens / (total_time /
    1e6);
    // mean generated tokens per step
    double mean_generated_tokens_per_step =
        (double)std::accumulate(profile_info.generated_tokens_per_step.begin(),
                                profile_info.generated_tokens_per_step.end(),
                                0);
    double total_request_steps =
        (double)std::accumulate(profile_info.requests_per_step.begin(),
                                profile_info.requests_per_step.end(),
                                0);
    mean_generated_tokens_per_step /= total_request_steps;
    double mean_tree_operation_time_per_step = 
        (double)std::accumulate(profile_info.tree_operation_step_times.begin(),
                                profile_info.tree_operation_step_times.end(),
                                0);
    mean_tree_operation_time_per_step /= profile_info.tree_operation_step_times.size();

    // add all metrics to csv
    row += model_metadata.llm_model_name + ",";
    row += target_partition + ",";
    row += std::to_string(max_tree_depth) + ",";
    row += std::to_string(online_tree_update) + ",";
    row += matching_strategy + ",";
    row += std::to_string(max_requests_per_batch) + ",";
    row += std::to_string(max_tokens_per_batch) + ",";

    // avg speculated length
    row += std::to_string(mean_speculated_tokens) + ",";
    // avg accepted candidate length
    row += std::to_string(mean_accepted_candidate_length) + ",";
    // avg acceptance rate
    row += std::to_string(mean_acceptance_rate) + ",";
    // avg prefix length
    row += std::to_string(mean_prefix_length) + ",";

    row += std::to_string((double)total_time / 1000.0) + ",";
    row += std::to_string(throughput_tokens_per_sec) + ",";
    row += std::to_string(mean_generated_tokens_per_step) + ",";
    row += std::to_string(mean_decoding_steps) + ",";
    row += std::to_string(mean_output_length) + ",";
    row += std::to_string(mean_e2e_latency) + ",";
    row += std::to_string(mean_llm_ttft) + ",";
    row += std::to_string(mean_llm_tpot) + ",";
    // mean_tree_operation_time_per_step
    row += std::to_string(mean_tree_operation_time_per_step) + ",";
    // mean_speculated_tokens_per_req
    row += vectorToString(mean_speculated_tokens_per_req) + ",";
    // mean_accepted_candidate_length_per_req
    row += vectorToString(mean_accepted_candidate_length_per_req) + ",";
    // mean_acceptance_rate_per_req
    row += vectorToString(mean_acceptance_rate_per_req) + ",";
    // mean_prefix_length_per_req
    row += vectorToString(mean_prefix_length_per_req) + ",";

    // generated_tokens_per_step
    row += vectorToStringInt(profile_info.generated_tokens_per_step) + ",";
    // num_active_requests_per_step
    row += vectorToStringInt(profile_info.requests_per_step);

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
    file << row << "\n";
    file.close();
  }
  */

  std::string header = "llm,partition,max_tree_depth,online_tree_update,matching_strategy,max_requests_per_batch,max_tokens_per_batch,request_guid,request_step_idx,timestamp,num_speculated_tokens,num_accepted_tokens,prefix_length,speculation_score,num_generated_tokens";  
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
  for (const auto& info : new_profiling_info) {
    file << model_metadata.llm_model_name + ",";
    file << target_partition + ",";
    file << std::to_string(max_tree_depth) + ",";
    file << std::to_string(online_tree_update) + ",";
    file << matching_strategy + ",";
    file << std::to_string(max_requests_per_batch) + ",";
    file <<  std::to_string(max_tokens_per_batch) + ",";
    file << info.request_guid << "," 
          << info.request_step_idx << ","
          << info.timestamp << ","
          << info.num_speculated_tokens << ","
          << info.num_accepted_tokens << ","
          << info.prefix_length << ","
          << info.speculation_score << ","
          << info.num_generated_tokens << "\n";
  }
  file.close();

  // Execution fence
  {
    Future future = runtime->issue_execution_fence(ctx);
    future.get_void_result();
  }

  // float* data
  std::cout << "----------inference finished--------------" << std::endl;
}

void FlexFlow::register_custom_tasks() {}
