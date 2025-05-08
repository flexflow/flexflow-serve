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
#include "models/starcoder.h"
#include <wordexp.h>

#include <nlohmann/json.hpp>

using namespace FlexFlow;
using namespace Legion;
using json = nlohmann::json;

Legion::Logger log_app("llama");

struct FilePaths {
  std::string cache_folder_path;
  std::string prompt_file_path;
  std::string dataset_file_path;
  std::string output_file_path;
  std::string profiling_folder_path;
};

void parse_input_args(char **argv,
                      int argc,
                      FilePaths &paths,
                      std::string &llm_model_name,
                      std::string &peft_model_name,
                      bool &use_full_precision,
                      bool &verbose,
                      bool &do_sample,
                      PeftSupportMode &peft_support_mode,
                      float &temperature,
                      float &topp,
                      int &max_requests_per_batch,
                      int &max_tokens_per_batch,
                      int &max_sequence_length,
                      int &num_kv_cache_slots,
                      int &max_training_epochs,
                      int &num_layers_per_finetuning_step,
                      bool &run_warmup) {
  for (int i = 1; i < argc; i++) {
    // llm model type
    if (!strcmp(argv[i], "-llm-model")) {
      llm_model_name = std::string(argv[++i]);
      for (char &c : llm_model_name) {
        c = std::tolower(c);
      }
      continue;
    }
    if (!strcmp(argv[i], "--peft-support-mode")) {
      std::string mode = argv[++i];
      // Convert to lowercase for comparison
      for (char &c : mode) {
        c = std::tolower(c);
      }
      if (mode == "disabled") {
        peft_support_mode = PEFT_DISABLED;
      } else if (mode == "inference_only" || mode == "inference-only") {
        peft_support_mode = PEFT_INFERENCE_ONLY;
      } else if (mode == "coserving") {
        peft_support_mode = COSERVING;
      } else if (mode == "temporal_sharing" || mode == "temporal-sharing") {
        peft_support_mode = TEMPORAL_SHARING;
      } else if (mode == "spatial_sharing" || mode == "spatial-sharing") {
        peft_support_mode = SPATIAL_SHARING;
      } else if (mode == "spatial_sharing_limited" || mode == "spatial-sharing-limited") {
        peft_support_mode = SPATIAL_SHARING_LIMITED;
      } else if (mode == "temporal_sharing_limited" || mode == "temporal-sharing-limited") {
        peft_support_mode = TEMPORAL_SHARING_LIMITED;
      } else if (mode == "spatial_sharing_separate_tasks" || mode == "spatial-sharing-separate-tasks") {
        peft_support_mode = SPATIAL_SHARING_SEPARATE_TASKS;
      } else {
        std::cerr << "Unknown peft support mode: " << mode << std::endl;
        assert(false && "Invalid peft support mode");
      }
      continue;
    }
    if (!strcmp(argv[i], "-peft-model")) {
      peft_model_name = std::string(argv[++i]);
      for (char &c : peft_model_name) {
        c = std::tolower(c);
      }
      continue;
    }
    // cache folder
    if (!strcmp(argv[i], "-cache-folder")) {
      paths.cache_folder_path = std::string(argv[++i]);
      continue;
    }
    // prompts
    if (!strcmp(argv[i], "-prompt")) {
      paths.prompt_file_path = std::string(argv[++i]);
      continue;
    }
    // dataset for finetuning
    if (!strcmp(argv[i], "-finetuning-dataset")) {
      paths.dataset_file_path = std::string(argv[++i]);
      continue;
    }
    // output file
    if (!strcmp(argv[i], "-output-file")) {
      paths.output_file_path = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-profiling-folder")) {
      paths.profiling_folder_path = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--use-full-precision")) {
      use_full_precision = true;
      continue;
    }
    if (!strcmp(argv[i], "--warmup")) {
      run_warmup = true;
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
    if (!strcmp(argv[i], "--temperature")) {
      temperature = std::stof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--topp")) {
      topp = std::stof(argv[++i]);
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
    // num kv cache slots for inference (i.e. number of tokens across all
    // requests)
    if (!strcmp(argv[i], "--num-kv-cache-slots")) {
      num_kv_cache_slots = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--max-training-steps")) {
      max_training_epochs = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--num-layers-per-finetuning-step")) {
      num_layers_per_finetuning_step = std::stoi(argv[++i]);
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

std::vector<Request> make_warmup_requests(int num_inf_request,
                                          int num_finetuning_steps,
                                          PEFTModelID *peft_model_id) {
  std::vector<Request> warmup_requests;

  for (int i = 0; i < num_inf_request; i++) {
    Request inference_req;
    inference_req.benchmarking_tokens = 512;
    inference_req.max_new_tokens = 30;
    inference_req.warmup = true;
    warmup_requests.push_back(inference_req);
  }
  Request finetuning_req;
  finetuning_req.req_type = RequestType::REQ_FINETUNING;
  finetuning_req.benchmarking_tokens = 1024;
  finetuning_req.max_length = 1024;
  finetuning_req.warmup = true;
  finetuning_req.peft_model_id =
      (peft_model_id != nullptr) ? *peft_model_id : PEFTModelID::NO_ID;
  finetuning_req.peft_finetuning_info.max_training_epochs =
      num_finetuning_steps;
  warmup_requests.push_back(finetuning_req);
  return warmup_requests;
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
  std::string llm_model_name, peft_model_name;
  bool use_full_precision = false;
  bool verbose = false;
  bool do_sample = false;
  ffconfig.peft_support_mode = PEFT_INFERENCE_ONLY;
  float temperature = 0.0f;
  float topp = 0.0f;
  int max_requests_per_batch = 1;
  int max_tokens_per_batch = 128;
  int max_sequence_length = 256;
  int max_training_epochs = 2;
  int num_layers_per_finetuning_step = -1;
  bool run_warmup = false;
  int num_kv_cache_slots = -1;

  InputArgs const &command_args = HighLevelRuntime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  parse_input_args(argv,
                   argc,
                   file_paths,
                   llm_model_name,
                   peft_model_name,
                   use_full_precision,
                   verbose,
                   do_sample,
                   ffconfig.peft_support_mode,
                   temperature,
                   topp,
                   max_requests_per_batch,
                   max_tokens_per_batch,
                   max_sequence_length,
                   num_kv_cache_slots,
                   max_training_epochs,
                   num_layers_per_finetuning_step,
                   run_warmup);

  if (num_kv_cache_slots == -1) {
    num_kv_cache_slots = max_sequence_length * max_requests_per_batch;
  }

  assert(ffconfig.data_parallelism_degree * ffconfig.tensor_parallelism_degree *
             ffconfig.pipeline_parallelism_degree ==
         ffconfig.numNodes * ffconfig.workersPerNode);
  if (!file_paths.dataset_file_path.empty()) {
    ffconfig.peft_support_mode = COSERVING;
  }

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
  if (!peft_enabled(ffconfig.peft_support_mode)) {
    std::cerr << "Running PEFT script with PEFT not enabled" << std::endl;
    assert(false);
  }
  if (peft_enabled(ffconfig.peft_support_mode) && peft_model_name.empty()) {
    std::cout << "PEFT enabled, but no PEFT model id passed" << std::endl;
    assert(false);
  } else if (!peft_enabled(ffconfig.peft_support_mode) && !peft_model_name.empty()) {
    std::cout << "PEFT model id passed, but PEFT is not enabled" << std::endl;
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
        str == "Qwen2ForCausalLM" || str == "MistralForCausalLM") {
      model_type = ModelType::LLAMA;
      break;
    } else if (str == "OPTForCausalLM") {
      model_type = ModelType::OPT;
      break;
    } else if (str == "RWForCausalLM" || str == "FalconForCausalLM") {
      model_type = ModelType::FALCON;
      break;
    } else if (str == "GPTBigCodeForCausalLM") {
      model_type = ModelType::STARCODER;
      break;
    } else if (str == "MPTForCausalLM") {
      model_type = ModelType::MPT;
      break;
    }
  }
  int bos_token_id = model_config.find("bos_token_id") == model_config.end()
                         ? -1
                         : (int)model_config.at("bos_token_id");
  // parse eos token id, which can be either a single integer or an array of
  // integers. Convert to std::vector<int>
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

  // load PEFT config
  LoraLinearConfig peft_config =
      peft_model_name.empty()
          ? LoraLinearConfig::EmptyConfig
          : LoraLinearConfig(file_paths.cache_folder_path, peft_model_name);

  LoraOptimizerConfig *optim_config = nullptr;
  if (peft_finetuning_enabled(ffconfig.peft_support_mode)) {
    // float sgd_learning_rate = 2e-1;
    float sgd_learning_rate = 0.001f;
    optim_config = new LoraSGDOptimizerConfig(sgd_learning_rate);
  }
  LoraLinearConfig peft_config_finetuning =
      !peft_finetuning_enabled(ffconfig.peft_support_mode)
          ? LoraLinearConfig::EmptyConfig
          : LoraLinearConfig(file_paths.cache_folder_path,
                             peft_model_name,
                             true /*trainable*/,
                             optim_config,
                             false /*init_lora_weights*/,
                             llm_model_name,
                             use_full_precision ? "fp32" : "fp16");

  GenerationConfig generationConfig(do_sample, temperature, topp);
  RequestManager *rm = RequestManager::get_request_manager();
  rm->set_verbose(verbose);
  rm->set_max_requests_per_batch(
      max_requests_per_batch +
      (int)peft_finetuning_enabled(ffconfig.peft_support_mode)); // add one slot for finetuning if needed
  // rm->set_max_concurrent_adapters(max_requests_per_batch +
  //                                 (int)peft_finetuning_enabled(ffconfig.peft_support_mode));
  rm->set_max_concurrent_adapters(1);
  rm->set_max_tokens_per_batch(max_tokens_per_batch);
  rm->set_max_sequence_length(max_sequence_length);
  rm->register_tokenizer(
      model_type, bos_token_id, eos_token_ids, tokenizer_filepath);
  rm->register_output_filepath(file_paths.output_file_path);
  rm->set_peft_support_mode(ffconfig.peft_support_mode);

  FFModel model(ffconfig, ffconfig.cpu_offload);
  model.set_num_kv_cache_pages(compute_num_kv_cache_pages_needed(
      max_sequence_length, max_requests_per_batch, false));
  if (model_type == ModelType::LLAMA) {
    LLAMA::create_llama_model(model,
                              config_filepath,
                              weights_filepath,
                              INC_DECODING_MODE,
                              generationConfig,
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
  } else if (model_type == ModelType::STARCODER) {
    STARCODER::create_starcoder_model(model,
                                      config_filepath,
                                      weights_filepath,
                                      INC_DECODING_MODE,
                                      generationConfig,
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

  rm->set_num_transformer_layers(model.current_transformer_layer_id + 1);
  if (num_layers_per_finetuning_step > 0) {
    rm->set_num_layers_per_finetuning_step(num_layers_per_finetuning_step);
  }

  // Start background server
  rm->start_background_server(&model);

  // Add PEFT adapter(s)
  PEFTModelID *peft_model_id = nullptr, *peft_model_id_finetuning = nullptr;
  if (!peft_model_name.empty() && !peft_finetuning_enabled(ffconfig.peft_support_mode)) {
    peft_model_id = model.register_peft_adapter(peft_config);
  }
  if (peft_finetuning_enabled(ffconfig.peft_support_mode)) {
    peft_model_id_finetuning =
        model.register_peft_adapter(peft_config_finetuning);
  }

  if (run_warmup) {
    std::vector<Request> warmup_requests =
        make_warmup_requests(10, 1000, peft_model_id_finetuning);
    std::vector<GenerationResult> warmup_result =
        model.generate(warmup_requests);
    rm->set_inference_finished(false); // reset inference finished flag
    std::cout << "----------warmup finished--------------" << std::endl;
  }

  // Run workload
  {
    std::vector<Request> requests;

    // Add inference requests
    if (!file_paths.prompt_file_path.empty()) {
      using json = nlohmann::json;
      std::ifstream file_handle(file_paths.prompt_file_path);
      assert(file_handle.good() && "Prompt file does not exist.");
      json prompt_json = json::parse(file_handle,
                                     /*parser_callback_t */ nullptr,
                                     /*allow_exceptions */ true,
                                     /*ignore_comments */ true);
      int total_num_requests = 0;
      for (auto &prompt : prompt_json) {
        std::string text = prompt.get<std::string>();
        printf("Inference prompt[%d]: %s\n", total_num_requests, text.c_str());
        Request inference_req;
        inference_req.prompt = text;
        inference_req.max_new_tokens = 128;
        inference_req.peft_model_id =
            (peft_model_id != nullptr) ? *peft_model_id : PEFTModelID::NO_ID;
        requests.push_back(inference_req);
        total_num_requests++;
      }
    }

    // Add fine-tuning request
    if (peft_finetuning_enabled(ffconfig.peft_support_mode)) {
      assert(!file_paths.dataset_file_path.empty() &&
             "Dataset file path is required for fine-tuning.");
      printf("Finetuning request with dataset %s\n",
             file_paths.dataset_file_path.c_str());
      Request fine_tuning_req;
      fine_tuning_req.req_type = RequestType::REQ_FINETUNING;
      fine_tuning_req.peft_model_id = (peft_model_id_finetuning != nullptr)
                                          ? *peft_model_id_finetuning
                                          : PEFTModelID::NO_ID;
      fine_tuning_req.peft_finetuning_info.dataset_filepath =
          file_paths.dataset_file_path;
      fine_tuning_req.peft_finetuning_info.max_training_epochs =
          max_training_epochs;
      requests.push_back(fine_tuning_req);
    }
    std::vector<GenerationResult> result = model.generate(requests);
  }

  // terminate the request manager by stopping the background thread
  rm->terminate_background_server();

  // Execution fence
  {
    Future future = runtime->issue_execution_fence(ctx);
    future.get_void_result();
  }

  if (!file_paths.profiling_folder_path.empty()) {
    std::cout << "Saving profiling info..." << std::endl;
    std::string dataset_name;
    // set dataset name to "wildchat" if the prompt file path contains
    // "wildchat"
    if (file_paths.prompt_file_path.find("wildchat") != std::string::npos) {
      dataset_name = "wildchat";
    } else if (file_paths.prompt_file_path.find("sharegpt") !=
               std::string::npos) {
      dataset_name = "sharegpt";
    } else {
      dataset_name = "unknown";
    }
    rm->save_profiling_info_to_csv(file_paths.profiling_folder_path,
                                   dataset_name,
                                   llm_model_name,
                                   model.config.tensor_parallelism_degree,
                                   max_requests_per_batch,
                                   max_tokens_per_batch,
                                   num_kv_cache_slots,
                                   0.0,                  // arrival rate
                                   run_warmup ? 10 : 0); // num_warmup_requests
  }

  if (peft_model_id != nullptr) {
    free(peft_model_id);
  }
  if (peft_model_id_finetuning != nullptr) {
    free(peft_model_id_finetuning);
  }

  std::cout << "----------inference finished--------------" << std::endl;

  // free tokenizer space in memory
}

void FlexFlow::register_custom_tasks() {}
