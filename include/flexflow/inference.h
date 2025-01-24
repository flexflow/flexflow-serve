/* Copyright 2022 CMU, Stanford, Facebook, LANL
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

#pragma once
#include "flexflow/batch_config.h"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace FlexFlow {

struct GenerationConfig {
  bool do_sample = false;
  bool spec_sample = false;
  float temperature = 0.8;
  // top-p renormalization
  float topp = 0.6;
  // top-k renormalization
  int topk = 16;
  GenerationConfig(bool _do_sample = false,
                   float _temperature = 0.8,
                   float _topp = 0.6,
                   bool _spec_sample = false,
                   int _topk = 16)
      : do_sample(_do_sample), temperature(_temperature), topp(_topp),
        spec_sample(_spec_sample), topk(_topk) {
    assert(temperature > 0.0);
    assert(topk <= BatchConfig::MAX_K_LOGITS);
  }
};

struct GenerationRequest {
  std::string prompt;
  bool add_special_tokens = true;
  double slo_ratio;
  double emission_time_ms;

  GenerationRequest(std::string const &prompt_,
                    double slo_ratio_,
                    double emission_time_ms_,
                    bool add_special_tokens_ = true)
      : prompt(prompt_), slo_ratio(slo_ratio_),
        emission_time_ms(emission_time_ms_),
        add_special_tokens(add_special_tokens_) {}
};

struct GenerationResult {
  using RequestGuid = BatchConfig::RequestGuid;
  using TokenId = BatchConfig::TokenId;
  RequestGuid guid;
  std::string input_text;
  std::string output_text;
  std::vector<TokenId> input_tokens;
  std::vector<TokenId> output_tokens;
  double slo_ratio;
  double emission_time_ms;
  int decoding_steps;
};

// Contains the configuration for how to emit requests to the server,
// managing the request arrival rate.
class EmissionMachine {
public:
  enum class EmissionMode { Constant, Poisson, Trace };
  EmissionMode mode;
  double elapsed_time_ms;
  double last_request_time_ms;
  double req_per_s;
  std::vector<std::pair<double, double>> slo_ratios;

  EmissionMachine(EmissionMode mode_,
                  double req_per_s_,
                  std::vector<std::pair<double, double>> slo_ratios_)
      : mode(mode_), elapsed_time_ms(0), last_request_time_ms(0),
        req_per_s(req_per_s_), slo_ratios(slo_ratios_) {
    // cumulate the slo ratios for sampling
    for (size_t i = 1; i < slo_ratios.size(); i++) {
      slo_ratios[i].second += slo_ratios[i - 1].second;
    }
  }
  void wait_until_next_request();

  // Simulate next request arrival time
  virtual double get_next_interval_ms() = 0;
  virtual double sample_slo_ratio();
  double get_elapsed_time_ms();
};

class EmissionTrace {
public:
  std::string prompt;
  int input_length, output_length;
  double slo_ratio;
  double emission_time_ms;

  EmissionTrace(std::string prompt_,
                int input_length_,
                int output_length_,
                double slo_ratio_,
                double emission_time_ms_)
      : prompt(prompt_), input_length(input_length_),
        output_length(output_length_), slo_ratio(slo_ratio_),
        emission_time_ms(emission_time_ms_) {}
  EmissionTrace(GenerationResult const &result)
      : prompt(result.input_text), input_length(result.input_tokens.size()),
        output_length(result.output_tokens.size()), slo_ratio(result.slo_ratio),
        emission_time_ms(result.emission_time_ms) {}
  EmissionTrace(json const &json_obj);

  json to_json() const;
};

class ConstantEmissionMachine : public EmissionMachine {
public:
  double interval_ms;

  ConstantEmissionMachine(double req_per_s_,
                          std::vector<std::pair<double, double>> slo_ratios_)
      : EmissionMachine(EmissionMode::Constant, req_per_s_, slo_ratios_),
        interval_ms(req_per_s_ > 0 ? 1e3 / req_per_s_ : 0) {}

  double get_next_interval_ms() override;
};

class PoissonEmissionMachine : public EmissionMachine {
public:
  double lambda;

  PoissonEmissionMachine(double req_per_s_,
                         std::vector<std::pair<double, double>> slo_ratios_)
      : EmissionMachine(EmissionMode::Poisson, req_per_s_, slo_ratios_),
        lambda(req_per_s_) {}

  double get_next_interval_ms() override;
};

class TraceEmissionMachine : public EmissionMachine {
public:
  std::vector<double> timestamps, ratios;
  size_t idx;

  TraceEmissionMachine(std::vector<double> const &timestamps_,
                       std::vector<double> const &ratios_)
      : EmissionMachine(EmissionMode::Trace, 0, {}), timestamps(timestamps_),
        ratios(ratios_), idx(0) {}

  double get_next_interval_ms() override;
  double sample_slo_ratio() override;
};

struct RotaryEmbeddingMeta {
  bool apply_rotary_embedding = false;
  float rope_theta = 10000.0f;
  std::string rope_type = "default";
  float factor = 8.0f;
  float low_freq_factor = 1.0f;
  float high_freq_factor = 4.0f;
  int original_max_position_embeddings = 8192;

  RotaryEmbeddingMeta(bool apply_rotary_embedding_ = false,
                      float rope_theta_ = 10000.0f,
                      std::string rope_type_ = "default",
                      float factor_ = 8.0f,
                      float low_freq_factor_ = 1.0f,
                      float high_freq_factor_ = 4.0f,
                      int original_max_position_embeddings_ = 8192)
      : apply_rotary_embedding(apply_rotary_embedding_),
        rope_theta(rope_theta_), rope_type(rope_type_), factor(factor_),
        low_freq_factor(low_freq_factor_), high_freq_factor(high_freq_factor_),
        original_max_position_embeddings(original_max_position_embeddings_) {}

  friend std::ostream &operator<<(std::ostream &os,
                                  RotaryEmbeddingMeta const &meta) {
    os << std::boolalpha // To print bool as true/false instead of 1/0
       << "RotaryEmbeddingMeta {\n"
       << "  apply_rotary_embedding: " << meta.apply_rotary_embedding << ",\n"
       << "  rope_theta: " << meta.rope_theta << ",\n"
       << "  rope_type: \"" << meta.rope_type << "\",\n"
       << "  factor: " << meta.factor << ",\n"
       << "  low_freq_factor: " << meta.low_freq_factor << ",\n"
       << "  high_freq_factor: " << meta.high_freq_factor << ",\n"
       << "  original_max_position_embeddings: "
       << meta.original_max_position_embeddings << "\n"
       << "}";
    return os;
  }
};

std::string join_path(std::vector<std::string> const &paths);

} // namespace FlexFlow
