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

#ifndef _FLEXFLOW_ATTENTION_CONFIG_H_
#define _FLEXFLOW_ATTENTION_CONFIG_H_
#include "flexflow/batch_config.h"

namespace FlexFlow {

constexpr uint32_t kPagesize = 64;

inline int ceilDiv(int const a, int const b) {
  assert(b != 0 && "Attempting to divide by 0");
  assert(a >= 0 && b > 0 && "Expected non-negative numbers");
  return (a + b - 1) / b;
}

inline int round_up_pages(int const num_elements) {
  return ceilDiv(num_elements, kPagesize);
}

#define DISPATCH_HEADDIM(head_dim, HEAD_DIM, ...)                              \
  switch (head_dim) {                                                          \
    case 64: {                                                                 \
      constexpr size_t HEAD_DIM = 64;                                          \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case 128: {                                                                \
      constexpr size_t HEAD_DIM = 128;                                         \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case 256: {                                                                \
      constexpr size_t HEAD_DIM = 256;                                         \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    default: {                                                                 \
      std::ostringstream err_msg;                                              \
      err_msg << "Unsupported head_dim: " << head_dim;                         \
      throw std::invalid_argument(err_msg.str());                              \
    }                                                                          \
  }

class AttentionMetaData {
public:
  AttentionMetaData() {
    num_q_heads_ = 0;
    num_kv_heads_ = 0;
    head_dim_ = 0;
    q_indptr_dec = nullptr;
    kv_indptr_dec = nullptr;
    kv_indices_dec = nullptr;
    kv_last_page_len_dec = nullptr;
    workspace_dec = nullptr;
    float_workspace_dec = nullptr;
    int_workspace_dec = nullptr;
    q_indptr_pref = nullptr;
    kv_indptr_pref = nullptr;
    kv_indices_pref = nullptr;
    kv_last_page_len_pref = nullptr;
    workspace_pref = nullptr;
    float_workspace_pref = nullptr;
    int_workspace_pref = nullptr;
    workspace_size = 0;
    float_workspace_size = 0;
    int_workspace_size = 0;
    mem_size_ = 0;
    enabled_ = false;
  }
  AttentionMetaData(AttentionMetaData const &rhs) {
    num_q_heads_ = rhs.num_q_heads_;
    num_kv_heads_ = rhs.num_kv_heads_;
    head_dim_ = rhs.head_dim_;
    q_indptr_dec = rhs.q_indptr_dec;
    kv_indptr_dec= rhs.kv_indptr_dec;
    kv_indices_dec = rhs.kv_indices_dec;
    kv_last_page_len_dec = rhs.kv_last_page_len_dec;
    workspace_dec = rhs.workspace_dec;
    float_workspace_dec = rhs.float_workspace_dec;
    int_workspace_dec = rhs.int_workspace_dec;
    q_indptr_pref = rhs.q_indptr_pref;
    kv_indptr_pref= rhs.kv_indptr_pref;
    kv_indices_pref = rhs.kv_indices_pref;
    kv_last_page_len_pref = rhs.kv_last_page_len_pref;
    workspace_pref = rhs.workspace_pref;
    float_workspace_pref = rhs.float_workspace_pref;
    int_workspace_pref = rhs.int_workspace_pref;
    workspace_size = rhs.workspace_size;
    float_workspace_size = rhs.float_workspace_size;
    int_workspace_size = rhs.int_workspace_size;
    mem_size_ = rhs.mem_size_;
    enabled_ = rhs.enabled_;
    decode_handler_collections = rhs.decode_handler_collections;
    prompt_handler_collections = rhs.prompt_handler_collections;
  }

  size_t mem_size() {
    if (mem_size_ > 0) {
      return mem_size_;
    }
    size_t batch_size = BatchConfig::max_requests_per_batch();
    size_t max_num_pages = round_up_pages(BatchConfig::max_sequence_length());
    size_t indices_size = std::max(
        (batch_size + 1) * 4 + max_num_pages * batch_size, 1ul * 1024 * 1024);

    float_workspace_size = 128 * 1024 * 1024; // 128 MB
    int_workspace_size = 8 * 1024 * 1024;     // 8 MB
    workspace_size =
        float_workspace_size + int_workspace_size; // float + int workspace

    mem_size_ = alignTo(2*(sizeof(int32_t) * indices_size + workspace_size),
                        16);
    return mem_size_;
  }

  void assign_address(void *ptr, int size) {
    if (ptr == nullptr) {
      q_indptr_dec = nullptr;
      kv_indptr_dec = nullptr;
      kv_indices_dec = nullptr;
      kv_last_page_len_dec = nullptr;
      workspace_dec = nullptr;
      float_workspace_dec = nullptr;
      int_workspace_dec = nullptr;
      q_indptr_pref = nullptr;
      kv_indptr_pref = nullptr;
      kv_indices_pref = nullptr;
      kv_last_page_len_pref = nullptr;
      workspace_pref = nullptr;
      float_workspace_pref = nullptr;
      int_workspace_pref = nullptr;
      return;
    }
    assert(size >= mem_size() &&
           "Insufficient memory size for attention metadata");
    size_t batch_size = BatchConfig::max_requests_per_batch();
    size_t max_num_pages = round_up_pages(BatchConfig::max_sequence_length());
    size_t indices_size = std::max(
        (batch_size + 1) * 4 + max_num_pages * batch_size, 1ul * 1024 * 1024);

    q_indptr_dec = static_cast<int32_t *>(ptr);
    kv_indptr_dec = q_indptr_dec + batch_size + 1;
    kv_indices_dec = kv_indptr_dec + batch_size + 1;
    kv_last_page_len_dec = kv_indices_dec + max_num_pages * batch_size;
    q_indptr_pref = static_cast<int32_t *>(ptr) + indices_size;
    kv_indptr_pref = q_indptr_pref + batch_size + 1;
    kv_indices_pref = kv_indptr_pref + batch_size + 1;
    kv_last_page_len_pref = kv_indices_pref + max_num_pages * batch_size;
    
    workspace_dec = static_cast<void *>(static_cast<uint8_t *>(ptr) +
                                    sizeof(int32_t) * indices_size * 2);
    float_workspace_dec = workspace_dec;
    int_workspace_dec = static_cast<void *>(static_cast<uint8_t *>(workspace_dec) +
                                        float_workspace_size);
    workspace_pref = static_cast<void *>(static_cast<uint8_t *>(ptr) +
                                    sizeof(int32_t) * indices_size * 2 +
                                    workspace_size);
    float_workspace_pref = workspace_pref;
    int_workspace_pref = static_cast<void *>(static_cast<uint8_t *>(workspace_pref) +
                                        float_workspace_size);
  }

  void set_num_q_heads(uint32_t const num_q_heads) {
    num_q_heads_ = num_q_heads;
  }
  void set_num_kv_heads(uint32_t const num_kv_heads) {
    num_kv_heads_ = num_kv_heads;
  }
  void set_head_dim(uint32_t const head_dim) {
    head_dim_ = head_dim;
  }
  uint32_t num_q_heads() const {
    return num_q_heads_;
  }
  uint32_t num_kv_heads() const {
    return num_kv_heads_;
  }
  uint32_t head_dim() const {
    return head_dim_;
  }

  void set_enabled(bool const enabled) {
    enabled_ = enabled;
  }
  bool enabled() const {
    return enabled_;
  }

  uint32_t num_q_heads_;
  uint32_t num_kv_heads_;
  uint32_t head_dim_;

  int32_t *q_indptr_dec;
  int32_t *kv_indptr_dec;
  int32_t *kv_indices_dec;
  int32_t *kv_last_page_len_dec;
  uint8_t *custom_mask_dec;
  void *workspace_dec;
  void *float_workspace_dec;
  void *int_workspace_dec;
  int32_t *q_indptr_pref;
  int32_t *kv_indptr_pref;
  int32_t *kv_indices_pref;
  int32_t *kv_last_page_len_pref;
  uint8_t *custom_mask_pref;
  void *workspace_pref;
  void *float_workspace_pref;
  void *int_workspace_pref;
  
  size_t workspace_size;
  size_t float_workspace_size;
  size_t int_workspace_size;
  size_t mem_size_;

  // batchsize -> handler
  bool enabled_;
  std::unordered_map<int, void *> decode_handler_collections;
  std::unordered_map<int, void *> prompt_handler_collections;
};
} // namespace FlexFlow

#endif // _FLEXFLOW_ATTENTION_CONFIG_H_