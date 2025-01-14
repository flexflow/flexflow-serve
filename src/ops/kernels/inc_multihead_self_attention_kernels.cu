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
#include "flexflow/batch_config.h"
#include <cassert>
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include "cuComplex.h"
#endif
#include "flashinfer/pos_enc.cuh"
#include "flexflow/attention_config.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/inc_multihead_self_attention.h"
#include "flexflow/ops/kernels/decompress_kernels.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_kernels.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_utils.cuh"
#include "flexflow/utils/cuda_helper.h"
#include <math_constants.h>

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

using flashinfer::BatchQKApplyLlama31Rotary;
using flashinfer::BatchQKApplyRotary;

#define WARP_SIZE 32

namespace Kernels {
namespace IncMultiHeadAttention {

// only used by MPT model. https://arxiv.org/abs/2108.12409
template <typename DT>
__global__ void apply_position_bias_qkprd(DT *input_ptr,
                                          int num_tokens,
                                          int num_total_tokens,
                                          int num_heads,
                                          int global_num_q_heads,
                                          int shard_id) {
  CUDA_KERNEL_LOOP(i, num_tokens * num_total_tokens * num_heads) {
    // get head_idx,
    int head_idx = i / (num_tokens * num_total_tokens) + (num_heads * shard_id);
    int position_idx = (i / num_tokens) % num_total_tokens;
    position_idx = position_idx + 1 - num_total_tokens;
    // 8 is alibi_bias_max in
    // https://huggingface.co/mosaicml/mpt-30b/blob/main/config.json
    float base = (float)(head_idx + 1) * 8 / global_num_q_heads;
    float slopes = 1.0 / pow(2, base);
    // if(i == 0){
    //   printf("see position: %d, %f, %f, %f\n", position_idx, base, slopes,
    //   position_idx * slopes);
    // }
    input_ptr[i] += static_cast<DT>(position_idx * slopes);
  }
}

template <typename DT>
__global__ void apply_proj_bias_w(DT *input_ptr,
                                  DT const *bias_ptr,
                                  int num_tokens,
                                  int qkv_weight_size,
                                  int o_dim) {
  CUDA_KERNEL_LOOP(i, num_tokens * o_dim) {
    int bias_idx = qkv_weight_size + i % o_dim;
    input_ptr[i] += bias_ptr[bias_idx];
  }
}

template <typename DT>
__global__ void apply_proj_bias_qkv(DT *input_ptr,
                                    DT const *bias_ptr,
                                    int shard_id,
                                    int num_tokens,
                                    int qk_dim,
                                    int v_dim,
                                    int global_num_q_heads,
                                    int num_q_heads,
                                    bool scaling_query,
                                    float scaling_factor,
                                    int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_tokens * hidden_size * QKV_WEIGHT_NUM) {
    // for simplicity, assume q, k, v is in same shape
    // 0->q, 1->k, 2->v
    // int qkv_index = i / (num_tokens * qk_dim) % 3;

    int token_idx = i / (hidden_size * QKV_WEIGHT_NUM);
    size_t in_token_idx = i - token_idx * hidden_size * QKV_WEIGHT_NUM;

    int qkv_index = in_token_idx / hidden_size;

    int proj_size = qkv_index == 0 ? qk_dim : qk_dim;

    int head_idx =
        (in_token_idx - qkv_index * num_q_heads * proj_size) / proj_size;
    int global_head_idx = head_idx + shard_id * num_q_heads;

    size_t pre_length =
        qkv_index == 0
            ? 0
            : (qkv_index == 1 ? qk_dim * global_num_q_heads
                              : qk_dim * global_num_q_heads * KV_WEIGHT_NUM);

    size_t bias_idx = pre_length + global_head_idx * proj_size + i % proj_size;

    input_ptr[i] += bias_ptr[bias_idx];

    if (scaling_query && qkv_index == 0) {
      input_ptr[i] *= scaling_factor;
    }
  }
}

template <typename DT>
__global__ void scaling_query_kernel(DT *input_ptr,
                                     int qk_dim,
                                     int num_tokens,
                                     int num_q_heads,
                                     float scaling_factor,
                                     int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_tokens * hidden_size) {
    int token_idx = i / hidden_size;
    input_ptr[i % hidden_size + token_idx * hidden_size * QKV_WEIGHT_NUM] *=
        scaling_factor;
  }
}

template <typename DT>
__global__ void
    update_qkv_in_batch_kernel(DT *qkv_proj_array,
                               DT *qTmp_ptr,
                               DT *keyCache_ptr,
                               DT *valueCache_ptr,
                               BatchConfig::PerTokenInfo const *tokenInfos,
                               int const max_num_pages,
                               int num_q_heads,
                               int num_kv_heads,
                               int head_dim,
                               int num_new_tokens) {
  int const q_hidden_size = num_q_heads * head_dim;
  int const temp_kv_hidden_size = num_q_heads * head_dim; // temporary hard code
  int const kv_hidden_size = num_kv_heads * head_dim;
  
  int const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int const token_idx = thread_idx / q_hidden_size;
  int const offset = thread_idx % q_hidden_size;
  if (token_idx >= num_new_tokens) {
    return;
  }

  int const req_idx = tokenInfos[token_idx].request_index;
  int token_abs_idx = tokenInfos[token_idx].abs_depth_in_request;

  size_t from_idx = token_idx * (q_hidden_size + temp_kv_hidden_size * 2);
  qTmp_ptr[token_idx * q_hidden_size + offset] = static_cast<DT>(qkv_proj_array[from_idx + offset]);

  if (offset < kv_hidden_size) {
    size_t to_k_idx = get_k_entry_offset(req_idx, token_abs_idx, max_num_pages, num_kv_heads, head_dim),
           to_v_idx = get_v_entry_offset(req_idx, token_abs_idx, max_num_pages, num_kv_heads, head_dim);
    assert(to_k_idx == to_v_idx);
    // key and value cache should be stored interleaved
    // int const stride = num_q_heads / num_kv_heads;
    // int const kv_offset =
    //     offset / head_dim * stride * head_dim + offset % head_dim;

    // size_t key_cache_size = num_kv_heads * head_dim *
    //                         8 * max_num_pages *
    //                         kPagesize;

    keyCache_ptr[to_k_idx + offset] = static_cast<half>(qkv_proj_array[from_idx + q_hidden_size + offset]);
    valueCache_ptr[to_v_idx + offset] = static_cast<half>(qkv_proj_array[from_idx + q_hidden_size + temp_kv_hidden_size + offset]);
  }
}

template <typename DT>
void update_qkv_in_batch(IncMultiHeadSelfAttentionMeta const *m,
                         BatchConfig const *bc,
                         cudaStream_t stream) {
  int num_new_tokens = bc->num_active_tokens();
  if (num_new_tokens == 0) {
    return;
  }
  int parallelism = m->qProjSize * m->num_q_heads * num_new_tokens;
  int const max_num_pages = round_up_pages(BatchConfig::max_sequence_length());
  update_qkv_in_batch_kernel<<<GET_BLOCKS(parallelism),
                               min(CUDA_NUM_THREADS, parallelism),
                               0,
                               stream>>>(static_cast<DT *>(m->devQKVProjArray),
                                         static_cast<DT *>(m->queryTmp),
                                         static_cast<DT *>(m->keyCache),
                                         static_cast<DT *>(m->valueCache),
                                         m->token_infos,
                                         max_num_pages,
                                         m->num_q_heads,
                                         m->num_kv_heads,
                                         m->qProjSize,
                                         num_new_tokens);
}

template <typename DT>
__global__ void produce_output_kernel(DT const *input_ptr,
                                      DT *output_ptr,
                                      int parallelism) {
  CUDA_KERNEL_LOOP(idx, parallelism) {
    output_ptr[idx] = static_cast<DT>(input_ptr[idx]);
  }
}

template <typename DT>
void produce_output(IncMultiHeadSelfAttentionMeta const *m,
                    BatchConfig const *bc,
                    DT *output_ptr,
                    cudaStream_t stream) {
  int const num_tokens = bc->num_active_tokens();
  if (num_tokens == 0) {
    return;
  }
  int parallelism = m->vProjSize * m->num_q_heads * num_tokens;
  produce_output_kernel<<<GET_BLOCKS(parallelism),
                          min(CUDA_NUM_THREADS, parallelism),
                          0,
                          stream>>>(static_cast<DT const*>(m->outputTmp), output_ptr, parallelism);
}


} // namespace IncMultiHeadAttention
} // namespace Kernels

using namespace Kernels::IncMultiHeadAttention;

template __global__ void Kernels::IncMultiHeadAttention::apply_position_bias_qkprd(float *input_ptr,
                                                  int num_tokens,
                                                  int num_total_tokens,
                                                  int num_heads,
                                                  int global_num_q_heads,
                                                  int shard_id);
template __global__ void Kernels::IncMultiHeadAttention::apply_position_bias_qkprd(half *input_ptr,
                                                  int num_tokens,
                                                  int num_total_tokens,
                                                  int num_heads,
                                                  int global_num_q_heads,
                                                  int shard_id);

template void Kernels::IncMultiHeadAttention::update_qkv_in_batch<float>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    cudaStream_t stream);

template void Kernels::IncMultiHeadAttention::update_qkv_in_batch<half>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    cudaStream_t stream);

template void Kernels::IncMultiHeadAttention::produce_output<float>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    float *output_ptr,
    cudaStream_t stream);

template void Kernels::IncMultiHeadAttention::produce_output<half>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    half *output_ptr,
    cudaStream_t stream);


}; // namespace FlexFlow
